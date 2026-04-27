import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as P
import time
import torchvision.models as models

# 假设这些是你项目中的原始依赖
from . import regist_model
from .RFS import RF_scale

# --- Mamba 及其他依赖的健壮导入 ---
try:
    from .pixel_shuffle import pixel_shuffle_up_sampling, pixel_shuffle_down_sampling
except ImportError:
    print("CRITICAL ERROR: pixel_shuffle.py not found.")
    pixel_shuffle_down_sampling = None
    pixel_shuffle_up_sampling = None

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba-ssm package not found. Mamba modules will fail if initialized.")
    Mamba = None
try:
    from einops import rearrange
except ImportError:
    print("Warning: einops package not found.")
    rearrange = None
    
def get_inverse_indices(indices):
    """(Mamba 依赖)"""
    inv_indices = torch.empty_like(indices, device=indices.device)
    inv_indices[indices] = torch.arange(len(indices), device=indices.device)
    return inv_indices
# --- 依赖结束 ---


# --- [Wrapper] 用于作图优化的包装类 ---
class SemanticRefinementStage(nn.Module):
    def __init__(self, si_modules_list, body2_modules_seq):
        super().__init__()
        # 使用 ModuleList 存储 SI blocks，因为 forward 需要传额外的 f_semantic 参数
        self.si_blocks = nn.ModuleList(si_modules_list)
        # Body2 是标准的 Sequential
        self.body2 = body2_modules_seq

    def forward(self, x, f_semantic):
        # 1. 执行所有的 SI Block
        for si_block in self.si_blocks:
            x = si_block(x, f_semantic)
        
        # 2. 执行 Body2 (作为收尾)
        x = self.body2(x)
        
        return x


# --- [Task 3] 多层 ResNet 语义提取器 ---
class MultiLayerResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        print(f'[{True}]    Init Multi-Layer ResNet backbone as sematic encoder')
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 1. 分解 ResNet
        self.layer0 = nn.Sequential(*list(resnet18.children())[0:4]) # conv1, bn1, relu, maxpool
        self.layer1 = resnet18.layer1 # -> 64 C
        self.layer2 = resnet18.layer2 # -> 128 C
        self.layer3 = resnet18.layer3 # -> 256 C
        self.layer4 = resnet18.layer4 # -> 512 C
        
        # 2. 全局池化层
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. 投影层 (全部统一到 512 C)
        self.proj1 = nn.Conv2d(64, 512, 1)  # 1x1 卷积
        self.proj2 = nn.Conv2d(128, 512, 1) # 1x1 卷积
        self.proj3 = nn.Conv2d(256, 512, 1) # 1x1 卷积
        self.proj4 = nn.Identity()         # layer4 已经是 512
        
        self.norm = nn.LayerNorm([512, 1, 1])

    def forward(self, x):
        # x shape: [B, C, H_down, W_down]
        
        # 提取多层特征
        x0 = self.layer0(x)
        x1 = self.layer1(x0) 
        x2 = self.layer2(x1) 
        x3 = self.layer3(x2) 
        x4 = self.layer4(x3) 
        
        # 全局池化
        p1 = self.pool(x1)
        p2 = self.pool(x2)
        p3 = self.pool(x3)
        p4 = self.pool(x4)
        
        # 投影与融合
        f1 = self.proj1(p1)
        f2 = self.proj2(p2)
        f3 = self.proj3(p3)
        f4 = self.proj4(p4)
        
        # 融合后的特征
        fused = f1 + f2 + f3 + f4
        fused = self.norm(fused) # [B, 512, 1, 1]
        
        return fused


@regist_model
class HSSDmamba(nn.Module):
    def __init__(self, pd_a=5, pd_b=2, pd_pad=2, R3=True, R3_T=8, R3_p=0.16,
                 max_epoch=128, sematic_type=None,
                 in_ch=3, bsn_base_ch=128,
                 bsn_num_module=9, is_refine=False):
        super().__init__()

        # network hyper-parameters
        self.pd_a = pd_a
        self.pd_b = pd_b
        self.pd_pad = pd_pad
        self.R3 = R3
        self.R3_T = R3_T
        self.R3_p = R3_p
        self.max_epoch = max_epoch
        self.is_refine = is_refine
        self.bsn = HSSDmamba_Model(in_ch=in_ch, out_ch=in_ch, base_ch=bsn_base_ch,
                                num_module=bsn_num_module, sematic_type=sematic_type)

    def forward(self, img, pd=None):
        if pd is None:
            pd = self.pd_a

        img_denoised = self.bsn(img, pd, pad=self.pd_pad)
        return img_denoised

    def denoise(self, x):  # Denoise process for inference.
        b, c, h, w = x.shape

        assert h % self.pd_b == 0, f'but w is {h}'
        assert w % self.pd_b == 0, f'but w is {w}'

        temp_pd_b = self.pd_b

        """forward process with inference pd factor = 2"""
        img_denoised = self.forward(img=x, pd=temp_pd_b)

        if not self.R3:
            return img_denoised
        else:
            """ with R3 strategy """
            denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(x)
                mask = indice < self.R3_p
                tmp_input = torch.clone(img_denoised).detach()
                tmp_input[mask] = x[mask]
                denoised[..., t] = self.bsn(
                    tmp_input, pd=temp_pd_b, is_refine=self.is_refine, pad=self.pd_pad)

            denoised = torch.mean(denoised, dim=-1)
            return denoised


class HSSDmamba_Model(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9,
                 head_ch=24, sematic_type=None):
        super().__init__()
        assert base_ch % 2 == 0, "base channel should be divided with 2"

        self.sematic_type = sematic_type

        ly = []
        ly += [nn.Conv2d(in_ch, head_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.in_ch = in_ch
        self.head = nn.Sequential(*ly)

        # 使用修改后的 Branch
        self.branch_11 = Branch(branch_type='11', dilated_factor=2,
                                head_ch=head_ch, out_ch=base_ch, num_module=num_module)
        self.branch_21 = Branch(branch_type='21', dilated_factor=3,
                                head_ch=head_ch, out_ch=base_ch, num_module=num_module)

        ly = []
        ly += [nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, out_ch, kernel_size=1)]
        self.tail = nn.Sequential(*ly)

        if sematic_type == 'ResNet':
            # 使用 Task 3 的 Encoder
            self.semantic_encoder = MultiLayerResNetEncoder()
            print('Warning Please check the norm mode of sematic branch')
        else:
            print(f'Warning: sematic_type is {sematic_type}, not using ResNet.')
            self.semantic_encoder = None

    def forward(self, x, pd=5, is_refine=False, pad=2):
        if is_refine is True:
            pd = 1

        if self.sematic_type == 'ResNet' and self.semantic_encoder is not None:
            x_norm = 2 * (x / 255 - 0.5)
            if is_refine is False:
                x_norm = pixel_shuffle_down_sampling(x_norm, f=pd, pad=pad)
            else:
                x_norm = F.pad(x_norm, (pad, pad, pad, pad))

            if self.in_ch == 1:
                f_semantic = self.semantic_encoder(x_norm.repeat(1, 3, 1, 1))
            elif self.in_ch == 3:
                f_semantic = self.semantic_encoder(x_norm)
            else:
                print('Wrong in_ch: hope 1 or 3, but get ', self.in_ch)
                f_semantic = self.semantic_encoder(
                    x_norm.repeat(1, 3, 1, 1))
        else:
            f_semantic = None

        x = self.head(x)
        br1 = self.branch_11(x, pd=pd, f_semantic=f_semantic,
                             is_refine=is_refine, pad=pad)
        br2 = self.branch_21(x, pd=pd, f_semantic=f_semantic,
                             is_refine=is_refine, pad=pad)
        x = torch.cat([br1, br2], dim=1)

        return self.tail(x)


class Branch(nn.Module):
    def __init__(self, branch_type, dilated_factor, head_ch, out_ch,
                 num_module, is_bias=True):
        super().__init__()

        padding_mode = 'zeros'
        self.branch_type = branch_type

        # 初始化 RFA (Hybrid)
        if branch_type == '11':
            self.rfa = RFA_11(head_ch, out_ch, dilated_factor=dilated_factor,
                              padding_mode=padding_mode, is_bias=is_bias)
            assert dilated_factor == 2

        elif branch_type == '21':
            self.rfa = RFA_21(head_ch, out_ch, dilated_factor=dilated_factor,
                              padding_mode=padding_mode, is_bias=is_bias)
            assert dilated_factor == 3

        # [Task 1] RFA 周围的残差 1x1 卷积
        self.residual_conv = nn.Conv2d(head_ch, out_ch, kernel_size=1, bias=is_bias)

        # [Wrapper] 准备 SI Blocks 列表
        in_ch = out_ch 
        si_list = [Semantic_Injection_block(dilated_factor, in_ch,
                                            padding_mode=padding_mode) for _ in range(num_module)]

        # [Wrapper] 准备 Body2 序列
        body2_list = []
        body2_list += [nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=is_bias)]
        body2_list += [nn.ReLU(inplace=True)]
        body2_seq = nn.Sequential(*body2_list)

        # [Wrapper] 打包 (核心修改)
        self.semantic_stage = SemanticRefinementStage(si_list, body2_seq)
        
        print('Semantic Stage Initialized (SI blocks + Body2 wrapped).')

    def forward(self, x: torch.Tensor, pd, f_semantic=None, is_refine=False,
                pad=2):
        
        # 残差连接
        identity = self.residual_conv(x)
        
        # RFA Module (Hybrid + Body1 inside)
        x_rfa = self.rfa(x, is_refine=is_refine, f=pd, pad=pad)

        # Add Residual
        x = x_rfa + identity

        # Semantic Stage (SI loop + Body2)
        x = self.semantic_stage(x, f_semantic)

        # Inverse PD
        if is_refine is False:
            x = pixel_shuffle_up_sampling(x, f=pd, pad=pad)
        else:
            if pad != 0:
                x = x[:, :, pad:-pad, pad:-pad]

        return x


class Semantic_Injection_block(nn.Module):
    def __init__(self, stride, in_ch, padding_mode='reflect'):
        super().__init__()

        self.semantic_affine = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(512, in_ch, kernel_size=1),  # semantic function
        )

        self.spatial_mix = nn.Sequential(
            # F-Conv + ReLu + 1x1 Conv
            Dilation_fork_3x3(in_ch, in_ch, kernel_size=3, stride=1, padding=stride,
                              dilation=stride, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1)
        )

    def forward(self, x, f_semantic: torch.Tensor):
        if f_semantic is None:
            y = x + self.spatial_mix(x)
        else:
            y = x + self.spatial_mix(self.semantic_injection_function(x,
                                                                    self.semantic_affine(f_semantic)))
        return y

    @staticmethod
    def semantic_injection_function(spatial_feature, semantic_feature):
        return spatial_feature * semantic_feature


# --- Mamba Core Module ---
class MambaHierarchicalBlindScan(nn.Module):
    def __init__(self, in_ch, out_ch, patch_size=8, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        self.patch_size = patch_size
        self.in_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.norm_local = nn.LayerNorm(out_ch)
        self.norm_global = nn.LayerNorm(out_ch)
        self.mamba_local = Mamba(
            d_model=out_ch, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand,
        )
        mamba_global_expand = mamba_expand // 2 if mamba_expand > 1 else 1
        self.mamba_global = Mamba(
            d_model=out_ch, d_state=mamba_d_state // 2, d_conv=mamba_d_conv, expand=mamba_global_expand,
        )
        self.out_proj = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.indices_cache = {} 
        print(f"[{self.__class__.__name__}] V18 4WAY-Scan Initialized.")

    def forward(self, x):
        B, C, H, W = x.shape
        D = self.in_proj.out_channels
        P = self.patch_size
        if P == 1:
            x_proj = self.in_proj(x)
            x_hw = x_proj.permute(0, 2, 3, 1).contiguous()
            x_flat = rearrange(x_hw, 'b h w d -> b (h w) d')
            x_wh = x_proj.permute(0, 3, 2, 1).contiguous()
            x_flat_v = rearrange(x_wh, 'b w h d -> b (w h) d')
            x_fwd = self.norm_global(x_flat)
            x_bwd = self.norm_global(torch.flip(x_flat, dims=[1]))
            x_v_fwd = self.norm_global(x_flat_v)
            x_v_bwd = self.norm_global(torch.flip(x_flat_v, dims=[1]))
            x_fwd_shifted = torch.roll(x_fwd, shifts=1, dims=1); x_fwd_shifted[:, 0, :] = 0.0
            x_bwd_shifted = torch.roll(x_bwd, shifts=1, dims=1); x_bwd_shifted[:, 0, :] = 0.0
            x_v_fwd_shifted = torch.roll(x_v_fwd, shifts=1, dims=1); x_v_fwd_shifted[:, 0, :] = 0.0
            x_v_bwd_shifted = torch.roll(x_v_bwd, shifts=1, dims=1); x_v_bwd_shifted[:, 0, :] = 0.0
            y_fwd = self.mamba_global(x_fwd_shifted)
            y_bwd = self.mamba_global(x_bwd_shifted)
            y_v_fwd = self.mamba_global(x_v_fwd_shifted)
            y_v_bwd = self.mamba_global(x_v_bwd_shifted)
            y_bwd = torch.flip(y_bwd, dims=[1])
            y_v_bwd = torch.flip(y_v_bwd, dims=[1])
            y_fwd = rearrange(y_fwd, 'b (h w) d -> b h w d', h=H, w=W)
            y_bwd = rearrange(y_bwd, 'b (h w) d -> b h w d', h=H, w=W)
            y_v_fwd = rearrange(y_v_fwd, 'b (w h) d -> b w h d', h=H, w=W).permute(0, 2, 1, 3)
            y_v_bwd = rearrange(y_v_bwd, 'b (w h) d -> b w h d', h=H, w=W).permute(0, 2, 1, 3)
            y_merged = (y_fwd + y_bwd + y_v_fwd + y_v_bwd) / 4.0
            y_out = rearrange(y_merged, 'b h w d -> b d h w')
            return self.out_proj(y_out)
            
        # P > 1 的路径
        if H < P or W < P:
            if H == 1 and W == 1:
                x_proj = self.in_proj(x); x_flat = rearrange(x_proj, 'b d h w -> b (h w) d')
                x_normed = self.norm_local(x_flat); x_mamba = self.mamba_local(x_normed) 
                x_out = rearrange(x_mamba, 'b (h w) d -> b d h w', h=H, w=W); return self.out_proj(x_out)
            else:
                x_proj = self.in_proj(x); x_flat = rearrange(x_proj, 'b d h w -> b (h w) d')
                x_normed = self.norm_local(x_flat); x_shifted = torch.roll(x_normed, shifts=1, dims=1)
                x_shifted[:, 0, :] = 0.0; x_mamba = self.mamba_local(x_shifted)
                x_out = rearrange(x_mamba, 'b (h w) d -> b d h w', h=H, w=W); return self.out_proj(x_out)
        pad_h, pad_w = 0, 0
        if H % P != 0 or W % P != 0:
            pad_h = (P - H % P) % P; pad_w = (P - W % P) % P
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H_pad, W_pad = x.shape[2], x.shape[3]
        else: H_pad, W_pad = H, W
        num_H_patches = H_pad // P; num_W_patches = W_pad // P
        x_proj = self.in_proj(x)
        x_patches_flat = rearrange(x_proj, 'b d (h p1) (w p2) -> (b h w) (p1 p2) d', p1=P, p2=P)
        key = (P, P, x.device, "snake")
        if key not in self.indices_cache:
            indices = torch.arange(P * P, device=x.device).reshape(P, P)
            indices[1::2] = torch.flip(indices[1::2], dims=[1])
            scan_indices = indices.flatten(); inv_indices = get_inverse_indices(scan_indices)
            self.indices_cache[key] = (scan_indices, inv_indices)
        local_indices, local_inv_indices = self.indices_cache[key]
        local_indices_exp = local_indices.view(1, -1, 1).expand(x_patches_flat.shape[0], -1, D)
        x_local_scanned = torch.gather(x_patches_flat, 1, local_indices_exp)
        x_local_normed = self.norm_local(x_local_scanned)
        x_local_shifted = torch.roll(x_local_normed, shifts=1, dims=1); x_local_shifted[:, 0, :] = 0.0
        y_local_scanned = self.mamba_local(x_local_shifted)
        local_inv_indices_exp = local_inv_indices.view(1, -1, 1).expand(x_patches_flat.shape[0], -1, D)
        y_local_flat = torch.gather(y_local_scanned, 1, local_inv_indices_exp)
        x_global_tokens = y_local_flat[:, 0, :]
        x_global_flat = rearrange(x_global_tokens, '(b h w) d -> b (h w) d', b=B, h=num_H_patches, w=num_W_patches)
        key_global = (num_H_patches, num_W_patches, x.device, "snake")
        if key_global not in self.indices_cache:
            indices = torch.arange(num_H_patches * num_W_patches, device=x.device).reshape(num_H_patches, num_W_patches)
            indices[1::2] = torch.flip(indices[1::2], dims=[1])
            scan_indices = indices.flatten(); inv_indices = get_inverse_indices(scan_indices)
            self.indices_cache[key_global] = (scan_indices, inv_indices)
        global_indices, global_inv_indices = self.indices_cache[key_global]
        global_indices_exp = global_indices.view(1, -1, 1).expand(B, -1, D)
        x_global_scanned = torch.gather(x_global_flat, 1, global_indices_exp)
        x_global_normed = self.norm_global(x_global_scanned)
        x_global_shifted = torch.roll(x_global_normed, shifts=1, dims=1); x_global_shifted[:, 0, :] = 0.0
        y_global_scanned = self.mamba_global(x_global_shifted)
        global_inv_indices_exp = global_inv_indices.view(1, -1, 1).expand(B, -1, D)
        y_global_flat = torch.gather(y_global_scanned, 1, global_inv_indices_exp)
        y_global_expanded = rearrange(y_global_flat, 'b (h w) d -> (b h w) 1 d', h=num_H_patches, w=num_W_patches)
        y_merged = y_local_flat + y_global_expanded
        y_out = rearrange(y_merged, '(b h w) (p1 p2) d -> b d (h p1) (w p2)',
                            b=B, h=num_H_patches, w=num_W_patches, p1=P, p2=P)
        x_out = self.out_proj(y_out)
        if pad_h > 0 or pad_w > 0:
            x_out = x_out[:, :, :H, :W]
        return x_out


# RFA modul 11
class RFA_11(nn.Module):
    """ Hybrid RFA: CNN + Mamba, body1 merged inside """
    def __init__(self, head_ch, out_ch, dilated_factor,
                 padding_mode='reflect', is_bias=True):
        super().__init__()
        self.ratio = 2 / 5
        self.pd1_flag = False
        self.padding = 5
        self.padding_mode = padding_mode
        self.shave = 5 + 3
        self.kernel_size = ks = 11

        # CNN Path
        self.d_conv = DSM_Convolution(head_ch, out_ch, kernel_size=ks, stride=1,
                                      padding=ks // 2, padding_mode=padding_mode, bias=is_bias)
        # Mamba Path
        self.normal_conv = MambaHierarchicalBlindScan(head_ch, out_ch, patch_size=1)
        
        self.is_diamond_mask = True if hasattr(self.d_conv, 'mask') else False
        if self.is_diamond_mask: print('11 d_conv get mask (CNN Path)')
        else: print('11 d_conv no mask (CNN Path)')
        print('[HYBRID RFA_11] Initialized: 1x DSM_Conv (CNN) + 1x MambaScan (Mamba)')

        self.test_d_conv = nn.Conv2d(head_ch, out_ch, kernel_size=(ks, ks),
                                     stride=(ks, ks), padding=0, bias=is_bias)

        self.f_conv = Dilation_fork_3x3(out_ch, out_ch, kernel_size=3, stride=1,
                                        dilation=dilated_factor, padding=dilated_factor, padding_mode=padding_mode,
                                        bias=is_bias)
        self.p_conv = Dilation_plus_3x3(out_ch, out_ch, kernel_size=3, stride=1,
                                        dilation=dilated_factor, padding=dilated_factor, padding_mode=padding_mode,
                                        bias=is_bias)

        self.rf_scale = RF_scale(head_ch, out_ch, kernel_size=ks, padding=ks //
                                 2, padding_mode=padding_mode, ratio=self.ratio)
        
        # [Task 2] body1 移入 RFA
        in_ch = out_ch
        body1 = []
        body1 += [nn.ReLU(inplace=True)]
        body1 += [nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=is_bias)]
        body1 += [nn.ReLU(inplace=True)]
        body1 += [nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=is_bias)]
        body1 += [nn.ReLU(inplace=True)]
        self.body1 = nn.Sequential(*body1)

    def forward(self, x: torch.Tensor, is_refine=False, f=5, pad=2):
        # Training
        if self.training:
            d_conv_x = self.d_conv(x)
            normal_conv_x = self.normal_conv(x) 
            
            d_conv_x = pixel_shuffle_down_sampling(d_conv_x, f, pad=pad)
            normal_conv_x = pixel_shuffle_down_sampling(normal_conv_x, f, pad=pad)
            
            out_x = self.f_conv(d_conv_x) + self.p_conv(normal_conv_x)
        # Testing
        else:
            if is_refine is False:
                self.rf_scale.ratio = 2 / 5
                d_conv_x, _ = self.forward_refine(x)
                normal_conv_x = self.normal_conv(x)
                
                d_conv_x = pixel_shuffle_down_sampling(d_conv_x, f, pad=pad)
                normal_conv_x = pixel_shuffle_down_sampling(normal_conv_x, f, pad=pad)
            else:
                self.rf_scale.ratio = 1 / 5
                d_conv_x, _ = self.forward_refine(x)
                normal_conv_x = self.normal_conv(x)
                
                p = pad
                d_conv_x = F.pad(d_conv_x, (p, p, p, p), mode='constant', value=0)
                normal_conv_x = F.pad(normal_conv_x, (p, p, p, p), mode='constant', value=0)

            out_x = self.f_conv(d_conv_x) + self.p_conv(normal_conv_x)
        
        # Apply body1
        out_x = self.body1(out_x)
        return out_x

    def forward_refine(self, x):
        n_GPUs_for_deformable = 1
        n_GPUs = min(n_GPUs_for_deformable, 4)
        b, c, h, w = x.shape
        top = slice(0, h // 2 + self.shave)
        bottom = slice(h - h // 2 - self.shave, h)
        left = slice(0, w // 2 + self.shave)
        right = slice(w - w // 2 - self.shave, w)
        x_chop = torch.cat([
            x[..., top, left], x[..., top, right],
            x[..., bottom, left], x[..., bottom, right]
        ])
        hole_chop = []
        assert h * w < 4 * 200000
        for i in range(0, 4, n_GPUs):
            x_c = x_chop[i:(i + n_GPUs)]
            x_offset = P.data_parallel(self.rf_scale, x_c, range(n_GPUs))
            self.test_d_conv.weight.data = self.d_conv.weight.data * self.d_conv.mask
            self.test_d_conv.bias = self.d_conv.bias
            temp_hole_chop = P.data_parallel(self.test_d_conv, x_offset, range(n_GPUs))
            del x_offset
            if not hole_chop:
                hole_chop = [c for c in temp_hole_chop.chunk(n_GPUs, dim=0)]
            else:
                hole_chop.extend(temp_hole_chop.chunk(n_GPUs, dim=0))
        assert h % 2 == 0 and w % 2 == 0
        top = slice(0, h // 2); bottom = slice(h - h // 2, h)
        left = slice(0, w // 2); right = slice(w - w // 2, w)
        right_r = slice(w // 2 - w, None); bottom_r = slice(h // 2 - h, None)
        b, c = hole_chop[0].size()[:-2]
        d_conv_x = hole_chop[0].new(b, c, h, w)
        d_conv_x[:, :, top, left] = hole_chop[0][:, :, top, left]
        d_conv_x[..., top, right] = hole_chop[1][..., top, right_r]
        d_conv_x[..., bottom, left] = hole_chop[2][..., bottom_r, left]
        d_conv_x[..., bottom, right] = hole_chop[3][..., bottom_r, right_r]
        return d_conv_x, None


# RFA modul 21
class RFA_21(nn.Module):
    """ Hybrid RFA: CNN + Mamba, body1 merged inside """
    def __init__(self, head_ch, out_ch, dilated_factor,
                 padding_mode='reflect', is_bias=True):
        super().__init__()
        self.ratio = 4 / 10
        self.padding = 10
        self.padding_mode = padding_mode
        self.shave = 10 + 2
        self.kernel_size = ks = 21

        # CNN Path
        self.d_conv = DSM_Convolution(head_ch, out_ch, kernel_size=ks, stride=1,
                                      padding=ks // 2, padding_mode=padding_mode, bias=is_bias)
        # Mamba Path
        self.normal_conv = MambaHierarchicalBlindScan(head_ch, out_ch, patch_size=1)
        
        self.is_diamond_mask = True if hasattr(self.d_conv, 'mask') else False
        if self.is_diamond_mask: print('21 d_conv get mask (CNN Path)')
        else: print('21 d_conv no mask (CNN Path)')
        print('[HYBRID RFA_21] Initialized: 1x DSM_Conv (CNN) + 1x MambaScan (Mamba)')

        self.test_d_conv = nn.Conv2d(head_ch, out_ch, kernel_size=(ks, ks),
                                     stride=(ks, ks), padding=0, bias=is_bias)

        self.f_conv = Dilation_fork_3x3(out_ch, out_ch, kernel_size=3, stride=1,
                                        dilation=dilated_factor, padding=dilated_factor, padding_mode=padding_mode,
                                        bias=is_bias)
        self.p_conv = Dilation_plus_3x3(out_ch, out_ch, kernel_size=3, stride=1,
                                        dilation=dilated_factor, padding=dilated_factor, padding_mode=padding_mode,
                                        bias=is_bias)

        self.rf_scale = RF_scale(head_ch, out_ch, kernel_size=ks, padding=ks //
                                 2, padding_mode=padding_mode, ratio=self.ratio)

        # [Task 2] body1 移入 RFA
        in_ch = out_ch
        body1 = []
        body1 += [nn.ReLU(inplace=True)]
        body1 += [nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=is_bias)]
        body1 += [nn.ReLU(inplace=True)]
        body1 += [nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=is_bias)]
        body1 += [nn.ReLU(inplace=True)]
        self.body1 = nn.Sequential(*body1)

    def forward(self, x: torch.Tensor, is_refine=False, f=5, pad=2):
        # Training
        if self.training:
            d_conv_x = self.d_conv(x)
            normal_conv_x = self.normal_conv(x)
            
            d_conv_x = pixel_shuffle_down_sampling(d_conv_x, f, pad=pad)
            normal_conv_x = pixel_shuffle_down_sampling(normal_conv_x, f, pad=pad)
            
            out_x = self.f_conv(d_conv_x) + self.p_conv(normal_conv_x)
        # Testing
        else:
            if is_refine is False:
                self.rf_scale.ratio = 2 / 5
                d_conv_x, _ = self.forward_refine(x)
                normal_conv_x = self.normal_conv(x)
                d_conv_x = pixel_shuffle_down_sampling(d_conv_x, f, pad=pad)
                normal_conv_x = pixel_shuffle_down_sampling(normal_conv_x, f, pad=pad)
            else:
                self.rf_scale.ratio = 1 / 5
                d_conv_x, _ = self.forward_refine(x)
                normal_conv_x = self.normal_conv(x)
                p = pad
                d_conv_x = F.pad(d_conv_x, (p, p, p, p), mode='constant', value=0)
                normal_conv_x = F.pad(normal_conv_x, (p, p, p, p), mode='constant', value=0)

            out_x = self.f_conv(d_conv_x) + self.p_conv(normal_conv_x)
            
        # Apply body1
        out_x = self.body1(out_x)
        return out_x

    def forward_refine(self, x):
        n_GPUs_for_deformable = 1
        n_GPUs = min(n_GPUs_for_deformable, 4)
        b, c, h, w = x.shape
        top = slice(0, h // 2 + self.shave)
        bottom = slice(h - h // 2 - self.shave, h)
        left = slice(0, w // 2 + self.shave)
        right = slice(w - w // 2 - self.shave, w)
        x_chop = torch.cat([
            x[..., top, left], x[..., top, right],
            x[..., bottom, left], x[..., bottom, right]
        ])
        hole_chop = []
        assert h * w < 4 * 200000
        for i in range(0, 4, n_GPUs):
            x_c = x_chop[i:(i + n_GPUs)]
            x_offset = P.data_parallel(self.rf_scale, x_c, range(n_GPUs))
            self.test_d_conv.weight.data = self.d_conv.weight.data * self.d_conv.mask
            self.test_d_conv.bias = self.d_conv.bias
            temp_hole_chop = P.data_parallel(self.test_d_conv, x_offset, range(n_GPUs))
            del x_offset
            if not hole_chop:
                hole_chop = [c for c in temp_hole_chop.chunk(n_GPUs, dim=0)]
            else:
                hole_chop.extend(temp_hole_chop.chunk(n_GPUs, dim=0))
        assert h % 2 == 0 and w % 2 == 0
        top = slice(0, h // 2); bottom = slice(h - h // 2, h)
        left = slice(0, w // 2); right = slice(w - w // 2, w)
        right_r = slice(w // 2 - w, None); bottom_r = slice(h // 2 - h, None)
        b, c = hole_chop[0].size()[:-2]
        d_conv_x = hole_chop[0].new(b, c, h, w)
        d_conv_x[:, :, top, left] = hole_chop[0][:, :, top, left]
        d_conv_x[..., top, right] = hole_chop[1][..., top, right_r]
        d_conv_x[..., bottom, left] = hole_chop[2][..., bottom_r, left]
        d_conv_x[..., bottom, right] = hole_chop[3][..., bottom_r, right_r]
        return d_conv_x, None


# D-Conv
class DSM_Convolution(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        center = kH // 2
        for i in range(kH):
            for j in range(kW):
                if abs(i - center) + abs(j - center) <= 5:
                    self.mask[:, :, i, j] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

# P-Conv
class Dilation_plus_3x3(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        self.mask[:, :, 0, 1] = 1
        self.mask[:, :, 1, 0] = 1
        self.mask[:, :, 1, 2] = 1
        self.mask[:, :, 2, 1] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

# F-Conv
class Dilation_fork_3x3(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, 0, 1] = 0
        self.mask[:, :, 1, 0] = 0
        self.mask[:, :, 1, 2] = 0
        self.mask[:, :, 2, 1] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

# --- 权重加载助手 ---
def load_compatible_weights(model, weight_path, device='cuda'):
    """
    用于加载旧权重到包含 Wrapper 的新模型中。
    自动处理 'si_blocks' -> 'semantic_stage.si_blocks' 的重命名。
    """
    print(f"Loading weights compatible mode: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)
    old_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    
    for k, v in old_state_dict.items():
        # 映射规则
        if 'si_blocks' in k and 'semantic_stage' not in k:
            new_k = k.replace('si_blocks', 'semantic_stage.si_blocks')
        elif 'body2' in k and 'semantic_stage' not in k:
            new_k = k.replace('body2', 'semantic_stage.body2')
        else:
            new_k = k
        new_state_dict[new_k] = v
        
    model.load_state_dict(new_state_dict)
    print("✅ Weights loaded successfully with key mapping.")
    return model

