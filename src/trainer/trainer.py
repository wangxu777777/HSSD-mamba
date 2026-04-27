# fileName: src/trainer/trainer.py (FIXED SYNTAX ERROR)
import os
import datetime

import torch

from . import regist_trainer
from .base import BaseTrainer
from ..model import get_model_class

# --- 步骤 1: 使用正确的相对路径导入指标 ---
try:
    from ..metrics import calculate_ssi, calculate_enl
except ImportError:
    print("CRITICAL ERROR: 'metrics.py' not found in 'src/' directory.")
    # (定义占位符以防万一)
    def calculate_ssi(*args): return 0.0
    def calculate_enl(*args): return 0.0
# --- 修改结束 ---


@regist_trainer
class Trainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @torch.no_grad()
    def test(self):
        ''' initialization test setting '''
        # initialization
        dataset_load = (self.cfg['test_img'] is None) and (self.cfg['test_dir'] is None)
        self._before_test(dataset_load=dataset_load)

        # -- [ TEST Single Image ] -- #
        if self.cfg['test_img'] is not None:
            print('test img mode')
            self.test_img(self.cfg['test_img'])
            exit()
        # -- [ TEST Image Directory ] -- #
        elif self.cfg['test_dir'] is not None:
            print('test dir mode')
            self.test_dir(self.cfg['test_dir'])
            exit()
        # -- [ TEST Dataset ] -- #
        # (这现在是第 32 行, 语法正确)
        else:
            print('test dataset mode')
            # set image save path
            for i in range(60):
                test_time = datetime.datetime.now().strftime('%m-%d-%H-%M') + '-%02d' % i
                img_save_path = 'img/test_%s_%03d_%s' % (self.cfg['test']['dataset'], self.epoch, test_time)
                if not self.file_manager.is_dir_exist(img_save_path): break
            
            # --- 步骤 2: 修改 'test' 方法以解包 5 个值 ---
            psnr, ssim, brisque, ssi, enl = self.test_dataloader_process(dataloader=self.test_dataloader['dataset'],
                                                                 add_con=0. if not 'add_con' in self.test_cfg else self.test_cfg['add_con'],
                                                                 floor=False if not 'floor' in self.test_cfg else self.test_cfg['floor'],
                                                                 img_save_path=img_save_path,
                                                                 img_save=self.test_cfg['save_image'])
            
            # print out result as filename
            result_filename = "" # 初始化
            
            if brisque is not None: 
                print(f"BRISQUE: {brisque:.6f} (Lower is Better)")
                result_filename += f"_brisque-{brisque:.3f}"
            if psnr is not None:
                result_filename = f"_psnr-{psnr:.2f}" + result_filename
            if ssim is not None:
                result_filename = f"_ssim-{ssim:.3f}" + result_filename

            # --- 在这里添加 SSI 和 ENL 的日志记录 ---
            if ssi is not None:
                print(f"SSI: {ssi:.4f} (Lower is Better)")
                result_filename += f"_ssi-{ssi:.4f}"
            if enl is not None:
                print(f"ENL: {enl:.2f} (Higher is Better)")
                result_filename += f"_enl-{enl:.2f}"
            # --- 修改结束 ---

            if result_filename: # 确保至少有一个指标
                with open(os.path.join(self.file_manager.get_dir(img_save_path),
                  result_filename + ".result"), 'w') as f:
                    if psnr is not None:
                        f.write(f"PSNR: {psnr:.4f}\n")
                    if ssim is not None:
                        f.write(f"SSIM: {ssim:.4f}\n")
                    if brisque is not None:
                        f.write(f"BRISQUE: {brisque:.6f}\n")
                    # --- 在这里添加 SSI 和 ENL 的文件写入 ---
                    if ssi is not None:
                        f.write(f"SSI: {ssi:.4f}\n")
                    if enl is not None:
                        f.write(f"ENL: {enl:.2f}\n")
                    # --- 修改结束 ---


    @torch.no_grad()
    def validation(self):
        # set denoiser
        self._set_denoiser()

        # wrapping denoiser w/ crop test
        if 'crop' in self.cfg['validation']:
            # (warning) self_ensemble cannot be applied with multi-input model
            denoiser_fn = self.denoiser
            self.denoiser = lambda *input_data: self.crop_test(denoiser_fn, *input_data, size=self.cfg['validation']['crop'], overlap=20)
            print('Enable validation crop with size', self.cfg['validation']['crop'], '\n')

        # make directories for image saving
        img_save_path = 'img/val_%03d' % self.epoch
        self.file_manager.make_dir(img_save_path)

        # --- 步骤 3: 修改 'validation' 方法以解包 5 个值 ---
        # test_dataloader_process 现在返回 5 个值
        psnr, ssim, current_brisque, current_ssi, current_enl = self.test_dataloader_process(
                                                                dataloader=self.val_dataloader['dataset'],
                                                                add_con=0. if not 'add_con' in self.val_cfg else self.val_cfg['add_con'],
                                                                floor=False if not 'floor' in self.val_cfg else self.val_cfg['floor'],
                                                                img_save_path=img_save_path,
                                                                img_save=self.val_cfg['save_image'])
        
        # 将所有验证分数返回给 _after_epoch (在 BaseTrainer 中)
        # BaseTrainer._after_epoch 期望接收 3 个值
        return current_brisque, current_ssi, current_enl
        # --- 修改结束 ---

    def _set_module(self):
        # ... (此函数保持不变) ...
        module = {}
        if self.cfg['model']['kwargs'] is None:
            module['denoiser'] = get_model_class(self.cfg['model']['type'])()
        else:
            module['denoiser'] = get_model_class(self.cfg['model']['type'])(**self.cfg['model']['kwargs'])
        return module

    def _set_optimizer(self):
        # ... (此函数保持不变) ...
        optimizer = {}
        for key in self.module:
            optimizer[key] = self._set_one_optimizer(opt=self.train_cfg['optimizer'],
                                                     parameters=self.module[key].parameters(),
                                                     lr=float(self.train_cfg['init_lr']))
        return optimizer

    def _forward_fn(self, module, loss, data):
        # ... (此函数保持不变) ...
        # forward
        if self.cfg['model_type'] == 'only_denoise':
            input_data = [data['dataset'][arg] for arg in self.cfg['model_input']]
            denoised_img = module['denoiser'](*input_data)
            model_output = {'recon': denoised_img}
            losses, tmp_info = loss(input_data, model_output, data['dataset'], module, ratio=(self.epoch - 1 + (self.iter - 1) / self.max_iter) / self.max_epoch)  # get losses
        else:
            print('error when forward_fn!')

        return losses, tmp_info