# fileName: src/metrics.py
import torch
import numpy as np # 确保导入 numpy

def calculate_ssi(img_denoised, img_noisy):
    """
    计算散斑抑制指数 (Speckle Suppression Index, SSI)。
    
    比较去噪后图像与原始带噪图像的变异系数 (CV = std / mean)。
    SSI = CV_denoised / CV_noisy
    
    指标含义: SSI 越低，散斑抑制效果越好。
    
    参数:
    img_denoised (torch.Tensor): 模型的去噪输出图像。
    img_noisy (torch.Tensor): 原始的带噪输入图像。
    
    返回:
    float: SSI 分数。
    """
    # 确保是浮点数以便计算
    img_denoised = img_denoised.float()
    img_noisy = img_noisy.float()
    
    # 加上一个极小值 (epsilon) 来防止除以零
    epsilon = 1e-8

    # --- 计算去噪图像的变异系数 (CV) ---
    mean_denoised = torch.mean(img_denoised)
    std_denoised = torch.std(img_denoised)
    cv_denoised = std_denoised / (mean_denoised + epsilon)

    # --- 计算带噪图像的变异系数 (CV) ---
    mean_noisy = torch.mean(img_noisy)
    std_noisy = torch.std(img_noisy)
    cv_noisy = std_noisy / (mean_noisy + epsilon)

    # --- 计算 SSI ---
    ssi = cv_denoised / (cv_noisy + epsilon)
    
    # .item() 将张量转换为 Python 数字，并释放 GPU 显存
    ssi_value = ssi.item()
    if np.isnan(ssi_value) or np.isinf(ssi_value):
        return 0.0 # 返回一个安全值
    return ssi_value


def calculate_enl(img_denoised):
    """
    计算等效视数 (Equivalent Number of Looks, ENL)。
    
    ENL = (mean / std)^2
    这个函数在 *整张图像* 上计算 ENL。
    
    指标含义: ENL 越高，图像越平滑 (散斑抑制得越好)。
    
    参数:
    img_denoised (torch.Tensor): 模型的去噪输出图像。
    
    返回:
    float: ENL 分数。
    """
    img_denoised = img_denoised.float()
    epsilon = 1e-8

    # (B, C, H, W) -> 计算整个批次和图像的统计量
    mean = torch.mean(img_denoised)
    std = torch.std(img_denoised)
    
    enl = (mean**2) / (std**2 + epsilon)
    
    enl_value = enl.item()
    if np.isnan(enl_value) or np.isinf(enl_value):
        return 0.0 # 返回一个安全值
    return enl_value