import numpy as np
from skimage import color, util
from scipy.ndimage import gaussian_filter
from scipy import linalg

# ⚠️ 示例 PRIS 均值和协方差，需要用论文提供的真实参数替换
_mu_pris = np.array([0.0, 0.0])
_cov_pris = np.eye(2)

def _mean_filter(image, size=7):
    return gaussian_filter(image, sigma=7/6, truncate=(size - 1) / (2 * (7/6)))

def _extract_subpatches(img, patch_size=8):
    H, W = img.shape
    patches = []
    for i in range(0, H - patch_size + 1):
        for j in range(0, W - patch_size + 1):
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(patch.flatten())
    return np.array(patches)

def compute_niqe_features(img):
    mu = _mean_filter(img)
    mu_sq = mu * mu
    sigma = np.sqrt(np.abs(_mean_filter(img * img) - mu_sq))
    structdis = (img - mu) / (sigma + 1)
    patches = _extract_subpatches(structdis)
    mu = np.mean(patches, axis=1)
    sigma = np.var(patches, axis=1)
    feats = np.column_stack((mu, sigma))
    return feats

def niqe(img, patch_size=8):
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = util.img_as_float32(img)

    feats = compute_niqe_features(img)
    mu_dist = np.mean(feats, axis=0)
    cov_dist = np.cov(feats.T)

    invcov = linalg.inv((_cov_pris + cov_dist) / 2.0)
    niqe_score = np.sqrt((_mu_pris - mu_dist).dot(invcov).dot((_mu_pris - mu_dist).T))
    return float(niqe_score)
