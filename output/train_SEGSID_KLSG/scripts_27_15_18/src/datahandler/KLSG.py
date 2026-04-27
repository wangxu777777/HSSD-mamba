import os
import cv2
import h5py
import numpy as np
import math
import torch.nn.functional as F

from src.datahandler.denoise_dataset import DenoiseDataSet
from . import regist_dataset
import xml.etree.ElementTree as ET


@regist_dataset
class KLSG_Train(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        # --- 修正：安全地接收 dataset_path ---
        self.dataset_path_from_yaml = kwargs.pop('dataset_path', None)
        super().__init__(*args, **kwargs)

    def _scan(self):
        # 优先使用 YAML 路径
        if self.dataset_path_from_yaml is not None:
            self.dataset_path = self.dataset_path_from_yaml
        else:
            # 否则使用默认路径
            self.dataset_path = os.path.join(self.dataset_dir, 'KLSG', 'train_dataset')
        
        assert os.path.exists(self.dataset_path), 'There is no dataset %s' % self.dataset_path

        for root, dirs, files in os.walk(self.dataset_path):
            self.img_paths = files
            self.img_paths.sort()
            break

        print('fetch {} samples for training'.format(len(self.img_paths)))

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]
        noisy_img = self._load_img(os.path.join(self.dataset_path, file_name), as_gray=True)

        return {'real_noisy': noisy_img, 'file_name': file_name[:-4]}


@regist_dataset
class KLSG_Train_plus(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        # --- 修正：安全地接收 dataset_path ---
        self.dataset_path_from_yaml = kwargs.pop('dataset_path', None)
        super().__init__(*args, **kwargs)

    def _scan(self):
        if self.dataset_path_from_yaml is not None:
            self.dataset_path = self.dataset_path_from_yaml
        else:
            self.dataset_path = os.path.join(self.dataset_dir, 'KLSG', 'test_dataset')
        
        assert os.path.exists(self.dataset_path), 'There is no dataset %s' % self.dataset_path

        for root, dirs, files in os.walk(self.dataset_path):
            self.img_paths = files
            self.img_paths.sort()
            break

        print('fetch {} samples for training'.format(len(self.img_paths)))

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]
        noisy_img = self._load_img(os.path.join(self.dataset_path, file_name), as_gray=True)

        return {'real_noisy': noisy_img, 'file_name': file_name[:-4]}


@regist_dataset
class KLSG_Test(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        # --- 修正：安全地接收 dataset_path ---
        self.dataset_path_from_yaml = kwargs.pop('dataset_path', None)
        super().__init__(*args, **kwargs)

    def _scan(self):
        if self.dataset_path_from_yaml is not None:
            self.dataset_path = self.dataset_path_from_yaml
        else:
            self.dataset_path = os.path.join(self.dataset_dir, 'KLSG', 'test_dataset')
        
        assert os.path.exists(self.dataset_path), 'There is no dataset %s' % self.dataset_path

        for root, dirs, files in os.walk(self.dataset_path):
            self.img_paths = files
            self.img_paths.sort()
            break

        print('fetch {} samples for test'.format(len(self.img_paths)))

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]
        noisy_img = self._load_img(os.path.join(self.dataset_path, file_name), as_gray=True)
        noisy_img = noisy_img.unsqueeze(0)
        _, _, h, w = noisy_img.shape
        if h < 196:
            noisy_img = F.pad(noisy_img, (0, 0, 0, 196 - h), mode='constant', value=0)
        if w < 196:
            noisy_img = F.pad(noisy_img, (0, 196 - w, 0, 0), mode='constant', value=0)         
        noisy_img = noisy_img.squeeze(0)
        return {'real_noisy': noisy_img, 'file_name': file_name[:-4]}


# ⬇️ ⬇️ ⬇️ 这是您缺失的 KLSG_Val 类 ⬇️ ⬇️ ⬇️
@regist_dataset
class KLSG_Val(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        # --- 修正：安全地接收 dataset_path ---
        self.dataset_path_from_yaml = kwargs.pop('dataset_path', None)
        super().__init__(*args, **kwargs)

    def _scan(self):
        # 优先使用 YAML 路径 (您在 YAML 中指定了 'val_dataset' 路径)
        if self.dataset_path_from_yaml is not None:
            self.dataset_path = self.dataset_path_from_yaml
        else:
            # 否则使用默认路径
            self.dataset_path = os.path.join(self.dataset_dir, 'KLSG', 'val_dataset') # 备用路径
        
        assert os.path.exists(self.dataset_path), 'There is no dataset %s' % self.dataset_path

        for root, dirs, files in os.walk(self.dataset_path):
            self.img_paths = files
            self.img_paths.sort()
            break

        print('fetch {} samples for validation'.format(len(self.img_paths))) # <-- 已改为 'validation'

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]
        noisy_img = self._load_img(os.path.join(self.dataset_path, file_name), as_gray=True)
        
        # (复制 KLSG_Test 的 padding 逻辑, 验证集和测试集应保持一致)
        noisy_img = noisy_img.unsqueeze(0)
        _, _, h, w = noisy_img.shape
        if h < 196:
            noisy_img = F.pad(noisy_img, (0, 0, 0, 196 - h), mode='constant', value=0)
        if w < 196:
            noisy_img = F.pad(noisy_img, (0, 196 - w, 0, 0), mode='constant', value=0)
        noisy_img = noisy_img.squeeze(0)

        return {'real_noisy': noisy_img, 'file_name': file_name[:-4]}
# ⬆️ ⬆️ ⬆️ 您需要添加上面这个新类 ⬆️ ⬆️ ⬆️


@regist_dataset
class prep_KLSG_Train(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        # --- 修正：安全地接收 dataset_path ---
        self.dataset_path_from_yaml = kwargs.pop('dataset_path', None)
        super().__init__(*args, **kwargs)

    def _scan(self):
        if self.dataset_path_from_yaml is not None:
            self.dataset_path = self.dataset_path_from_yaml
        else:
            self.dataset_path = os.path.join(self.dataset_dir, 'prep/KLSG_Train')

        assert os.path.exists(self.dataset_path), 'There is no dataset %s' % self.dataset_path
        
        # 路径中包含 'RN' (Real Noisy)
        rn_path = os.path.join(self.dataset_path, 'RN')
        assert os.path.exists(rn_path), 'There is no RN folder in %s' % self.dataset_path
        
        for root, _, files in os.walk(rn_path):
            self.img_paths = files
            self.img_paths.sort()
            break

        print('fetch {} samples for training'.format(len(self.img_paths)))

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]
        noisy_img = self._load_img(os.path.join(self.dataset_path, 'RN', file_name), as_gray=True)
        return {'real_noisy': noisy_img}