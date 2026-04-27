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
class DEBRIS_Train(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        # --- 改进：借鉴 KLSG 写法，从 yaml 读取路径 ---
        self.dataset_path_from_yaml = kwargs.pop('dataset_path', None)
        super().__init__(*args, **kwargs)

    def _scan(self):
        if self.dataset_path_from_yaml is not None:
            self.dataset_path = self.dataset_path_from_yaml
        else:
            # 默认路径
            self.dataset_path = os.path.join(self.dataset_dir, 'DEBRIS', 'train_dataset')
        
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
class DEBRIS_Train_plus(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        # --- 改进：借鉴 KLSG 写法，从 yaml 读取路径 ---
        self.dataset_path_from_yaml = kwargs.pop('dataset_path', None)
        super().__init__(*args, **kwargs)

    def _scan(self):
        if self.dataset_path_from_yaml is not None:
            self.dataset_path = self.dataset_path_from_yaml
        else:
            # 默认路径
            self.dataset_path = os.path.join(self.dataset_dir, 'DEBRIS', 'test_dataset')
        
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
class DEBRIS_Test(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        # --- 改进：借鉴 KLSG 写法，从 yaml 读取路径 ---
        self.dataset_path_from_yaml = kwargs.pop('dataset_path', None)
        super().__init__(*args, **kwargs)

    def _scan(self):
        if self.dataset_path_from_yaml is not None:
            self.dataset_path = self.dataset_path_from_yaml
        else:
            # 默认路径
            self.dataset_path = os.path.join(self.dataset_dir, 'DEBRIS', 'test_dataset')
        
        assert os.path.exists(self.dataset_path), 'There is no dataset %s' % self.dataset_path

        for root, dirs, files in os.walk(self.dataset_path):
            self.img_paths = files
            self.img_paths.sort()
            break

        self.xml_folder = os.path.join(self.dataset_dir, 'DEBRIS', 'test_annotation')

        print('fetch {} samples for test'.format(len(self.img_paths)))

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]
        noisy_img = self._load_img(os.path.join(self.dataset_path, file_name), as_gray=True)
        noisy_img = noisy_img.unsqueeze(0)
        _, _, h, w = noisy_img.shape
        if h < 196:
            noisy_img = F.pad(noisy_img, (0, 0, 0, 196 - h), mode='reflect')
        if w < 196:
            noisy_img = F.pad(noisy_img, (0, 196 - w, 0, 0), mode='reflect')
        noisy_img = noisy_img.squeeze(0)

        return {'real_noisy': noisy_img, 'file_name': file_name[:-4]}


@regist_dataset
class prep_DEBRIS_Train(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        # --- 改进：借鉴 KLSG 写法，从 yaml 读取路径 ---
        self.dataset_path_from_yaml = kwargs.pop('dataset_path', None)
        super().__init__(*args, **kwargs)

    def _scan(self):
        if self.dataset_path_from_yaml is not None:
            self.dataset_path = self.dataset_path_from_yaml
        else:
            # 默认路径
            if self.crop_size[0] == 160:
                self.dataset_path = os.path.join(self.dataset_dir, 'prep/DEBRIS_Train')
            # 你可能需要为其他 crop_size 提供一个 else 路径
            
        assert os.path.exists(self.dataset_path), 'There is no dataset %s' % self.dataset_path
        
        self.sub_folder = 'RN'
        
        # --- 改进：更明确的路径检查 ---
        rn_path = os.path.join(self.dataset_path, self.sub_folder)
        assert os.path.exists(rn_path), 'There is no %s folder in %s' % (self.sub_folder, self.dataset_path)

        for root, _, files in os.walk(rn_path):
            self.img_paths = files
            self.img_paths.sort()
            break

        print('fetch {} samples for training'.format(len(self.img_paths)))

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]
        noisy_img = self._load_img(os.path.join(self.dataset_path, self.sub_folder, file_name), as_gray=True)
        return {'real_noisy': noisy_img}


# ⬇️ ⬇️ ⬇️ 
# --- 新增：你需要的预处理验证集类 ---
# ⬇️ ⬇️ ⬇️
@regist_dataset
class prep_DEBRIS_Val(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        # --- 借鉴 KLSG 写法，从 yaml 读取路径 ---
        self.dataset_path_from_yaml = kwargs.pop('dataset_path', None)
        super().__init__(*args, **kwargs)

    def _scan(self):
        if self.dataset_path_from_yaml is not None:
            self.dataset_path = self.dataset_path_from_yaml
        else:
            # 默认路径
            # 假设验证集和训练集使用相同的 crop_size 逻辑
            if self.crop_size[0] == 160: 
                self.dataset_path = os.path.join(self.dataset_dir, 'prep/DEBRIS_Val') # <-- 更改为 Val 路径
            
        assert os.path.exists(self.dataset_path), 'There is no dataset %s' % self.dataset_path
        
        # 假设验证集预处理后也放在 'RN' 子文件夹中
        self.sub_folder = 'RN'
        
        rn_path = os.path.join(self.dataset_path, self.sub_folder)
        assert os.path.exists(rn_path), 'There is no %s folder in %s' % (self.sub_folder, self.dataset_path)

        for root, _, files in os.walk(rn_path):
            self.img_paths = files
            self.img_paths.sort()
            break

        print('fetch {} samples for validation'.format(len(self.img_paths))) # <-- 更改为 validation

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]
        noisy_img = self._load_img(os.path.join(self.dataset_path, self.sub_folder, file_name), as_gray=True)
        # return {'real_noisy': noisy_img}
        return {'real_noisy': noisy_img, 'file_name': file_name[:-4]}
# ⬇️ ⬇️ ⬇️ 
# --- 把这个新类添加到你的 DEBRIS.py 文件中 ---
# ⬇️ ⬇️ ⬇️
@regist_dataset
class DEBRIS_Val(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        # --- 这部分允许从 yaml 配置文件加载路径 ---
        self.dataset_path_from_yaml = kwargs.pop('dataset_path', None)
        super().__init__(*args, **kwargs)

    def _scan(self):
        if self.dataset_path_from_yaml is not None:
            self.dataset_path = self.dataset_path_from_yaml
        else:
            # --- 默认路径：我使用了你刚刚提供的【精确路径】 ---
            self.dataset_path = "/root/autodl-tmp/SEGSID-main/dataset/DEBRIS/val_dataset"
        
        # 这行代码会检查上面的路径是否存在
        assert os.path.exists(self.dataset_path), 'There is no dataset %s' % self.dataset_path

        for root, dirs, files in os.walk(self.dataset_path):
            self.img_paths = files
            self.img_paths.sort()
            break
        print('fetch {} samples for val (source)'.format(len(self.img_paths)))

    def _load_data(self, data_idx):
        # (这个逻辑从 DEBRIS_Test 复制而来，确保 padding 填充方式一致)
        file_name = self.img_paths[data_idx]
        noisy_img = self._load_img(os.path.join(self.dataset_path, file_name), as_gray=True)
        noisy_img = noisy_img.unsqueeze(0)
        _, _, h, w = noisy_img.shape
        if h < 196:
            noisy_img = F.pad(noisy_img, (0, 0, 0, 196 - h), mode='reflect')
        if w < 196:
            noisy_img = F.pad(noisy_img, (0, 196 - w, 0, 0), mode='reflect')
        noisy_img = noisy_img.squeeze(0)
        return {'real_noisy': noisy_img, 'file_name': file_name[:-4]}
# ⬆️ ⬆️ ⬆️ 
# --- 新类的结尾 ---
# ⬆️ ⬆️ ⬆️