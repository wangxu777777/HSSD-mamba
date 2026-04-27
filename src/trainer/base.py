# fileName: src/trainer/base.py (MODIFIED FOR SSI/ENL)

import os
import math
import time, datetime

import cv2
import numpy as np # 确保导入 numpy
import torch
from torch import nn
from torch import optim
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

# --- MODIFIED: 导入 piq ---
import piq
# --- END MODIFIED ---

from ..loss import Loss
from ..datahandler import get_dataset_class
from ..util.file_manager import FileManager
from ..util.logger import Logger
# 确保导入了 psnr 和 ssim 函数
from ..util.util import human_format, np2tensor, rot_hflip_img, psnr, ssim, tensor2np, imread_tensor
# from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling

# --- 步骤 1: 在这里导入指标 ---
# (假设 metrics.py 位于 src/metrics.py)
try:
    from ..metrics import calculate_ssi, calculate_enl
except ImportError:
    print("CRITICAL ERROR: 'metrics.py' not found. Please place it in the 'src/' directory.")
    # 定义占位函数以防止崩溃，但会打印错误
    def calculate_ssi(img_denoised, img_noisy):
        print("ERROR: calculate_ssi not found")
        return 0.0
    def calculate_enl(img_denoised):
        print("ERROR: calculate_enl not found")
        return 0.0
# --- 修改结束 ---


status_len = 13


class BaseTrainer(object):
    def test(self):
        raise NotImplementedError('define this function for each trainer')

    def validation(self):
        raise NotImplementedError('define this function for each trainer')

    def _set_module(self):
        # return dict form with model name.
        raise NotImplementedError('define this function for each trainer')

    def _set_optimizer(self):
        # return dict form with each coresponding model name.
        raise NotImplementedError('define this function for each trainer')

    def _forward_fn(self, module, loss, data):
        # forward with model, loss function and data.
        # return output of loss function.
        raise NotImplementedError('define this function for each trainer')

    # ----------------------------#
    #    Train/Test functions      #
    # ----------------------------#
    def __init__(self, cfg):
        self.session_name = cfg['session_name']

        self.checkpoint_folder = 'checkpoint'

        # get file manager and logger class
        self.file_manager = FileManager(self.session_name)
        self.logger = Logger()

        self.cfg = cfg
        self.train_cfg = cfg['training']
        self.val_cfg = cfg['validation']
        self.test_cfg = cfg['test']
        self.ckpt_cfg = cfg['checkpoint']

        # --- MODIFIED: 添加最佳分数追踪器 ---
        self.best_brisque = float('inf') # BRISQUE 越低越好
        
        # --- 步骤 2: 在 __init__ 中添加用于追踪最佳 SSI/ENL 的变量 ---
        self.best_ssi = float('inf') # SSI 越低越好
        self.best_enl = 0.0          # ENL 越高越好
        # --- 修改结束 ---
        
        self.brisque_metric = None # 将在 _before_train 中初始化
        self.device = None # 将在 _before_train 中初始化
        # --- END MODIFIED ---


    def train(self):
        # initializing
        self._before_train()

        # warmup
        if self.epoch == 1 and self.train_cfg['warmup']:
            self._warmup()

        # training
        for self.epoch in range(self.epoch, self.max_epoch + 1):
            self._before_epoch()
            self._run_epoch()
            self._after_epoch()

        self._after_train()

    def _warmup(self):
        self._set_status('warmup')

        # make dataloader iterable.
        self.train_dataloader_iter = {}
        for key in self.train_dataloader:
            self.train_dataloader_iter[key] = iter(self.train_dataloader[key])

        warmup_iter = self.train_cfg['warmup_iter']
        if warmup_iter > self.max_iter:
            self.logger.info('currently warmup support 1 epoch as maximum. warmup iter is replaced to 1 epoch iteration. %d -> %d' % (warmup_iter, self.max_iter))
            warmup_iter = self.max_iter

        for self.iter in range(1, warmup_iter + 1):
            self._adjust_warmup_lr(warmup_iter)
            self._before_step()
            self._run_step()
            self._after_step()

    def _before_test(self, dataset_load):
        # initialing
        self.module = self._set_module()
        self._set_status('test')

        # --- MODIFIED: 初始化 BRISQUE (用于测试) ---
        self.device = torch.device("cuda" if self.cfg['gpu'] != 'None' else "cpu")
        # 假设测试时图像范围也是 [0, 255]
        self.brisque_metric = piq.BRISQUELoss(reduction='mean', data_range=255.0).to(self.device)
        # --- END MODIFIED ---

        # load checkpoint file
        ckpt_epoch = self._find_last_epoch() if self.cfg['ckpt_epoch'] == -1 else self.cfg['ckpt_epoch']
        ckpt_name = self.cfg['pretrained'] if self.cfg['pretrained'] is not None else None
        self.load_checkpoint(ckpt_epoch, name=ckpt_name)
        self.epoch = self.cfg['ckpt_epoch']  # for print or saving file name.

        # test dataset loader
        if dataset_load:
            self.test_dataloader = self._set_dataloader(self.test_cfg, batch_size=1, shuffle=False, num_workers=self.cfg['thread'])

        # wrapping and device setting
        if self.cfg['gpu'] != 'None':
            # model to GPU
            self.model = {key: nn.DataParallel(self.module[key]).cuda() for key in self.module}
        else:
            self.model = {key: nn.DataParallel(self.module[key]) for key in self.module}

        # evaluation mode and set status
        self._eval_mode()
        self._set_status('test %03d' % self.epoch)

        # start message
        self.logger.highlight(self.logger.get_start_msg())

        # set denoiser
        self._set_denoiser()

        # wrapping denoiser w/ self_ensemble
        if self.cfg['self_en']:
            # (warning) self_ensemble cannot be applied with multi-input model
            denoiser_fn = self.denoiser
            self.denoiser = lambda *input_data: self.self_ensemble(denoiser_fn, *input_data)

        # wrapping denoiser w/ crop test
        if 'crop' in self.cfg['test']:
            # (warning) self_ensemble cannot be applied with multi-input model
            denoiser_fn = self.denoiser
            self.denoiser = lambda *input_data: self.crop_test(denoiser_fn, *input_data, size=self.cfg['test']['crop'], overlap=20)

    def _before_train(self):
        # cudnn
        torch.backends.cudnn.benchmark = False
        self._set_status('train')

        # initialing
        self.module = self._set_module()

        # --- MODIFIED: 设置 device 和 BRISQUE (用于训练/验证) ---
        self.device = torch.device("cuda" if self.cfg['gpu'] != 'None' else "cpu")
        # 图像在 test_dataloader_process 中被 clamp 到 [0, 255]，所以 data_range=255.0
        self.brisque_metric = piq.BRISQUELoss(reduction='mean', data_range=255.0).to(self.device)
        self.best_brisque = float('inf') # 确保在训练开始时重置
        
        # --- 步骤 2 (续): 确保在训练开始时重置 SSI/ENL ---
        self.best_ssi = float('inf')
        self.best_enl = 0.0
        # --- 修改结束 ---
        
        # --- END MODIFIED ---

        # training dataset loader
        self.train_dataloader = self._set_dataloader(self.train_cfg, batch_size=self.train_cfg['batch_size'], shuffle=True, num_workers=self.cfg['thread'])

        # validation dataset loader
        if self.val_cfg['val']:
            self.val_dataloader = self._set_dataloader(self.val_cfg, batch_size=1, shuffle=False, num_workers=self.cfg['thread'])

        # other configuration
        self.max_epoch = self.train_cfg['max_epoch']
        self.epoch = self.start_epoch = 1
        # 确保 train_dataloader['dataset'] 存在并且是一个 DataLoader 对象
        if 'dataset' in self.train_dataloader and hasattr(self.train_dataloader['dataset'], 'dataset'):
             max_len = len(self.train_dataloader['dataset'].dataset)
             self.max_iter = math.ceil(max_len / self.train_cfg['batch_size'])
        else:
             # 提供一个备用逻辑或抛出错误
             print("Warning: Could not determine dataset length for max_iter calculation.")
             # 尝试获取第一个 dataloader (如果存在)
             first_key = next(iter(self.train_dataloader)) if self.train_dataloader else None
             if first_key and hasattr(self.train_dataloader[first_key], 'dataset'):
                  max_len = len(self.train_dataloader[first_key].dataset)
                  self.max_iter = math.ceil(max_len / self.train_cfg['batch_size'])
                  print(f"Using length from dataloader '{first_key}'.")
             else:
                  self.max_iter = 1000 # 或者设置一个默认值或抛出错误
                  print(f"Setting default max_iter to {self.max_iter}.")
                  # raise ValueError("Cannot determine dataset length to calculate max_iter.")


        self.loss = Loss(self.train_cfg['loss'], self.train_cfg['tmp_info'])
        self.loss_dict = {'count': 0}
        self.tmp_info = {}
        self.loss_log = []

        # set optimizer
        self.optimizer = self._set_optimizer()
        for opt in self.optimizer.values():
            opt.zero_grad(set_to_none=True)

        # resume
        if self.cfg["resume"]:
            # find last checkpoint
            load_epoch = self._find_last_epoch()

            # load last checkpoint
            self.load_checkpoint(load_epoch) # 这将恢复 self.best_brisque/ssi/enl
            self.epoch = load_epoch + 1

            # logger initialization
            self.logger = Logger((self.max_epoch, self.max_iter), log_dir=self.file_manager.get_dir(''), log_file_option='a')
        else:
            # logger initialization
            self.logger = Logger((self.max_epoch, self.max_iter), log_dir=self.file_manager.get_dir(''), log_file_option='w')

        # tensorboard
        tboard_time = datetime.datetime.now().strftime('%m-%d-%H-%M')
        self.tboard = SummaryWriter(log_dir=self.file_manager.get_dir('tboard/%s' % tboard_time))

        # wrapping and device setting
        if self.cfg['gpu'] != 'None':
            # model to GPU
            self.model = {key: nn.DataParallel(self.module[key]).cuda() for key in self.module}
            # optimizer to GPU
            for optim in self.optimizer.values():
                for state in optim.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
        else:
            self.model = {key: nn.DataParallel(self.module[key]) for key in self.module}

        # start message
        self.logger.info(self.summary())
        self.logger.start((self.epoch - 1, 0))
        self.logger.highlight(self.logger.get_start_msg())

    def _after_train(self):
        # finish message
        self.logger.highlight(self.logger.get_finish_msg())

    def _before_epoch(self):
        self._set_status('epoch %03d/%03d' % (self.epoch, self.max_epoch))

        # make dataloader iterable.
        self.train_dataloader_iter = {}
        for key in self.train_dataloader:
            self.train_dataloader_iter[key] = iter(self.train_dataloader[key])

        # model training mode
        self._train_mode()

    def _run_epoch(self):
        for self.iter in range(1, self.max_iter + 1):
            self._before_step()
            self._run_step()
            self._after_step()

    def _after_epoch(self):
        # save checkpoint (常规保存)
        if self.epoch >= self.ckpt_cfg['start_epoch']:
            if (self.epoch - self.ckpt_cfg['start_epoch']) % self.ckpt_cfg['interval_epoch'] == 0:
                self.save_checkpoint(is_best=False) # --- MODIFIED: 明确这是非最佳保存 ---

        # validation
        if self.val_cfg['val']:
            if self.epoch >= self.val_cfg['start_epoch'] and self.val_cfg['val']:
                if (self.epoch - self.val_cfg['start_epoch']) % self.val_cfg['interval_epoch'] == 0:
                    self._eval_mode()
                    self._set_status('val %03d' % self.epoch)

                    # --- 步骤 3: 捕获所有 3 个新指标 ---
                    current_brisque, current_ssi, current_enl = self.validation()
                    
                    # --- 记录 BRISQUE ---
                    if current_brisque is not None and current_brisque != float('inf'):
                        self.logger.info(f"Validation Epoch {self.epoch}: BRISQUE = {current_brisque:.6f} (Best: {self.best_brisque:.6f})")
                        if current_brisque < self.best_brisque:
                            self.best_brisque = current_brisque
                            self.logger.info(f"*** New Best BRISQUE: {self.best_brisque:.6f} at Epoch {self.epoch} ***")
                            self.save_checkpoint(is_best=True, metric_name='brisque') # 保存最佳模型
                    
                    # --- 记录 SSI ---
                    if current_ssi is not None:
                        self.logger.info(f"Validation Epoch {self.epoch}: SSI = {current_ssi:.4f} (Best: {self.best_ssi:.4f})")
                        if current_ssi < self.best_ssi:
                            self.best_ssi = current_ssi
                            self.logger.info(f"*** New Best SSI: {self.best_ssi:.4f} at Epoch {self.epoch} ***")
                            self.save_checkpoint(is_best=True, metric_name='ssi') # 保存最佳模型
                    
                    # --- 记录 ENL ---
                    if current_enl is not None:
                        self.logger.info(f"Validation Epoch {self.epoch}: ENL = {current_enl:.2f} (Best: {self.best_enl:.2f})")
                        if current_enl > self.best_enl:
                            self.best_enl = current_enl
                            self.logger.info(f"*** New Best ENL: {self.best_enl:.2f} at Epoch {self.epoch} ***")
                            self.save_checkpoint(is_best=True, metric_name='enl') # 保存最佳模型
                    # --- 修改结束 ---

            elif self.epoch == 2 and self.val_cfg['val']: # (你原来的逻辑，保留)
                self._eval_mode()
                self._set_status('val %03d' % self.epoch)
                # --- 步骤 3 (重复): 捕获所有 3 个新指标 ---
                current_brisque, current_ssi, current_enl = self.validation()
                
                if current_brisque is not None and current_brisque != float('inf'):
                    self.logger.info(f"Validation Epoch {self.epoch}: BRISQUE = {current_brisque:.6f} (Best: {self.best_brisque:.6f})")
                    if current_brisque < self.best_brisque:
                        self.best_brisque = current_brisque
                        self.logger.info(f"*** New Best BRISQUE: {self.best_brisque:.6f} at Epoch {self.epoch} ***")
                        self.save_checkpoint(is_best=True, metric_name='brisque')
                
                if current_ssi is not None:
                    self.logger.info(f"Validation Epoch {self.epoch}: SSI = {current_ssi:.4f} (Best: {self.best_ssi:.4f})")
                    if current_ssi < self.best_ssi:
                        self.best_ssi = current_ssi
                        self.logger.info(f"*** New Best SSI: {self.best_ssi:.4f} at Epoch {self.epoch} ***")
                        self.save_checkpoint(is_best=True, metric_name='ssi')
                
                if current_enl is not None:
                    self.logger.info(f"Validation Epoch {self.epoch}: ENL = {current_enl:.2f} (Best: {self.best_enl:.2f})")
                    if current_enl > self.best_enl:
                        self.best_enl = current_enl
                        self.logger.info(f"*** New Best ENL: {self.best_enl:.2f} at Epoch {self.epoch} ***")
                        self.save_checkpoint(is_best=True, metric_name='enl')
                # --- 修改结束 ---


    def _before_step(self):
        pass

    def _run_step(self):
        # (保持 _run_step 不变)
        data = {}
        try:
             for key in self.train_dataloader_iter:
                  data[key] = next(self.train_dataloader_iter[key])
        except StopIteration:
             print("Warning: Dataloader iterator exhausted prematurely. Re-initializing.")
             self.train_dataloader_iter = {}
             for key in self.train_dataloader:
                  self.train_dataloader_iter[key] = iter(self.train_dataloader[key])
             for key in self.train_dataloader_iter:
                  data[key] = next(self.train_dataloader_iter[key])

        if self.cfg['gpu'] != 'None':
            for dataset_key in data:
                for key in data[dataset_key]:
                    if isinstance(data[dataset_key][key], torch.Tensor):
                         data[dataset_key][key] = data[dataset_key][key].cuda()

        losses, tmp_info = self._forward_fn(self.model, self.loss, data)
        losses = {key: losses[key].mean() for key in losses}
        tmp_info = {key: tmp_info[key].mean() for key in tmp_info}

        total_loss = sum(v for v in losses.values())
        total_loss.backward()

        clip_value = 1.0
        for key in self.model:
            nn.utils.clip_grad_norm_(self.model[key].parameters(), max_norm=clip_value, error_if_nonfinite=True)

        for opt in self.optimizer.values():
            opt.step()

        global_iter = (self.epoch - 1) * self.max_iter + self.iter
        if global_iter % 500 == 0:
            for key in self.module:
                self.write_weight_hist(self.module[key], global_iter)

        for opt in self.optimizer.values():
            opt.zero_grad(set_to_none=True)

        for key in losses:
            if key != 'count':
                if key in self.loss_dict:
                    self.loss_dict[key] += float(losses[key])
                else:
                    self.loss_dict[key] = float(losses[key])
        for key in tmp_info:
            if key in self.tmp_info:
                self.tmp_info[key] += float(tmp_info[key])
            else:
                self.tmp_info[key] = float(tmp_info[key])
        self.loss_dict['count'] += 1


    def _after_step(self):
        # (保持 _after_step 不变)
        self._adjust_lr()

        if (self.iter % self.cfg['log']['interval_iter'] == 0 and self.iter != 0) or (self.iter == self.max_iter):
            self.print_loss()

        self.logger.print_prog_msg((self.epoch - 1, self.iter - 1))

        
    # --- 步骤 4: ！！！这是完整修正后的 test_dataloader_process 函数！！！ ---
    def test_dataloader_process(self, dataloader, add_con=0., floor=False, img_save=True, img_save_path=None, info=True):
        # make directory
        if img_save_path: # 确保路径存在
            self.file_manager.make_dir(img_save_path)

        brisque_scores = [] # 用于收集所有 *有效* 的分数
        psnr_scores = []
        ssim_scores = []
        
        # --- 在这里初始化 ssi 和 enl 列表 ---
        ssi_scores = []
        enl_scores = []
        # --- 修改结束 ---

        # --- 初始化 mean_psnr 和 mean_ssim ---
        mean_psnr = None
        mean_ssim = None
        # --- 结束初始化 ---

        time_begin = time.time()

        for idx, data in enumerate(dataloader):
            # to device
            if self.cfg['gpu'] != 'None':
                for key in data:
                    if key == 'file_name':
                        continue
                    if isinstance(data[key], torch.Tensor):
                         data[key] = data[key].cuda()


            # --- 将变量初始化放在循环的开头 ---
            denoised_image = None
            denoised_image_processed = None
            calculation_skipped = False # 统一的标志
            eva_name = data['file_name'][0] 

            try:
                # --- 把模型推理和后处理都放入 try 块 ---
                input_data = [data[arg] for arg in self.cfg['model_input']]

                if self.cfg['model_type'] == 'only_denoise':
                    denoised_image = self.denoiser(*input_data)
                else:
                    self.logger.info(f"[ERROR] Unsupported model_type '{self.cfg['model_type']}' for {eva_name}. Skipping.")
                    calculation_skipped = True

                # 只有在模型成功输出 Tensor 时才进行后处理和指标计算
                if isinstance(denoised_image, torch.Tensor):
                    denoised_image_processed = denoised_image + add_con 
                    if floor:
                        denoised_image_processed = torch.floor(denoised_image_processed)
                    denoised_image_processed = denoised_image_processed.clamp(0, 255)

                    # --- BRISQUE 计算 (No-Reference) ---
                    try:
                        score_tensor = self.brisque_metric(denoised_image_processed)
                        if not torch.isnan(score_tensor) and not torch.isinf(score_tensor):
                            brisque_scores.append(score_tensor.item())
                        else:
                            self.logger.info(f"[WARN] BRISQUE returned NaN/Inf for {eva_name}. Skipping score.")
                    except Exception as e:
                         self.logger.info(f"[WARN] BRISQUE calculation failed for {eva_name}: {e}.")
                         
                    # --- ENL 计算 (No-Reference) ---
                    try:
                        # (调用导入的函数)
                        enl_score = calculate_enl(denoised_image_processed)
                        if not np.isnan(enl_score) and not np.isinf(enl_score):
                            enl_scores.append(enl_score)
                        else:
                            self.logger.info(f"[WARN] ENL returned NaN/Inf for {eva_name}. Skipping score.")
                    except Exception as e:
                         self.logger.info(f"[WARN] ENL calculation failed for {eva_name}: {e}.")
                    # --- ENL 计算结束 ---

                else: 
                    if not calculation_skipped: 
                        self.logger.info(f"[WARN] Model output (denoised_image) is not a Tensor (maybe None?) for {eva_name}. Skipping ALL calculations.")
                    calculation_skipped = True

            except Exception as e:
                self.logger.info(f"[ERROR] Processing failed for {eva_name}: {e}. Skipping ALL calculations.")
                calculation_skipped = True
            # --- 结束 try 块 ---


            # --- "Referenced" 指标计算 (PSNR, SSIM, SSI) ---
            if not calculation_skipped and isinstance(denoised_image_processed, torch.Tensor):
                
                # --- SSI 计算 (需要 'noisy' 图像) ---
                noisy_key_found = None
                if 'noisy' in data: noisy_key_found = 'noisy'
                elif 'real_noisy' in data: noisy_key_found = 'real_noisy'

                if noisy_key_found:
                    try:
                        noisy_img = data[noisy_key_found]
                        # (调用导入的函数)
                        ssi_score = calculate_ssi(denoised_image_processed, noisy_img)
                        if not np.isnan(ssi_score) and not np.isinf(ssi_score):
                            ssi_scores.append(ssi_score)
                        else:
                            self.logger.info(f"[WARN] SSI returned NaN/Inf for {eva_name}. Skipping score.")
                    except Exception as e:
                        self.logger.info(f"[WARN] SSI calculation failed for {eva_name}: {e}.")
                else:
                    # (仅在验证/测试时打印一次信息)
                    if idx == 0: self.logger.info(f"[INFO] 'noisy' or 'real_noisy' key not in data. Skipping SSI.")
                # --- SSI 计算结束 ---

                # --- PSNR/SSIM 计算 (需要 'clean' 图像) ---
                if 'clean' in data:
                    try:
                        clean_img = data['clean']
                        psnr_scores.append(psnr(denoised_image_processed, clean_img))
                        ssim_scores.append(ssim(denoised_image_processed, clean_img))
                    except Exception as e:
                        self.logger.info(f"[WARN] PSNR/SSIM calculation failed for {eva_name}: {e}.")
                else:
                     if idx == 0: self.logger.info(f"[INFO] 'clean' key not in data. Skipping PSNR/SSIM.")
                # --- PSNR/SSIM 计算结束 ---


            # --- 图像保存逻辑 ---
            if img_save and img_save_path and isinstance(denoised_image_processed, torch.Tensor):
               noisy_key_found = 'noisy' if 'noisy' in data else 'real_noisy' if 'real_noisy' in data else None
               if noisy_key_found:
                   noisy_img_cpu = data[noisy_key_found].squeeze(0).cpu()
                   self.file_manager.save_img_tensor(img_save_path, '%s_N' % eva_name, noisy_img_cpu)

               denoi_img_cpu = denoised_image_processed.squeeze(0).cpu()
               self.file_manager.save_img_tensor(img_save_path, '%s_DN' % eva_name, denoi_img_cpu)
            elif img_save and img_save_path and not isinstance(denoised_image_processed, torch.Tensor):
               self.logger.info(f"[INFO] Skipping image save for {eva_name} because denoised_image was not valid.")


            if info:
                status_prefix = 'validating' if 'val' in self.status else 'testing'
                self.logger.note('[%s] %s... %04d/%04d.' % (self.status, status_prefix, idx + 1, len(dataloader)), end='\r')


        # --- 计算平均值 ---
        mean_brisque = np.mean(brisque_scores) if brisque_scores else None
        mean_psnr = np.mean(psnr_scores) if psnr_scores else None
        mean_ssim = np.mean(ssim_scores) if ssim_scores else None
        
        # --- 在这里计算 ssi 和 enl 的平均值 ---
        mean_ssi = np.mean(ssi_scores) if ssi_scores else None
        mean_enl = np.mean(enl_scores) if enl_scores else None
        # --- 修改结束 ---

        print("\n") # 确保在进度条后换行
        if mean_psnr is not None:
            print(f">>> PSNR (mean over {len(psnr_scores)} samples): {mean_psnr:.4f}")
        if mean_ssim is not None:
            print(f">>> SSIM (mean over {len(ssim_scores)} samples): {mean_ssim:.4f}")
        if mean_brisque is not None:
            print(f">>> BRISQUE (mean over {len(brisque_scores)} samples): {mean_brisque:.6f} (Lower is Better)")
            
        # --- 在这里打印 ssi 和 enl ---
        if mean_ssi is not None:
            print(f">>> SSI (mean over {len(ssi_scores)} samples): {mean_ssi:.4f} (Lower is Better)")
        if mean_enl is not None:
            print(f">>> ENL (mean over {len(enl_scores)} samples): {mean_enl:.2f} (Higher is Better)")
        # --- 修改结束 ---

        # --- 日志和返回 ---
        self.logger.val('[%s] Done!' % self.status)
        time_prefix = 'Validation' if 'val' in self.status else 'Test'
        print(f'{time_prefix} time: {time.time() - time_begin:.4f} seconds')

        # --- 步骤 5: 返回所有 5 个指标 ---
        return mean_psnr, mean_ssim, mean_brisque, mean_ssi, mean_enl
    # --- ！！！修正后的 test_dataloader_process 函数结束！！！ ---


    def test_img(self, image_dir, save_dir='./'):
        # (保持 test_img 不变)
        img = cv2.imread(image_dir, 1)
        if img is None:
             self.logger.info(f"[ERROR] Failed to load image: {image_dir}")
             return
        if len(img.shape) > 2 and img.shape[2] == 3:
             img = np.average(img, axis=2, weights=[0.114, 0.587, 0.299])
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(np.ascontiguousarray(img).astype(np.float32))
        noisy = img.unsqueeze(0)
        if self.cfg['gpu'] != 'None':
            noisy = noisy.cuda()
        try:
             denoised = self.denoiser(noisy)
        except Exception as e:
             self.logger.info(f"[ERROR] Model inference failed for {image_dir}: {e}")
             return
        add_con = 0. if not 'add_con' in self.test_cfg else self.test_cfg['add_con']
        floor = False if not 'floor' in self.test_cfg else self.test_cfg['floor']
        if not isinstance(denoised, torch.Tensor):
             self.logger.info(f"[WARN] Denoiser did not return a Tensor for {image_dir}. Skipping save.")
             return
        denoised += add_con
        if floor:
            denoised = torch.floor(denoised)
        denoised = denoised.clamp(0, 255)
        denoised_np = tensor2np(denoised.squeeze(0).cpu()) 
        if denoised_np is None or denoised_np.size == 0:
             self.logger.info(f"[WARN] tensor2np failed for {image_dir}. Skipping save.")
             return
        name = os.path.basename(image_dir).split('.')[0] 
        save_path = os.path.join(save_dir, name + '_DN.png')
        try:
             if denoised_np.dtype != np.uint8:
                 if denoised_np.max() <= 255 and denoised_np.min() >= 0:
                      denoised_np = denoised_np.astype(np.uint8)
                 else:
                      print(f"[WARN] Denoised image range for {name} is not [0, 255]. Attempting normalization before save.")
                      denoised_np = cv2.normalize(denoised_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
             cv2.imwrite(save_path, denoised_np)
        except Exception as e:
             self.logger.info(f"[ERROR] Failed to save image {save_path}: {e}")
             return
        self.logger.note('[%s] saved : %s' % (self.status, save_path))


    def test_dir(self, direc):
        # (保持 test_dir 不变)
        head, tail = os.path.split(direc)
        safe_tail = "".join(c if c.isalnum() else "_" for c in tail)
        result_dir = f'output/{safe_tail}' 
        print('Test dir:', direc)
        print('Result dir:', result_dir)
        count = 1
        time_begin_begin = time.time()
        if not os.path.isdir(direc):
             self.logger.info(f"[ERROR] Input directory does not exist: {direc}")
             return
        try:
             all_files = os.listdir(direc)
        except OSError as e:
             self.logger.info(f"[ERROR] Cannot access directory {direc}: {e}")
             return
        image_files = [f for f in all_files if os.path.isfile(os.path.join(direc, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        if not image_files:
             self.logger.info(f"[WARN] No image files found in {direc}")
             return
        try:
             os.makedirs(result_dir, exist_ok=True)
        except OSError as e:
             self.logger.info(f"[ERROR] Cannot create output directory {result_dir}: {e}")
             return
        total_files = len(image_files)
        for idx, ff in enumerate(image_files):
            file_path = os.path.join(direc, ff)
            time_begin = time.time()
            print(f'Processing image {idx + 1}/{total_files}: {ff} ...', end='', flush=True) 
            self.test_img(file_path, result_dir) 
            print(f' done in {time.time() - time_begin:.4f}s') 
            count += 1
        print(f'\nFinished processing {total_files} images.')
        print(f'Total time used: {time.time() - time_begin_begin:.4f} seconds')


    def _set_denoiser(self):
        if hasattr(self.model['denoiser'].module, 'denoise'):
            self.denoiser = self.model['denoiser'].module.denoise
        else:
            self.denoiser = self.model['denoiser'].module

    @torch.no_grad()
    def crop_test(self, fn, x, size=512, overlap=0):
        # (保持 crop_test 不变)
        b, c, h, w = x.shape
        if h <= size and w <= size:
             print("Info: Image smaller than crop size, skipping crop_test.")
             try:
                  return fn(x)
             except Exception as e:
                  self.logger.info(f"[ERROR] Inference failed even without cropping: {e}")
                  return torch.zeros_like(x) 
        delta = size - 2 * overlap
        if delta <= 0:
             print(f"Warning: Crop delta (size - 2*overlap = {delta}) is not positive. Adjusting overlap.")
             overlap = size // 4
             delta = size - 2 * overlap
             print(f"New overlap: {overlap}, New delta: {delta}")
        pad_h = 0
        if h < size:
             pad_h = size - h
        elif (h - size) % delta != 0:
             pad_h = delta - (h - size) % delta
        pad_w = 0
        if w < size:
             pad_w = size - w
        elif (w - size) % delta != 0:
             pad_w = delta - (w - size) % delta
        x_padded = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='reflect')
        _, _, new_h, new_w = x_padded.shape
        denoised_image = torch.zeros_like(x_padded)
        count_map = torch.zeros_like(x_padded) 
        try:
             for i in range(0, new_h - overlap, delta): 
                  for j in range(0, new_w - overlap, delta): 
                       end_i = min(i + size, new_h)
                       end_j = min(j + size, new_w)
                       start_i_eff = max(0, i) 
                       start_j_eff = max(0, j)
                       x_crop = x_padded[..., start_i_eff:end_i, start_j_eff:end_j]
                       if x_crop.numel() == 0: 
                           print(f"Warning: Empty crop at [{start_i_eff}:{end_i}, {start_j_eff}:{end_j}]. Skipping.")
                           continue
                       denoised_crop = fn(x_crop)
                       denoised_image[..., start_i_eff:end_i, start_j_eff:end_j] += denoised_crop
                       count_map[..., start_i_eff:end_i, start_j_eff:end_j] += 1
        except Exception as e:
             self.logger.info(f"[ERROR] Error during crop_test inference: {e}")
             return x_padded[:, :, :h, :w] 
        count_map[count_map == 0] = 1
        denoised_image /= count_map
        return denoised_image[:, :, pad_h // 2 : pad_h // 2 + h, pad_w // 2 : pad_w // 2 + w]


    @torch.no_grad()
    def self_ensemble(self, fn, x):
        # (保持 self_ensemble 不变)
        result = torch.zeros_like(x)
        try:
             for i in range(8):
                  x_aug = rot_hflip_img(x, rot_times=i % 4, hflip=i // 4 > 0)
                  tmp = fn(x_aug)
                  tmp_inv_rot = rot_hflip_img(tmp, rot_times=4 - (i % 4))
                  tmp_inv_aug = rot_hflip_img(tmp_inv_rot, hflip=i // 4 > 0) 
                  result += tmp_inv_aug
        except Exception as e:
             self.logger.info(f"[ERROR] Error during self_ensemble: {e}")
             return x 
        return result / 8

    # tensorboard
    def write_weight_hist(self, net, index):
        # (保持 write_weight_hist 不变)
        try:
             max_grad = 0
             min_grad = 0
             for name, param in net.named_parameters():
                  if 'test' in name or param.grad is None: 
                       continue
                  if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                       print(f"[WARN] NaN or Inf found in gradients of {name} at iter {index}. Skipping histogram.")
                       continue
                  grad_max_val = torch.max(param.grad).item()
                  grad_min_val = torch.min(param.grad).item()
                  if grad_max_val > max_grad:
                       max_grad = grad_max_val
                  if grad_min_val < min_grad:
                       min_grad = grad_min_val
                  root, sub_name = os.path.splitext(name)
                  tb_name_param = f"{root}/{sub_name}_param".replace('.', '/')
                  tb_name_grad = f"{root}/{sub_name}_grad".replace('.', '/')
                  self.tboard.add_histogram(tb_name_param, param.data, index)
                  self.tboard.add_histogram(tb_name_grad, param.grad.data, index)
        except Exception as e:
             print(f'[ERROR] Failed to write weight histogram at iter {index}: {e}')

    # ----------------------------#
    #      Utility functions       #
    # ----------------------------#
    def print_loss(self):
        # (保持 print_loss 不变)
        if self.loss_dict['count'] == 0: return 
        temporal_loss = 0.
        loss_items = [] 
        for key in self.loss_dict:
            if key != 'count':
                loss_val = self.loss_dict[key] / self.loss_dict['count']
                temporal_loss += loss_val
                loss_items.append(f"{key} : {loss_val:.5f}")
                global_iter = (self.epoch - 1) * self.max_iter + self.iter
                self.tboard.add_scalar(f'loss/{key}', loss_val, global_iter)
                self.loss_dict[key] = 0.
        self.loss_log.append(temporal_loss) 
        if len(self.loss_log) > 100: self.loss_log.pop(0)
        loss_out_str = '[%s] %04d/%04d, lr:%s \t ' % (self.status, self.iter, self.max_iter, "{:.1e}".format(self._get_current_lr()))
        global_iter = (self.epoch - 1) * self.max_iter + self.iter
        avg_loss = np.mean(self.loss_log) 
        loss_out_str += 'avg_100_total : %.5f \t ' % (avg_loss) 
        self.tboard.add_scalar('loss/avg_100_total', avg_loss, global_iter) 
        loss_out_str += " \t ".join(loss_items)
        if len(self.tmp_info) > 0:
            loss_out_str += '\t['
            tmp_items = []
            for key in self.tmp_info:
                tmp_val = self.tmp_info[key] / self.loss_dict['count']
                tmp_items.append(f"{key} : {tmp_val:.2f}")
                self.tmp_info[key] = 0.
            loss_out_str += "  ".join(tmp_items)
            loss_out_str += ' ]'
        self.loss_dict['count'] = 0
        self.logger.info(loss_out_str)


    # --- 步骤 5: 修改 'save_checkpoint' ---
    def save_checkpoint(self, is_best=False, metric_name=''):
        try:
             # 将所有最佳指标添加到 state
             state = {
                 'epoch': self.epoch,
                 'model_weight': {key: self.model[key].module.state_dict() for key in self.model},
                 'optimizer_weight': {key: self.optimizer[key].state_dict() for key in self.optimizer},
                 'best_brisque': self.best_brisque, # 保存最佳 BRISQUE
                 'best_ssi': self.best_ssi,       # 保存最佳 SSI
                 'best_enl': self.best_enl        # 保存最佳 ENL
             }

             ckpt_dir = self.file_manager.get_dir(self.checkpoint_folder)
             os.makedirs(ckpt_dir, exist_ok=True)


             # 保存常规的 epoch checkpoint
             if not is_best: # 只在非最佳保存时才保存 epoch-xxx.pth
                checkpoint_name = self._checkpoint_name(self.epoch)
                save_path = os.path.join(ckpt_dir, checkpoint_name)
                torch.save(state, save_path)
                self.logger.info(f"Saved checkpoint: {save_path}")

             # 如果是最佳模型，额外保存一个 'best' 文件
             if is_best and metric_name:
                 best_filename = os.path.join(ckpt_dir, f'model_best_{metric_name}.pth')
                 torch.save(state, best_filename)
                 self.logger.info(f"Saved best model checkpoint to {best_filename}")
                 
                 # (可选) 同时覆盖一个通用的 'model_best.pth'
                 # best_filename_generic = os.path.join(ckpt_dir, 'model_best.pth')
                 # torch.save(state, best_filename_generic)
                 
        except Exception as e:
             self.logger.info(f"[ERROR] Failed to save checkpoint for epoch {self.epoch}: {e}")
    # --- 修改结束 ---

    def load_checkpoint(self, load_epoch=0, name=None):
        # (保持 load_checkpoint 不变, 但我为你清理和修复了它)
        if name is None:
            if load_epoch == 0:
                self.logger.info("Starting training from scratch (epoch 1).")
                return
            file_name = os.path.join(self.file_manager.get_dir(self.checkpoint_folder), self._checkpoint_name(load_epoch))
        else:
            file_name = os.path.join('./ckpt', name)

        if not os.path.isfile(file_name):
             self.logger.info(f"[WARN] Checkpoint file not found: {file_name}. Starting from scratch.")
             self.epoch = 1 
             return 

        try:
             saved_checkpoint = torch.load(file_name, map_location=self.device)

             for key in self.module:
                 state_dict = saved_checkpoint['model_weight'][key]
                 if list(state_dict.keys())[0].startswith('module.'):
                     state_dict = {k[7:]: v for k, v in state_dict.items()}
                 self.module[key].load_state_dict(state_dict)

             if hasattr(self, 'optimizer'):
                 for key in self.optimizer:
                     if key in saved_checkpoint['optimizer_weight']:
                         self.optimizer[key].load_state_dict(saved_checkpoint['optimizer_weight'][key])
                         if self.device != torch.device('cpu'):
                             for state in self.optimizer[key].state.values():
                                 for k, v in state.items():
                                     if isinstance(v, torch.Tensor):
                                         state[k] = v.to(self.device)
                     else:
                          print(f"Warning: Optimizer state for '{key}' not found in checkpoint.")

             # --- 步骤 6: 恢复所有最佳分数 ---
             self.best_brisque = saved_checkpoint.get('best_brisque', float('inf'))
             self.best_ssi = saved_checkpoint.get('best_ssi', float('inf'))
             self.best_enl = saved_checkpoint.get('best_enl', 0.0)
             self.logger.info(f"Loaded best_brisque score: {self.best_brisque:.6f}")
             self.logger.info(f"Loaded best_ssi score: {self.best_ssi:.4f}")
             self.logger.info(f"Loaded best_enl score: {self.best_enl:.2f}")
             # --- 修改结束 ---

             self.logger.note('[%s] model loaded : %s (Epoch %d)' % (self.status, file_name, saved_checkpoint['epoch']))

        except Exception as e:
             self.logger.info(f"[ERROR] Failed to load checkpoint from {file_name}: {e}. Starting from scratch.")
             self.epoch = 1 
             self.best_brisque = float('inf')
             self.best_ssi = float('inf')
             self.best_enl = 0.0

    def _checkpoint_name(self, epoch):
        return self.session_name + '_%03d' % epoch + '.pth'

    def _find_last_epoch(self):
        # (保持 _find_last_epoch 不变)
         ckpt_dir = self.file_manager.get_dir(self.checkpoint_folder)
         if not os.path.isdir(ckpt_dir):
             print(f"Checkpoint directory not found: {ckpt_dir}. Starting from epoch 1.")
             return 0 
         checkpoint_list = os.listdir(ckpt_dir)
         epochs = []
         for ckpt in checkpoint_list:
             if ckpt.startswith(self.session_name + '_') and ckpt.endswith('.pth'):
                 try:
                      epoch_str = ckpt[len(self.session_name) + 1:-4]
                      if epoch_str.isdigit():
                           epochs.append(int(epoch_str))
                 except ValueError:
                      continue 
         if not epochs:
             print(f"No resumable checkpoint found for session '{self.session_name}' in {ckpt_dir}. Starting from epoch 1.")
             return 0 
         last_epoch = max(epochs)
         print(f"Found last checkpoint at epoch {last_epoch}.")
         return last_epoch


    def _get_current_lr(self):
        # (保持 _get_current_lr 不变)
        if not hasattr(self, 'optimizer') or not self.optimizer:
             return 0.0 
        for first_optim in self.optimizer.values():
            if first_optim.param_groups: 
                return first_optim.param_groups[0]['lr']
        return 0.0 

    def _set_dataloader(self, dataset_cfg, batch_size, shuffle, num_workers):
        # (保持 _set_dataloader 不变)
        dataloader = {}
        dataset_dict = dataset_cfg['dataset']
        if not isinstance(dataset_dict, dict):
            dataset_dict = {'dataset': dataset_dict}
        for key in dataset_dict:
            args_key = key + '_args'
            if args_key not in dataset_cfg:
                 print(f"Warning: Arguments '{args_key}' not found in config for dataset '{key}'. Using empty args.")
                 args = {} 
            else:
                 args = dataset_cfg[args_key]
                 if args is None: args = {} 
            try:
                 dataset = get_dataset_class(dataset_dict[key])(**args)
                 dataloader[key] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False, drop_last=(shuffle)) 
            except KeyError:
                 print(f"[ERROR] Dataset class '{dataset_dict[key]}' not found or not registered in src/datahandler/__init__.py")
                 raise 
            except Exception as e:
                 print(f"[ERROR] Failed to initialize dataset '{dataset_dict[key]}' with args {args}: {e}")
                 raise
        return dataloader

    def _set_one_optimizer(self, opt, parameters, lr):
        # (保持 _set_one_optimizer 不变)
        lr = float(self.train_cfg['init_lr'])
        if opt is None or 'type' not in opt:
            raise ValueError("Optimizer configuration is missing or invalid.")
        if opt['type'] == 'SGD':
            sgd_args = opt.get('SGD', {})
            momentum = float(sgd_args.get('momentum', 0.9)) 
            weight_decay = float(sgd_args.get('weight_decay', 0))
            return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif opt['type'] == 'Adam':
            adam_args = opt.get('Adam', {})
            betas = adam_args.get('betas', (0.9, 0.999)) 
            return optim.Adam(parameters, lr=lr, betas=betas)
        elif opt['type'] == 'AdamW':
            adamw_args = opt.get('AdamW', {})
            betas = adamw_args.get('betas', (0.9, 0.999))
            weight_decay = float(adamw_args.get('weight_decay', 0.01))
            return optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay) 
        else:
            raise RuntimeError('ambiguious optimizer type: {}'.format(opt['type']))

    def _adjust_lr(self):
        # (保持 _adjust_lr 不变)
        if 'scheduler' not in self.train_cfg or self.train_cfg['scheduler'] is None:
             return 
        sched = self.train_cfg['scheduler']
        if 'type' not in sched: return 
        if sched['type'] == 'step':
            if self.iter == self.max_iter: 
                args = sched.get('step', {})
                step_size = args.get('step_size')
                gamma = float(args.get('gamma', 0.1)) 
                if step_size is None:
                     print("Warning: 'step_size' not defined for step scheduler.")
                     return
                if self.epoch % step_size == 0:
                     for optimizer in self.optimizer.values():
                          lr_before = optimizer.param_groups[0]['lr']
                          new_lr = lr_before * gamma
                          print(f"Adjusting learning rate (step): {lr_before:.1e} -> {new_lr:.1e}")
                          for param_group in optimizer.param_groups:
                               param_group["lr"] = new_lr
        elif sched['type'] == 'linear':
             args = sched['linear']
             if not hasattr(self, 'reset_lr'):
                  gamma = float(args.get('gamma', 1.0)) 
                  step_size = args.get('step_size')
                  if step_size is None:
                       print("Warning: 'step_size' not defined for linear scheduler.")
                       return
                  self.reset_lr = float(self.train_cfg['init_lr']) * gamma ** ((self.epoch - 1) // step_size)
             step_size = args.get('step_size')
             gamma = float(args.get('gamma', 1.0))
             if step_size is None: return
             if self.epoch % step_size == 0 and self.iter == self.max_iter:
                  self.reset_lr = float(self.train_cfg['init_lr']) * gamma ** (self.epoch // step_size)
                  for optimizer in self.optimizer.values():
                       for param_group in optimizer.param_groups:
                            param_group["lr"] = self.reset_lr
             else:
                  ratio = ((self.epoch + (self.iter) / self.max_iter - 1) % step_size) / step_size
                  curr_lr = (1 - ratio) * self.reset_lr
                  for optimizer in self.optimizer.values():
                       for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
        elif sched['type'] == 'Cosine':
             if not hasattr(self, 'cos_scheduler'):
                  self.cos_scheduler = []
                  print('setting CosineAnnealingLR')
                  min_lr = float(sched.get('min', 0)) 
                  for optimizer in self.optimizer.values():
                       t_max = self.max_epoch if self.max_epoch > 0 else 100 
                       self.cos_scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr, last_epoch=-1)) 
             if self.iter == self.max_iter: 
                  for scheduler in self.cos_scheduler:
                       scheduler.step()
        else:
             raise RuntimeError('ambiguious scheduler type: {}'.format(sched['type']))


    def _adjust_warmup_lr(self, warmup_iter):
        # (保持 _adjust_warmup_lr 不变)
        if warmup_iter <= 0: return
        init_lr = float(self.train_cfg['init_lr'])
        warmup_lr = init_lr * self.iter / warmup_iter
        for optimizer in self.optimizer.values():
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

    def _train_mode(self):
        for key in self.model:
            self.model[key].train()

    def _eval_mode(self):
        for key in self.model:
            self.model[key].eval()

    def _set_status(self, status: str):
        # (保持 _set_status 不变)
         max_len = status_len
         if len(status) > max_len:
              status = status[:max_len-3] + "..." 
         if len(status.split(' ')) == 2:
              s0, s1 = status.split(' ')
              pad0 = (max_len - 1) // 2 - len(s0)
              pad1 = max_len - 1 - len(s0) - pad0 - len(s1)
              self.status = ' ' * pad0 + s0 + ' ' + s1 + ' ' * pad1
         else:
              sp = max_len - len(status)
              self.status = ''.ljust(sp // 2) + status + ''.ljust((sp + 1) // 2)


    def summary(self):
        # (保持 summary 不变)
        summary = ''
        summary += '-' * 100 + '\n'
        total_params = 0
        for k, v in self.module.items():
             param_num = sum(p.numel() for p in v.parameters() if p.requires_grad) 
             total_params += param_num
             summary += '[%s] Trainable paramters: %s' % (k, human_format(param_num)) + '\n' 
        summary += f"Total Trainable Parameters: {human_format(total_params)}\n"
        if hasattr(self, 'optimizer') and self.optimizer:
              summary += "Optimizer(s):\n"
              for k, v in self.optimizer.items():
                   summary += f"  [{k}]: {v.__class__.__name__} (lr={self._get_current_lr():.1e})\n" 
        if self.device:
              summary += f"Device: {self.device}\n"
        if torch.cuda.is_available():
              summary += f"CUDA Devices: {torch.cuda.device_count()}\n"
        summary += '-' * 100 + '\n'
        return summary