HSSD-mamba

Official implementation of "Leakage-Free Self-Supervised Sonar Image Denoising via Omnidirectional Blind Scanning and Multiscale Semantic Refinement".

🛠 Setup
Requirements

Our experiments were conducted with the following environment:

Python 3.9.15

PyTorch 1.12.1

CUDA Toolkit 11.3.1

Core Libs: numpy 1.23.5, opencv-python 4.7.0.72, scikit-image 0.19.3

Installation
Bash
# Clone the repository
git clone https://github.com/wangxu777777/HSSD-mamba
cd HSSD-mamba

# (Optional) Install dependencies
# pip install -r requirements.txt

🚀 Training & Test
Training

You can control detailed experimental configurations (e.g., loss function, epochs, batch size) via the configuration files in the configs/ directory.

Bash
# Train HSSD-mamba on the KLSG dataset (using GPU 0)
python train.py --session_name train_HSSD-mamba_KLSG \
                --config KLSG/config \
                --gpu 0

# Train HSSD-mamba on the DEBRIS dataset
python train.py --session_name train_HSSD-mamba_DEBRIS \
                --config DEBRIS/config \
                --gpu 0

Note: Ensure your dataset is placed in ./dataset/prep/. You can specify --input_dir and --label_dir if they are not defined in the config file.

Test
To evaluate the model, point to the pre-trained checkpoint:

Bash
# Test KLSG dataset with trained HSSD-mamba
python test.py --session_name Test_HSSD-mamba_KLSG \
               --config KLSG/config \
               --pretrained ./ckpt/SEGSID_KLSG.pth \
               --gpu 0

