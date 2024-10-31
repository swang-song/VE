# Environment Setup Guide
 
## On Windows
### 1. Install WSL 2

1. Install Ubuntu 22.04.

```
wsl --install -d Ubuntu-22.04
wsl -l -v
```

2. (Optional) Move the WSL 2 environment to another drive.

```
wsl --export Ubuntu-22.04 <path-to-portable-drive>\Ubuntu-22.04.tar
wsl --unregister Ubuntu-22.04
wsl --import Ubuntu-22.04 <path-to-portable-drive>\Ubuntu <path-to-portable-drive>\Ubuntu-22.04.tar --version 2
wsl
```

3. (Optional) Set the default user.

```
ubuntu2004.exe config --default-user <user_name>
```

## On Ubuntu 22.04
### 1. Install Conda

```
sudo apt-get update && sudo apt-get upgrade -y
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
export PATH="/home/<user_name>/miniconda3/bin:$PATH"
conda init
source ~/.bashrc
conda --version
```

### 2. Create virtual environment with Python 3.8.

```
conda create -n myenv python=3.8
echo 'conda activate myenv' >> ~/.bashrc
source ~/.bashrc
python --version
```

### 3. Install [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local).

```
cd ~/
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --silent --driver
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc -V
```

### 4. Install PyTorch 2.3.1

```
# Using pip to install torch==2.3.1+cu121
pip install torch==2.3.1+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Alternatively, using Conda
conda install pytorch==2.3.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Verify installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 5. Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

