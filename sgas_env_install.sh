#!/usr/bin/env bash
# make sure command is : source sgas_env_install.sh

# install anaconda3.
# cd ~/
# wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
# bash Anaconda3-2019.07-Linux-x86_64.sh

# module load, uncommet if using local machine
#module purge
#module load gcc
#module load cuda/10.1.105

# make sure your annaconda3 is added to bashrc
#source activate
#source ~/.bashrc

conda create --name sgas
conda activate sgas
conda install -y pytorch=1.4.0 torchvision cudatoolkit=10.0 python=3.6.8 Pillow==6.1 -c pytorch # make sure python=3.6.8

# install useful modules
pip install tqdm tensorboardX graphviz

# install pyg
#  https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
CUDA=cu100
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip install torch-sparse==0.6.1 -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip install torch-cluster==1.4.5 -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip install torch-geometric==1.4.3
pip install requests

