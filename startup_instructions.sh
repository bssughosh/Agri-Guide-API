#!/bin/bash
sudo apt update
sudo apt install bzip2 libxml2-dev libsm6 libxrender1 libfontconfig1 wget
cd
wget https://repo.anaconda.com/miniconda/Miniconda3-4.7.10-Linux-x86_64.sh
bash Miniconda3-4.7.10-Linux-x86_64.sh
read -p "Press enter to continue"
source ~/.bashrc
rm Miniconda3-4.7.10-Linux-x86_64.sh
conda create -n fyp python=3.7
conda activate fyp
cd Agri-Guide-API/
pip install -r requirements.txt

