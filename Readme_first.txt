# Project Name: LVM_HVS Submission
# This project evaluates the alignment of large vision models (LVMs) with the human visual system (HVS) using standardized protocols. Follow the instructions below to set up the environment and run tests.

## Environment Setup
conda create -n lvm_hvs python=3.12

## Check your CUDA version:
conda activate lvm_hvs
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tqdm opencv-python matplotlib scikit-learn scipy transformers diffusers accelerate

## Install SAM (Run all these commands in the LVM_HVS_submission directory):
pip install git+https://github.com/facebookresearch/segment-anything.git
## Please download all the model checkpoints from the SAM official website, and put in the SAM_repo

## The SAM-2 repo is always updating, so we don't provide the code for running the SAM-2 test

## Install ColorVideoVDP (Run all these commands in the LVM_HVS_submission directory):
pip install pynvml
conda install ffmpeg conda-forge::freeimage
git clone git@github.com:gfxdisp/ColorVideoVDP.git % Please skip the tests of ColorVideoVDP if you cannot download it.
cd ColorVideoVDP
pip install -e .

## Install OpenCLIP
pip install open_clip_torch

# What commands are useful?

## It is recommended to run the code using a compiler, such as PyCharm.
## To generate test stimulus (Optional):
Band_limit_noise_generator/generate_plot_band_lim_noise.py
Contrast_masking_generator/generate_plot_contrast_masking.py
Contrast_masking_generator/generate_plot_contrast_masking_gabor_on_noise.py
Gabor_test_stimulus_generator/generate_plot_gabor_functions_new.py
Sinusoidal_grating_generator/generate_plot_sinusoidal_grating.py

## You need to revise the test_main.py if you cannot test the ColorVideoVDP
## To test:
test_main.py

## To plot:
plot_main.py

## To compute model alignment scores:
compute_error_main.py

