# [CVPR2025] Do computer vision foundation models learn the low-level characteristics of the human visual system?

<img src="images/cvpr-navbar-logo-2.png" width="300"/>

<a href="https://caiyancheng.github.io/academic.html">Yancheng Cai</a>,
<a href="https://feiiyin.github.io/">Fei Yin</a>,
<a href="https://www.cst.cam.ac.uk/people/dh706">Dounia Hammou</a>,
<a href="https://www.cl.cam.ac.uk/~rkm38/">Rafał K. Mantiuk</a>
; University of Cambridge, UK

[[Paper](https://arxiv.org/abs/2502.20256)] | [[Project](https://www.cl.cam.ac.uk/research/rainbow/projects/vfm_hvs/)] | [[Results](https://caiyancheng.github.io/synthetic_test_webpage_lvm/index.html)]
## Abstract
Computer vision foundation models, such as DINO or OpenCLIP, are trained in a self-supervised manner on large image datasets. Analogously, substantial evidence suggests that the human visual system (HVS) is influenced by the statistical distribution of colors and patterns in the natural world, characteristics also present in the training data of foundation models. The question we address in this paper is whether foundation models trained on natural images mimic some of the low-level characteristics of the human visual system, such as contrast detection, contrast masking, and contrast constancy. Specifically, we designed a protocol comprising nine test types to evaluate the image encoders of 45 foundation and generative models. Our results indicate that some foundation models (e.g., DINO, DINOv2, and OpenCLIP), share some of the characteristics of human vision, but other models show little resemblance. Foundation models tend to show smaller sensitivity to low contrast and rather irregular responses to contrast across frequencies. The foundation models show the best agreement with human data in terms of contrast masking. Our findings suggest that human vision and computer vision may take both similar and different paths when learning to interpret images of the real world. Overall, while differences remain, foundation models trained on vision tasks start to align with low-level human vision, with DINOv2 showing the closest resemblance.

&ensp;
<p align="center">
  <img src="images/first_figure_8.png" width="60%">
</p>

## Setup
### Conda Environment Setup
```bash
conda create -n lvm_hvs python=3.12
conda activate lvm_hvs
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tqdm opencv-python matplotlib scikit-learn scipy transformers diffusers accelerate
```
### Models Setup
```bash
git clone https://github.com/caiyancheng/VFM_HVS_CVPR2025.git
cd VFM_HVS_CVPR2025
```
Install SAM
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
## Please download all the model checkpoints from the SAM official website, and put in the SAM_repo
## The SAM-2 repo is always updating, so we don't provide the code for running the SAM-2 test
```
Install ColorVideoVDP
```bash
pip install pynvml
conda install ffmpeg conda-forge::freeimage
git clone git@github.com:gfxdisp/ColorVideoVDP.git % Please skip the tests of ColorVideoVDP if you cannot download it.
cd ColorVideoVDP
pip install -e .
```
Install OpenCLIP
```bash
pip install open_clip_torch
```
## Usage
It is recommended to run the code using a compiler, such as PyCharm.
### Generate test stimulus (Optional)
```bash
Band_limit_noise_generator/generate_plot_band_lim_noise.py
Contrast_masking_generator/generate_plot_contrast_masking.py
Contrast_masking_generator/generate_plot_contrast_masking_gabor_on_noise.py
Gabor_test_stimulus_generator/generate_plot_gabor_functions_new.py
Sinusoidal_grating_generator/generate_plot_sinusoidal_grating.py
```
### Test
```bash
test_main.py
```
### Plot figures
```bash
plot_main.py
```
### Compute model alignment scores
```bash
compute_error_main.py
```

## Citation
If you find this work helpful in your research, please cite.
````
@article{cai2025computer,
  title={Do computer vision foundation models learn the low-level characteristics of the human visual system?},
  author={Cai, Yancheng and Yin, Fei and Hammou, Dounia and Mantiuk, Rafal},
  journal={arXiv preprint arXiv:2502.20256},
  year={2025}
}
````
