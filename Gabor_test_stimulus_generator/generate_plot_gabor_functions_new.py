import numpy as np
import matplotlib.pyplot as plt
from display_encoding import display_encode
from Color_space_Transform import *
import cv2
from PIL import Image

Luminance_min = 1e-4
display_encode_tool = display_encode(400)
# dkl_ratios = np.array([1, 0.610649, 4.203636])
dkl_ratios = np.array([1, 1, 1])
white_point_d65 = np.array([0.9505, 1.0000, 1.0888])

def generate_gabor_patch(W, H, R, rho, O, L_b, contrast, ppd, color_direction):
    if color_direction == 'ach':
        col_dir = np.array([dkl_ratios[0], 0, 0])
    elif color_direction == 'rg':
        col_dir = np.array([0, dkl_ratios[1], 0])
    elif color_direction == 'yv':
        col_dir = np.array([0, 0, dkl_ratios[2]])
    else:
        raise ValueError('Color Direction Value is not correct. We only support ach, rg, yv')
    x = np.linspace(-W // 2, W // 2, W)
    y = np.linspace(-H // 2, H // 2, H)
    X, Y = np.meshgrid(x, y)
    theta = np.deg2rad(O)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
    gaussian = np.exp(-0.5 * (X_rot ** 2 + Y_rot ** 2) / (ppd * R) ** 2)
    sinusoid = np.sin(2 * np.pi * rho * X_rot / ppd) * contrast * L_b
    T_vid_single_channel = gaussian * sinusoid + L_b
    C_dkl = lms2dkl_d65(xyz2lms2006(white_point_d65 * L_b))
    I_dkl_ref = np.ones((H, W, 3)) * C_dkl.reshape(1, 1, 3)
    I_dkl_test = (I_dkl_ref + (T_vid_single_channel[:, :, np.newaxis] - L_b) * col_dir.reshape(1, 1, 3))
    T_vid_rgb = cm_xyz2rgb(lms2006_2xyz(dkl2lms_d65(I_dkl_test)))
    R_vid_rgb = cm_xyz2rgb(lms2006_2xyz(dkl2lms_d65(I_dkl_ref)))
    assert np.all(T_vid_rgb >= 0), "We cannot have any out of gamut colours"
    T_vid_rgb[T_vid_rgb < Luminance_min] = Luminance_min
    return T_vid_rgb, R_vid_rgb

def plot_gabor(T_vid, R_vid):
    T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
    R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
    plt.figure(figsize=(4, 4), dpi=300)
    plt.imshow(T_vid_c, extent=(-W // 2, W // 2, -H // 2, H // 2))
    # plt.title(f'Radius = {R} degree, \n S_freq = {rho} cpd, Contrast = {contrast}, \n ppd = {ppd}, W = {W}, H = {H}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('CM.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.show()

def save_gabor(T_vid, R_vid):
    T_vid_c = np.array(display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8)
    T_vid_c = cv2.cvtColor(T_vid_c, cv2.COLOR_RGB2BGR)
    cv2.imwrite("T_vid_c.jpg", T_vid_c)

if __name__ == '__main__':
    scale_k1 = 1
    scale_k2 = 1

    W = 224 * scale_k2  # Width of the canvas (pixels)
    H = 224 * scale_k2  # Height of the canvas (pixels)
    R = 1 * scale_k1 * scale_k2  # Radius of the Gabor stimulus (degrees)
    rho = 4 / scale_k1 / scale_k2  # Spatial frequency of the Gabor stimulus (cycles per degree)
    O = 0  # Orientation of the Gabor stimulus (degrees)
    L_b = 100 # Luminance of the background
    contrast = 0.2 # Contrast of the gabor
    ppd = 60 / scale_k1
    color_direction = 'yv'

    T_vid, R_vid = generate_gabor_patch(W, H, R, rho, O, L_b, contrast, ppd, color_direction)
    # save_gabor(T_vid, R_vid)
    plot_gabor(T_vid, R_vid)

