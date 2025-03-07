import numpy as np
import matplotlib.pyplot as plt
from display_encoding import display_encode

Luminance_min = 1e-4
display_encode_tool = display_encode(400)


def generate_sinusoidal_grating(W, H, spatial_frequency, orientation, L_b, contrast, ppd):
    x = np.linspace(-W // 2, W // 2, W)
    y = np.linspace(-H // 2, H // 2, H)
    X, Y = np.meshgrid(x, y)
    theta = np.deg2rad(orientation)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    sinusoid = np.sin(2 * np.pi * spatial_frequency * X_rot / ppd) * contrast * L_b
    T_vid = sinusoid + L_b
    return T_vid


def plot_sinusoidal_grating(T_vid, W, H, spatial_frequency, contrast, ppd):
    T_vid_c = display_encode_tool.L2C_sRGB(T_vid) * 255
    plt.figure(figsize=(4, 4))
    plt.imshow(T_vid_c, cmap='gray', vmin=0, vmax=255, extent=(-W // 2, W // 2, -H // 2, H // 2))
    plt.title(f'Spatial Frequency = {spatial_frequency} cpd, Contrast = {contrast}, \n ppd = {ppd}, W = {W}, H = {H}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    scale_k1 = 1
    scale_k2 = 1

    W = 224 * scale_k2  # Width of the canvas (pixels)
    H = 224 * scale_k2  # Height of the canvas (pixels)
    spatial_frequency = 2 / scale_k1 / scale_k2  # Spatial frequency (cycles per degree)
    orientation = 0  # Orientation (degrees)
    L_b = 10  # Luminance of the background
    contrast = 1  # Contrast
    ppd = 60 / scale_k1  # Pixels per degree

    T_vid = generate_sinusoidal_grating(W, H, spatial_frequency, orientation, L_b, contrast, ppd)
    plot_sinusoidal_grating(T_vid, W, H, spatial_frequency, contrast, ppd)
