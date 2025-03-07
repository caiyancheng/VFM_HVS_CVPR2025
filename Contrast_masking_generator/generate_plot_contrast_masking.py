import numpy as np
import matplotlib.pyplot as plt
from display_encoding import display_encode
import cv2

Luminance_min = 1e-4
display_encode_tool = display_encode(400)

def generate_masking_stimulus(W, H, rho, O, L_b, contrast_mask, contrast_test, ppd, gabor_radius):
    size_deg = np.array([W, H]) / ppd
    sigma = gabor_radius
    XX, YY = np.meshgrid(np.linspace(0, size_deg[0], W),
                         np.linspace(0, size_deg[1], H))
    gauss_env = np.exp(-((XX - size_deg[0] / 2) ** 2 + (YY - size_deg[1] / 2) ** 2) / (2 * sigma ** 2))
    DD = np.sqrt(XX ** 2 + YY ** 2)
    cosB = (XX * (-np.sin(np.radians(O))) + YY * np.cos(np.radians(O))) / DD
    cosB[0, 0] = 0  # Avoid division by zero
    d = np.sqrt(1 - cosB ** 2) * DD
    img_mask = np.cos(2 * np.pi * d * rho) * contrast_mask * L_b + L_b
    d = XX
    img_target = np.cos(2 * np.pi * d * rho) * contrast_test * L_b * gauss_env
    S = img_mask + img_target
    return S

def generate_contrast_masking(W, H, rho, O, L_b, contrast_mask, contrast_test, ppd, gabor_radius):
    T_vid = generate_masking_stimulus(W, H, rho, O, L_b, contrast_mask, contrast_test, ppd, gabor_radius)
    R_vid = generate_masking_stimulus(W, H, rho, O, L_b, contrast_mask, 0, ppd, gabor_radius)
    T_vid[T_vid < 0] = 0
    R_vid[R_vid < 0] = 0
    return T_vid, R_vid

def plot_contrast_masking(T_vid, R_vid):
    T_vid_c = display_encode_tool.L2C_sRGB(T_vid) * 255
    # R_vid_c = display_encode_tool.L2C(R_vid) * 255
    plt.figure(figsize=(4, 4))
    plt.imshow(T_vid_c, cmap='gray', vmin=0, vmax=255, extent=(-W // 2, W // 2, -H // 2, H // 2))
    plt.title(
        f'contrast_mask = {contrast_mask}, contrast_test = {contrast_test}, \n rho = {rho} cpd, L_b = {L_b} $cd/m^2$, \n ppd = {ppd}, W = {W}, H = {H}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_contrast_masking(T_vid, R_vid):
    T_vid_c = np.array(display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8)
    cv2.imwrite("T_vid_c.jpg", T_vid_c)

if __name__ == '__main__':
    scale_k1 = 1
    scale_k2 = 1

    W = 224 * scale_k2  # Width of the canvas (pixels)
    H = 224 * scale_k2  # Height of the canvas (pixels)
    rho = 2 / scale_k1 / scale_k2  # Spatial frequency of the Gabor stimulus (cycles per degree)
    O = 0  # Orientation of the Gabor stimulus (degrees)
    L_b = 32 # Luminance of the background
    contrast_mask = 0.2
    contrast_test = 0.5
    ppd = 60 / scale_k1
    gabor_radius = 0.5

    T_vid, R_vid = generate_contrast_masking(W, H, rho, O, L_b, contrast_mask, contrast_test, ppd, gabor_radius)
    plot_contrast_masking(T_vid, R_vid)
    # save_contrast_masking(T_vid, R_vid)

