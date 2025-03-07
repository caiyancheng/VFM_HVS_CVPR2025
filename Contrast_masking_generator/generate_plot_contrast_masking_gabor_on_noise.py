import numpy as np
import matplotlib.pyplot as plt
from display_encoding import display_encode
import cv2

Luminance_min = 1e-4
display_encode_tool = display_encode(400)

def create_cycdeg_image(im_size, pix_per_deg):
    nyquist_freq = 0.5 * pix_per_deg
    KX0 = (np.mod(1 / 2 + np.arange(im_size[1]) / im_size[1], 1) - 1 / 2)
    KX1 = KX0 * nyquist_freq * 2
    KY0 = (np.mod(1 / 2 + np.arange(im_size[0]) / im_size[0], 1) - 1 / 2)
    KY1 = KY0 * nyquist_freq * 2
    XX, YY = np.meshgrid(KX1, KY1)
    D = np.sqrt(XX ** 2 + YY ** 2)
    return D

def generate_test_stimulus_gabor(W, H, sigma, rho, O, L_b, contrast_test, ppd):
    size_deg = np.array([W, H]) / ppd
    XX, YY = np.meshgrid(np.linspace(0, size_deg[0], W),
                         np.linspace(0, size_deg[1], H))
    gauss_env = np.exp(-((XX - size_deg[0] / 2) ** 2 + (YY - size_deg[1] / 2) ** 2) / (2 * sigma ** 2))
    d = XX
    img_target = np.cos(2 * np.pi * d * rho) * contrast_test * L_b * gauss_env
    return img_target

def generate_mask_stimulus_band_limit_noise(W, H, Mask_freq_band, L_b, contrast_mask, contrast_test, ppd):
    np.random.seed(8)
    Noise = np.random.randn(W, H)
    Mask_Noise_f = np.fft.fft2(Noise)
    rho = create_cycdeg_image([W, H], ppd)
    Mask_log2_freq_band = np.log2(Mask_freq_band)
    Mask_freq_edge_low = 2 ** (Mask_log2_freq_band - 0.5)
    Mask_freq_edge_high = 2 ** (Mask_log2_freq_band + 0.5)
    Mask_Noise_f[(rho < Mask_freq_edge_low) | (rho > Mask_freq_edge_high)] = 0
    Mask_Noise_bp = np.real(np.fft.ifft2(Mask_Noise_f))
    Mask_Noise_bp = Mask_Noise_bp / np.std(Mask_Noise_bp)
    img_mask = L_b + Mask_Noise_bp * L_b * contrast_mask
    return img_mask

def generate_mask_stimulus_flat_noise(W, H, Mask_freq_band, L_b, contrast_mask, contrast_test, ppd):
    np.random.seed(8)
    Noise = np.random.randn(W, H)
    Mask_Noise_f = np.fft.fft2(Noise)
    rho = create_cycdeg_image([W, H], ppd)
    Mask_Noise_f[rho > 12] = 0
    Mask_Noise_bp = np.real(np.fft.ifft2(Mask_Noise_f))
    Mask_Noise_bp = Mask_Noise_bp / np.std(Mask_Noise_bp)
    img_mask = L_b + Mask_Noise_bp * L_b * contrast_mask
    return img_mask

def generate_contrast_masking_gabor_on_noise(W, H, sigma, rho, Mask_upper_frequency, L_b, contrast_mask, contrast_test, ppd):
    image_test = generate_test_stimulus_gabor(W, H, sigma, rho, 0, L_b, contrast_test, ppd)
    # image_mask = generate_mask_stimulus_band_limit_noise(W, H, Mask_freq_band, L_b, contrast_mask, contrast_test, ppd)
    image_mask = generate_mask_stimulus_flat_noise(W, H, Mask_upper_frequency, L_b, contrast_mask, contrast_test, ppd)
    T_vid = image_test + image_mask
    R_vid = image_mask
    T_vid[T_vid < Luminance_min] = Luminance_min
    R_vid[R_vid < Luminance_min] = Luminance_min
    return T_vid, R_vid

def plot_contrast_masking_gabor_on_noise(T_vid, R_vid):
    T_vid_c = display_encode_tool.L2C_sRGB(T_vid) * 255
    # R_vid_c = display_encode_tool.L2C_sRGB(R_vid) * 255
    plt.figure(figsize=(4, 4))
    plt.imshow(T_vid_c, cmap='gray', vmin=0, vmax=255, extent=(-W // 2, W // 2, -H // 2, H // 2))
    plt.title(
        f'contrast_mask = {contrast_mask}, contrast_test = {contrast_test}, \n'
        f'rho_test = {rho_test} cpd, \n'
        f'L_b = {L_b} $cd/m^2$,  ppd = {ppd},\n'
        f' W = {W}, H = {H}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_contrast_masking_gabor_on_noise(T_vid, R_vid):
    T_vid_c = np.array(display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8)
    cv2.imwrite("T_vid_c.jpg", T_vid_c)

if __name__ == '__main__':
    W = 224 # Width of the canvas (pixels)
    H = 224 # Height of the canvas (pixels)
    # Mask_freq_band = 12
    Mask_upper_frequency = 12
    L_b = 37  # Luminance of the background
    contrast_mask = 0.1
    contrast_test = 0.5
    ppd = 60
    radius_test = 0.8
    rho_test = 1.2

    T_vid, R_vid = generate_contrast_masking_gabor_on_noise(W, H, radius_test, rho_test, Mask_upper_frequency, L_b, contrast_mask, contrast_test, ppd)
    plot_contrast_masking_gabor_on_noise(T_vid, R_vid)
    # save_contrast_masking_gabor_on_noise(T_vid, R_vid)
