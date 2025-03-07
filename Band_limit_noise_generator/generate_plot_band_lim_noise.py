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
def generate_band_lim_noise(W, H, freq_band, L_b, contrast, ppd):
    Noise = np.random.randn(W, H)
    Noise_f = np.fft.fft2(Noise)
    rho = create_cycdeg_image([W,H], ppd)
    log2_freq_band = np.log2(freq_band)
    freq_edge_low = 2 ** (log2_freq_band - 0.5)
    freq_edge_high = 2 ** (log2_freq_band + 0.5)
    Noise_f[(rho<freq_edge_low) | (rho>freq_edge_high)] = 0
    Noise_bp = np.real(np.fft.ifft2(Noise_f))
    Noise_bp = Noise_bp / np.std(Noise_bp)
    R_vid = np.ones([W,H]) * L_b #Refrence
    T_vid = np.maximum(Luminance_min, R_vid + Noise_bp * L_b * contrast) #Test
    return T_vid, R_vid

def generate_band_lim_noise_fix_random_seed(W, H, freq_band, L_b, contrast, ppd):
    np.random.seed(8)
    Noise = np.random.randn(W, H)
    Noise_f = np.fft.fft2(Noise)
    rho = create_cycdeg_image([W,H], ppd)
    log2_freq_band = np.log2(freq_band)
    freq_edge_low = 2 ** (log2_freq_band - 0.5)
    freq_edge_high = 2 ** (log2_freq_band + 0.5)
    Noise_f[(rho<freq_edge_low) | (rho>freq_edge_high)] = 0
    Noise_bp = np.real(np.fft.ifft2(Noise_f))
    Noise_bp = Noise_bp / np.std(Noise_bp)
    R_vid = np.ones([W,H]) * L_b #Refrence
    T_vid = np.maximum(Luminance_min, R_vid + Noise_bp * L_b * contrast) #Test
    return T_vid, R_vid

def plot_band_lim_noise(T_vid, R_vid):
    T_vid_c = display_encode_tool.L2C_sRGB(T_vid) * 255
    # R_vid_c = display_encode_tool.L2C(R_vid) * 255
    plt.figure(figsize=(4,4))
    plt.imshow(T_vid_c, cmap='gray', vmin=0, vmax=255, extent=(-W // 2, W // 2, -H // 2, H // 2))
    plt.title(f'Freq_band = {freq_band} cpd, Contrast = {contrast}, \n L_b = {L_b} $cd/m^2$, \n ppd = {ppd}, W = {W}, H = {H}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
def save_band_lim_noise(T_vid, R_vid):
    T_vid_c = np.array(display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8)
    R_vid_c = np.array(display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8)
    cv2.imwrite("T_vid_c.jpg", T_vid_c)

if __name__ == '__main__':
    scale_k1 = 1
    scale_k2 = 1

    W = 224 * scale_k2  # Width of the canvas (pixels)
    H = 224 * scale_k2  # Height of the canvas (pixels)
    freq_band = 1 / scale_k1 / scale_k2
    L_b = 100  # Luminance of the background
    contrast = 0.5  # Contrast of the gabor
    ppd = 60 / scale_k1

    T_vid, R_vid = generate_band_lim_noise_fix_random_seed(W, H, freq_band, L_b, contrast, ppd)
    plot_band_lim_noise(T_vid, R_vid)
    # save_band_lim_noise(T_vid, R_vid)

