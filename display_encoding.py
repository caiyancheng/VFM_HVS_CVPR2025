import numpy as np
import matplotlib.pyplot as plt
import torch

class display_encode():
    def __init__(self, display_encoded_a):
        self.display_encoded_a = display_encoded_a
    def L2C_gamma(self, Luminance):
        C = (Luminance / self.display_encoded_a) ** (1 / 2.2)
        return C
    def C2L_gamma(self, Color):
        L = self.display_encoded_a * Color ** 2.2
        return L
    def L2C_sRGB(self, Luminance):
        C = np.zeros(Luminance.shape)
        threshold = self.display_encoded_a * 0.04045 / 12.92
        C[Luminance <= threshold] = (Luminance[Luminance <= threshold] * 12.92) / self.display_encoded_a
        C[Luminance > threshold] = 1.055 * (Luminance[Luminance > threshold] / self.display_encoded_a) ** (
                1 / 2.4) - 0.055
        return C
    def C2L_sRGB(self, Color):
        Color = np.ones(Color.shape) * Color
        L = np.zeros(Color.shape)
        L[Color <= 0.04045] = self.display_encoded_a * Color[Color <= 0.04045] / 12.92
        L[Color > 0.04045] = self.display_encoded_a * ((Color[Color > 0.04045] + 0.055) / 1.055) ** 2.4
        return L

    def C2L_sRGB_tensor(self, Color):
        L = torch.zeros(Color.shape)
        L[Color <= 0.04045] = self.display_encoded_a * Color[Color <= 0.04045] / 12.92
        L[Color > 0.04045] = self.display_encoded_a * ((Color[Color > 0.04045] + 0.055) / 1.055) ** 2.4
        return L

if __name__ == "__main__":
    Color_list = np.logspace(np.log10(0.01), np.log10(1))
    display_encode_tool = display_encode(400)
    Gamma_Luminance_list = display_encode_tool.C2L_gamma(Color_list)
    sRGB_Luminance_list = display_encode_tool.C2L_sRGB(Color_list)
    Color_list_Gamma_return = display_encode_tool.L2C_gamma(Gamma_Luminance_list)
    Color_list_sRGB_return = display_encode_tool.L2C_sRGB(sRGB_Luminance_list)
    plt.figure()
    plt.plot(Color_list, Gamma_Luminance_list, 'r')
    plt.plot(Color_list, sRGB_Luminance_list, 'b')
    # plt.xscale('log')
    plt.yscale('log')
    plt.show()
