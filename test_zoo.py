import sys
import numpy as np
import torch
import math
from Gabor_test_stimulus_generator.generate_plot_gabor_functions_new import generate_gabor_patch
from Band_limit_noise_generator.generate_plot_band_lim_noise import generate_band_lim_noise, generate_band_lim_noise_fix_random_seed
from Sinusoidal_grating_generator.generate_plot_sinusoidal_grating import generate_sinusoidal_grating
from Contrast_masking_generator.generate_plot_contrast_masking import generate_contrast_masking
from Contrast_masking_generator.generate_plot_contrast_masking_gabor_on_noise import generate_contrast_masking_gabor_on_noise
import os
from tqdm import tqdm
import torch.nn.functional as F
import json
from scipy.optimize import minimize, brute, differential_evolution, root_scalar
import gc

class Contrast_Detection_Area:
    def __init__(self, sample_num):
        self.W = 224
        self.H = 224
        self.sample_num = sample_num
        R_min = 0.1
        R_max = 1
        self.Area_list = np.logspace(np.log10(math.pi * R_min ** 2), np.log10(math.pi * R_max ** 2), self.sample_num)
        self.R_list = (self.Area_list / math.pi) ** 0.5
        self.rho = 8
        self.contrast_list = np.logspace(np.log10(0.001), np.log10(1), self.sample_num)
        self.O = 0
        self.L_b = 100
        self.ppd = 60
        self.test_type = 'Contrast Detection - Area'
        self.test_short_name = 'contrast_detection_area'
        self.multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)
        castleCSF_result_json = r'gt_json/castleCSF_area_sensitivity_data.json'
        with open(castleCSF_result_json, 'r') as fp:
            castleCSF_result_data = json.load(fp)
        self.castleCSF_result_area_list = castleCSF_result_data['area_list']
        self.castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

    def test_models(self, model_class_instance_list):
        print(self.test_type)
        for model_class in model_class_instance_list:
            backbone_list = model_class.backbone_list
            model_general_name = model_class.name
            print(model_general_name)
            save_root_path = f'data_logs/test_{self.test_short_name}/{model_general_name}'
            os.makedirs(save_root_path, exist_ok=True)
            json_plot_data = {}
            json_plot_data['json_backbone_name_list'] = []
            for backbone_name in tqdm(backbone_list):
                if isinstance(backbone_name, tuple):
                    json_backbone_name = "_".join(map(str, backbone_name))
                else:
                    json_backbone_name = backbone_name.split('/')[-1]
                json_plot_data[json_backbone_name] = {}
                json_plot_data['json_backbone_name_list'].append(json_backbone_name)
                backbone_model = model_class.load_pretrained(backbone_name)

                # Compute the Spearman Correlation
                Spearman_matrix_score = np.zeros([len(self.Area_list), len(self.multiplier_list)])
                for Area_index, Area_value in enumerate(self.Area_list):
                    S_gt = np.interp(Area_value, self.castleCSF_result_area_list, self.castleCSF_result_sensitivity_list)
                    for multiplier_index, multiplier_value in enumerate(self.multiplier_list):
                        S_test = multiplier_value * S_gt
                        T_L_array, R_L_array = generate_gabor_patch(W=self.W, H=self.H, R=(Area_value / math.pi) ** 0.5,
                                                            rho=self.rho, O=self.O, L_b=self.L_b,
                                                            contrast=1 / S_test, ppd=self.ppd,
                                                            color_direction='ach')
                        if model_general_name.startswith('cvvdp'):
                            JOD_score = float(model_class.compute_score(T_L_array, R_L_array).cpu())
                            Spearman_score = 1 - JOD_score / 10
                        else:
                            test_feature = model_class.forward_feature(backbone_model, T_L_array)
                            reference_feature = model_class.forward_feature(backbone_model, R_L_array)
                            cos_similarity = float(
                                F.cosine_similarity(test_feature.reshape(1, -1),
                                                    reference_feature.reshape(1, -1)).cpu())
                            if cos_similarity > 1:
                                cos_similarity = 1
                            elif cos_similarity < -1:
                                cos_similarity = -1
                            Spearman_score = np.arccos(cos_similarity) / np.arccos(-1)
                        Spearman_matrix_score[Area_index, multiplier_index] = Spearman_score

                # Compute All Scores
                radius_matrix = np.zeros([len(self.R_list), len(self.contrast_list)])
                area_matrix = np.zeros([len(self.R_list), len(self.contrast_list)])
                contrast_matrix = np.zeros([len(self.R_list), len(self.contrast_list)])

                if model_general_name.startswith('cvvdp'):
                    JOD_matrix = np.zeros([len(self.R_list), len(self.contrast_list)])
                    JOD_scale_matrix = np.zeros([len(self.R_list), len(self.contrast_list)])
                else:
                    L1_similarity_matrix = np.zeros([len(self.R_list), len(self.contrast_list)])
                    L2_similarity_matrix = np.zeros([len(self.R_list), len(self.contrast_list)])
                    cos_similarity_matrix = np.zeros([len(self.R_list), len(self.contrast_list)])
                    arccos_cos_similarity_matrix = np.zeros([len(self.R_list), len(self.contrast_list)])
                for R_index in tqdm(range(len(self.R_list))):
                    R_value = self.R_list[R_index]
                    A_value = self.Area_list[R_index]
                    for contrast_index in range(len(self.contrast_list)):
                        contrast_value = self.contrast_list[contrast_index]
                        T_L_array, R_L_array = generate_gabor_patch(W=self.W, H=self.H, R=R_value, rho=self.rho,
                                                                    O=self.O,
                                                                    L_b=self.L_b, contrast=contrast_value, ppd=self.ppd,
                                                                    color_direction='ach')
                        radius_matrix[R_index, contrast_index] = R_value
                        area_matrix[R_index, contrast_index] = A_value
                        contrast_matrix[R_index, contrast_index] = contrast_value
                        if model_general_name.startswith('cvvdp'):
                            JOD_score = model_class.compute_score(T_L_array, R_L_array)
                            JOD_matrix[R_index, contrast_index] = JOD_score
                            JOD_scale_matrix[R_index, contrast_index] = 1 - JOD_score / 10
                        else:
                            test_feature = model_class.forward_feature(backbone_model, T_L_array)
                            reference_feature = model_class.forward_feature(backbone_model, R_L_array)
                            L1_similarity = float(torch.norm(test_feature - reference_feature, p=1).cpu())
                            L2_similarity = float(torch.norm(test_feature - reference_feature, p=2).cpu())
                            cos_similarity = float(
                                F.cosine_similarity(test_feature.reshape(1, -1),
                                                    reference_feature.reshape(1, -1)).cpu())
                            if cos_similarity > 1:
                                cos_similarity = 1
                            elif cos_similarity < -1:
                                cos_similarity = -1
                            L1_similarity_matrix[R_index, contrast_index] = L1_similarity
                            L2_similarity_matrix[R_index, contrast_index] = L2_similarity
                            cos_similarity_matrix[R_index, contrast_index] = cos_similarity
                            arccos_cos_similarity_matrix[R_index, contrast_index] = np.arccos(cos_similarity) / np.arccos(-1)
                    json_plot_data[json_backbone_name]['radius_matrix'] = radius_matrix.tolist()
                    json_plot_data[json_backbone_name]['area_matrix'] = area_matrix.tolist()
                    json_plot_data[json_backbone_name]['contrast_matrix'] = contrast_matrix.tolist()
                    if model_general_name.startswith('cvvdp'):
                        json_plot_data[json_backbone_name]['JOD_matrix'] = JOD_matrix.tolist()
                        json_plot_data[json_backbone_name]['JOD_scale_matrix'] = JOD_scale_matrix.tolist()
                    else:
                        json_plot_data[json_backbone_name]['L1_similarity_matrix'] = L1_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name]['L2_similarity_matrix'] = L2_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name]['cos_similarity_matrix'] = cos_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name][
                            'arccos_cos_similarity_matrix'] = arccos_cos_similarity_matrix.tolist()
                    json_plot_data[json_backbone_name]['area_list'] = self.Area_list.tolist()
                    json_plot_data[json_backbone_name]['multiplier_list'] = self.multiplier_list.tolist()
                    json_plot_data[json_backbone_name]['Spearman_matrix_score'] = Spearman_matrix_score.tolist()

            with open(os.path.join(save_root_path,
                                   f'{model_general_name}_test_{self.test_short_name}.json'),
                      'w') as fp:
                json.dump(json_plot_data, fp)


class Contrast_Detection_Luminance:
    def __init__(self, sample_num):
        self.W = 224
        self.H = 224
        self.sample_num = sample_num
        self.R = 1
        self.rho = 2
        self.contrast_list = np.logspace(np.log10(0.001), np.log10(1), self.sample_num)
        self.O = 0
        self.L_b_list = np.logspace(np.log10(0.1), np.log10(200), self.sample_num)
        self.ppd = 60
        self.test_type = 'Contrast Detection - Luminance'
        self.test_short_name = 'contrast_detection_luminance'
        self.multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)
        castleCSF_result_json = r'gt_json/castleCSF_luminance_sensitivity_data.json'
        with open(castleCSF_result_json, 'r') as fp:
            castleCSF_result_data = json.load(fp)
        self.castleCSF_result_L_list = castleCSF_result_data['luminance_list']
        self.castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

    def test_models(self, model_class_instance_list):
        print(self.test_type)
        for model_class in model_class_instance_list:
            gc.collect()
            backbone_list = model_class.backbone_list
            model_general_name = model_class.name
            print(model_general_name)
            save_root_path = f'data_logs/test_{self.test_short_name}/{model_general_name}'
            os.makedirs(save_root_path, exist_ok=True)
            json_plot_data = {}
            json_plot_data['json_backbone_name_list'] = []
            for backbone_name in tqdm(backbone_list):
                if isinstance(backbone_name, tuple):
                    json_backbone_name = "_".join(map(str, backbone_name))
                else:
                    json_backbone_name = backbone_name.split('/')[-1]
                json_plot_data[json_backbone_name] = {}
                json_plot_data['json_backbone_name_list'].append(json_backbone_name)
                backbone_model = model_class.load_pretrained(backbone_name)

                # Compute the Spearman Correlation
                Spearman_matrix_score = np.zeros([len(self.L_b_list), len(self.multiplier_list)])
                for L_b_index, L_b_value in enumerate(self.L_b_list):
                    S_gt = np.interp(L_b_value, self.castleCSF_result_L_list, self.castleCSF_result_sensitivity_list)
                    for multiplier_index, multiplier_value in enumerate(self.multiplier_list):
                        S_test = multiplier_value * S_gt
                        T_L_array, R_L_array = generate_gabor_patch(W=self.W, H=self.H, R=self.R,
                                                                    rho=self.rho, O=self.O, L_b=L_b_value,
                                                                    contrast=1 / S_test, ppd=self.ppd,
                                                                    color_direction='ach')
                        if model_general_name.startswith('cvvdp'):
                            JOD_score = float(model_class.compute_score(T_L_array, R_L_array).cpu())
                            Spearman_score = 1 - JOD_score / 10
                        else:
                            test_feature = model_class.forward_feature(backbone_model, T_L_array)
                            reference_feature = model_class.forward_feature(backbone_model, R_L_array)
                            cos_similarity = float(
                                F.cosine_similarity(test_feature.reshape(1, -1),
                                                    reference_feature.reshape(1, -1)).cpu())
                            if cos_similarity > 1:
                                cos_similarity = 1
                            elif cos_similarity < -1:
                                cos_similarity = -1
                            Spearman_score = np.arccos(cos_similarity) / np.arccos(-1)
                        Spearman_matrix_score[L_b_index, multiplier_index] = Spearman_score

                # Compute All Scores
                L_b_matrix = np.zeros([len(self.L_b_list), len(self.contrast_list)])
                contrast_matrix = np.zeros([len(self.L_b_list), len(self.contrast_list)])
                if model_general_name.startswith('cvvdp'):
                    JOD_matrix = np.zeros([len(self.L_b_list), len(self.contrast_list)])
                    JOD_scale_matrix = np.zeros([len(self.L_b_list), len(self.contrast_list)])
                else:
                    L1_similarity_matrix = np.zeros([len(self.L_b_list), len(self.contrast_list)])
                    L2_similarity_matrix = np.zeros([len(self.L_b_list), len(self.contrast_list)])
                    cos_similarity_matrix = np.zeros([len(self.L_b_list), len(self.contrast_list)])
                    arccos_cos_similarity_matrix = np.zeros([len(self.L_b_list), len(self.contrast_list)])
                for L_b_index in tqdm(range(len(self.L_b_list))):
                    L_b_value = self.L_b_list[L_b_index]
                    for contrast_index in range(len(self.contrast_list)):
                        contrast_value = self.contrast_list[contrast_index]
                        T_L_array, R_L_array = generate_gabor_patch(W=self.W, H=self.H, R=self.R, rho=self.rho,
                                                                    O=self.O,
                                                                    L_b=L_b_value, contrast=contrast_value, ppd=self.ppd,
                                                                    color_direction='ach')
                        L_b_matrix[L_b_index, contrast_index] = L_b_value
                        contrast_matrix[L_b_index, contrast_index] = contrast_value
                        if model_general_name.startswith('cvvdp'):
                            JOD_score = float(model_class.compute_score(T_L_array, R_L_array).cpu())
                            JOD_matrix[L_b_index, contrast_index] = JOD_score
                            JOD_scale_matrix[L_b_index, contrast_index] = 1 - JOD_score / 10
                        else:
                            test_feature = model_class.forward_feature(backbone_model, T_L_array)
                            reference_feature = model_class.forward_feature(backbone_model, R_L_array)
                            L1_similarity = float(torch.norm(test_feature - reference_feature, p=1).cpu())
                            L2_similarity = float(torch.norm(test_feature - reference_feature, p=2).cpu())
                            cos_similarity = float(
                                F.cosine_similarity(test_feature.reshape(1, -1),
                                                    reference_feature.reshape(1, -1)).cpu())
                            if cos_similarity > 1:
                                cos_similarity = 1
                            elif cos_similarity < -1:
                                cos_similarity = -1
                            L1_similarity_matrix[L_b_index, contrast_index] = L1_similarity
                            L2_similarity_matrix[L_b_index, contrast_index] = L2_similarity
                            cos_similarity_matrix[L_b_index, contrast_index] = cos_similarity
                            arccos_cos_similarity_matrix[L_b_index, contrast_index] = np.arccos(
                                cos_similarity) / np.arccos(-1)
                    json_plot_data[json_backbone_name]['L_b_matrix'] = L_b_matrix.tolist()
                    json_plot_data[json_backbone_name]['contrast_matrix'] = contrast_matrix.tolist()
                    if model_general_name.startswith('cvvdp'):
                        json_plot_data[json_backbone_name]['JOD_matrix'] = JOD_matrix.tolist()
                        json_plot_data[json_backbone_name]['JOD_scale_matrix'] = JOD_scale_matrix.tolist()
                    else:
                        json_plot_data[json_backbone_name]['L1_similarity_matrix'] = L1_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name]['L2_similarity_matrix'] = L2_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name]['cos_similarity_matrix'] = cos_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name][
                            'arccos_cos_similarity_matrix'] = arccos_cos_similarity_matrix.tolist()
                    json_plot_data[json_backbone_name]['L_b_list'] = self.L_b_list.tolist()
                    json_plot_data[json_backbone_name]['multiplier_list'] = self.multiplier_list.tolist()
                    json_plot_data[json_backbone_name]['Spearman_matrix_score'] = Spearman_matrix_score.tolist()
            with open(os.path.join(save_root_path,
                                   f'{model_general_name}_test_{self.test_short_name}.json'),
                      'w') as fp:
                json.dump(json_plot_data, fp)

class Contrast_Detection_SpF_Gabor_Ach:
    def __init__(self, sample_num):
        self.W = 224
        self.H = 224
        self.sample_num = sample_num
        self.R = 1
        self.rho_list = np.logspace(np.log10(0.5), np.log10(32), self.sample_num)
        self.contrast_list = np.logspace(np.log10(0.001), np.log10(1), self.sample_num)
        self.O = 0
        self.L_b = 100
        self.ppd = 60
        self.test_type = 'Contrast Detection - SpF_Gabor_ach'
        self.test_short_name = 'contrast_detection_SpF_Gabor_ach'
        self.multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)
        castleCSF_result_json = r'gt_json/castleCSF_rho_sensitivity_data.json'
        with open(castleCSF_result_json, 'r') as fp:
            castleCSF_result_data = json.load(fp)
        self.castleCSF_result_rho_list = castleCSF_result_data['rho_list']
        self.castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

    def test_models(self, model_class_instance_list):
        print(self.test_type)
        for model_class in model_class_instance_list:
            backbone_list = model_class.backbone_list
            model_general_name = model_class.name
            print(model_general_name)
            save_root_path = f'data_logs/test_{self.test_short_name}/{model_general_name}'
            os.makedirs(save_root_path, exist_ok=True)
            json_plot_data = {}
            json_plot_data['json_backbone_name_list'] = []
            for backbone_name in tqdm(backbone_list):
                if isinstance(backbone_name, tuple):
                    json_backbone_name = "_".join(map(str, backbone_name))
                else:
                    json_backbone_name = backbone_name.split('/')[-1]
                json_plot_data[json_backbone_name] = {}
                json_plot_data['json_backbone_name_list'].append(json_backbone_name)
                backbone_model = model_class.load_pretrained(backbone_name)

                # Compute the Spearman Correlation
                Spearman_matrix_score = np.zeros([len(self.rho_list), len(self.multiplier_list)])
                for rho_index, rho_value in enumerate(self.rho_list):
                    S_gt = np.interp(rho_value, self.castleCSF_result_rho_list, self.castleCSF_result_sensitivity_list)
                    for multiplier_index, multiplier_value in enumerate(self.multiplier_list):
                        S_test = multiplier_value * S_gt
                        T_L_array, R_L_array = generate_gabor_patch(W=self.W, H=self.H, R=self.R,
                                                                    rho=rho_value, O=self.O, L_b=self.L_b,
                                                                    contrast=1 / S_test, ppd=self.ppd,
                                                                    color_direction='ach')
                        if model_general_name.startswith('cvvdp'):
                            JOD_score = float(model_class.compute_score(T_L_array, R_L_array).cpu())
                            Spearman_score = 1 - JOD_score / 10
                        else:
                            test_feature = model_class.forward_feature(backbone_model, T_L_array)
                            reference_feature = model_class.forward_feature(backbone_model, R_L_array)
                            cos_similarity = float(
                                F.cosine_similarity(test_feature.reshape(1, -1),
                                                    reference_feature.reshape(1, -1)).cpu())
                            if cos_similarity > 1:
                                cos_similarity = 1
                            elif cos_similarity < -1:
                                cos_similarity = -1
                            Spearman_score = np.arccos(cos_similarity) / np.arccos(-1)
                        Spearman_matrix_score[rho_index, multiplier_index] = Spearman_score

                # Compute All Scores
                rho_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                contrast_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                if model_general_name.startswith('cvvdp'):
                    JOD_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                    JOD_scale_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                else:
                    L1_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                    L2_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                    cos_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                    arccos_cos_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                for rho_index in tqdm(range(len(self.rho_list))):
                    rho_value = self.rho_list[rho_index]
                    for contrast_index in range(len(self.contrast_list)):
                        contrast_value = self.contrast_list[contrast_index]
                        T_L_array, R_L_array = generate_gabor_patch(W=self.W, H=self.H, R=self.R, rho=rho_value,
                                                                    O=self.O,
                                                                    L_b=self.L_b, contrast=contrast_value, ppd=self.ppd,
                                                                    color_direction='ach')
                        rho_matrix[rho_index, contrast_index] = rho_value
                        contrast_matrix[rho_index, contrast_index] = contrast_value
                        if model_general_name.startswith('cvvdp'):
                            JOD_score = float(model_class.compute_score(T_L_array, R_L_array).cpu())
                            JOD_matrix[rho_index, contrast_index] = JOD_score
                            JOD_scale_matrix[rho_index, contrast_index] = 1 - JOD_score / 10
                        else:
                            test_feature = model_class.forward_feature(backbone_model, T_L_array)
                            reference_feature = model_class.forward_feature(backbone_model, R_L_array)
                            L1_similarity = float(torch.norm(test_feature - reference_feature, p=1).cpu())
                            L2_similarity = float(torch.norm(test_feature - reference_feature, p=2).cpu())
                            cos_similarity = float(
                                F.cosine_similarity(test_feature.reshape(1, -1),
                                                    reference_feature.reshape(1, -1)).cpu())
                            if cos_similarity > 1:
                                cos_similarity = 1
                            elif cos_similarity < -1:
                                cos_similarity = -1
                            L1_similarity_matrix[rho_index, contrast_index] = L1_similarity
                            L2_similarity_matrix[rho_index, contrast_index] = L2_similarity
                            cos_similarity_matrix[rho_index, contrast_index] = cos_similarity
                            arccos_cos_similarity_matrix[rho_index, contrast_index] = np.arccos(
                                cos_similarity) / np.arccos(-1)
                    json_plot_data[json_backbone_name]['rho_matrix'] = rho_matrix.tolist()
                    json_plot_data[json_backbone_name]['contrast_matrix'] = contrast_matrix.tolist()
                    if model_general_name.startswith('cvvdp'):
                        json_plot_data[json_backbone_name]['JOD_matrix'] = JOD_matrix.tolist()
                        json_plot_data[json_backbone_name]['JOD_scale_matrix'] = JOD_scale_matrix.tolist()
                    else:
                        json_plot_data[json_backbone_name]['L1_similarity_matrix'] = L1_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name]['L2_similarity_matrix'] = L2_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name]['cos_similarity_matrix'] = cos_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name][
                            'arccos_cos_similarity_matrix'] = arccos_cos_similarity_matrix.tolist()
                    json_plot_data[json_backbone_name]['rho_list'] = self.rho_list.tolist()
                    json_plot_data[json_backbone_name]['multiplier_list'] = self.multiplier_list.tolist()
                    json_plot_data[json_backbone_name]['Spearman_matrix_score'] = Spearman_matrix_score.tolist()
            with open(os.path.join(save_root_path,
                                   f'{model_general_name}_test_{self.test_short_name}.json'),
                      'w') as fp:
                json.dump(json_plot_data, fp)

class Contrast_Detection_SpF_Noise_Ach:
    def __init__(self, sample_num):
        self.W = 224
        self.H = 224
        self.sample_num = sample_num
        self.rho_list = np.logspace(np.log10(0.5), np.log10(32), self.sample_num)
        self.contrast_list = np.logspace(np.log10(0.001), np.log10(1), self.sample_num)
        self.L_b = 100
        self.ppd = 60
        self.test_type = 'Contrast Detection - SpF_Noise_ach'
        self.test_short_name = 'contrast_detection_SpF_Noise_ach'
        self.multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)
        castleCSF_result_json = r'gt_json/castleCSF_rho_sensitivity_data_band_lim_noise.json'
        with open(castleCSF_result_json, 'r') as fp:
            castleCSF_result_data = json.load(fp)
        self.castleCSF_result_rho_list = castleCSF_result_data['rho_list']
        self.castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

    def test_models(self, model_class_instance_list):
        print(self.test_type)
        for model_class in model_class_instance_list:
            backbone_list = model_class.backbone_list
            model_general_name = model_class.name
            print(model_general_name)
            save_root_path = f'data_logs/test_{self.test_short_name}/{model_general_name}'
            os.makedirs(save_root_path, exist_ok=True)
            json_plot_data = {}
            json_plot_data['json_backbone_name_list'] = []
            for backbone_name in tqdm(backbone_list):
                if isinstance(backbone_name, tuple):
                    json_backbone_name = "_".join(map(str, backbone_name))
                else:
                    json_backbone_name = backbone_name.split('/')[-1]
                json_plot_data[json_backbone_name] = {}
                json_plot_data['json_backbone_name_list'].append(json_backbone_name)
                backbone_model = model_class.load_pretrained(backbone_name)

                # Compute the Spearman Correlation
                Spearman_matrix_score = np.zeros([len(self.rho_list), len(self.multiplier_list)])
                for rho_index, rho_value in enumerate(self.rho_list):
                    S_gt = np.interp(rho_value, self.castleCSF_result_rho_list, self.castleCSF_result_sensitivity_list)
                    for multiplier_index, multiplier_value in enumerate(self.multiplier_list):
                        S_test = multiplier_value * S_gt
                        T_L_array, R_L_array = generate_band_lim_noise_fix_random_seed(W=self.W, H=self.H,
                                                                                       freq_band=rho_value,
                                                                                       L_b=self.L_b,
                                                                                       contrast=1 / S_test,
                                                                                       ppd=self.ppd)
                        T_L_array = np.stack([T_L_array] * 3, axis=-1)
                        R_L_array = np.stack([R_L_array] * 3, axis=-1)
                        if model_general_name.startswith('cvvdp'):
                            JOD_score = float(model_class.compute_score(T_L_array, R_L_array).cpu())
                            Spearman_score = 1 - JOD_score / 10
                        else:
                            test_feature = model_class.forward_feature(backbone_model, T_L_array)
                            reference_feature = model_class.forward_feature(backbone_model, R_L_array)
                            cos_similarity = float(
                                F.cosine_similarity(test_feature.reshape(1, -1),
                                                    reference_feature.reshape(1, -1)).cpu())
                            if cos_similarity > 1:
                                cos_similarity = 1
                            elif cos_similarity < -1:
                                cos_similarity = -1
                            Spearman_score = np.arccos(cos_similarity) / np.arccos(-1)
                        Spearman_matrix_score[rho_index, multiplier_index] = Spearman_score

                rho_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                contrast_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                if model_general_name.startswith('cvvdp'):
                    JOD_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                    JOD_scale_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                else:
                    L1_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                    L2_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                    cos_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                    arccos_cos_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                for rho_index in tqdm(range(len(self.rho_list))):
                    rho_value = self.rho_list[rho_index]
                    for contrast_index in range(len(self.contrast_list)):
                        contrast_value = self.contrast_list[contrast_index]
                        T_L_array, R_L_array = generate_band_lim_noise_fix_random_seed(W=self.W, H=self.H,
                                                                                       freq_band=rho_value,
                                                                                       L_b=self.L_b,
                                                                                       contrast=contrast_value,
                                                                                       ppd=self.ppd)
                        T_L_array = np.stack([T_L_array] * 3, axis=-1)
                        R_L_array = np.stack([R_L_array] * 3, axis=-1)
                        rho_matrix[rho_index, contrast_index] = rho_value
                        contrast_matrix[rho_index, contrast_index] = contrast_value
                        if model_general_name.startswith('cvvdp'):
                            JOD_score = float(model_class.compute_score(T_L_array, R_L_array).cpu())
                            JOD_matrix[rho_index, contrast_index] = JOD_score
                            JOD_scale_matrix[rho_index, contrast_index] = 1 - JOD_score / 10
                        else:
                            test_feature = model_class.forward_feature(backbone_model, T_L_array)
                            reference_feature = model_class.forward_feature(backbone_model, R_L_array)
                            L1_similarity = float(torch.norm(test_feature - reference_feature, p=1).cpu())
                            L2_similarity = float(torch.norm(test_feature - reference_feature, p=2).cpu())
                            cos_similarity = float(
                                F.cosine_similarity(test_feature.reshape(1, -1),
                                                    reference_feature.reshape(1, -1)).cpu())
                            if cos_similarity > 1:
                                cos_similarity = 1
                            elif cos_similarity < -1:
                                cos_similarity = -1
                            L1_similarity_matrix[rho_index, contrast_index] = L1_similarity
                            L2_similarity_matrix[rho_index, contrast_index] = L2_similarity
                            cos_similarity_matrix[rho_index, contrast_index] = cos_similarity
                            arccos_cos_similarity_matrix[rho_index, contrast_index] = np.arccos(
                                cos_similarity) / np.arccos(-1)
                    json_plot_data[json_backbone_name]['rho_matrix'] = rho_matrix.tolist()
                    json_plot_data[json_backbone_name]['contrast_matrix'] = contrast_matrix.tolist()
                    if model_general_name.startswith('cvvdp'):
                        json_plot_data[json_backbone_name]['JOD_matrix'] = JOD_matrix.tolist()
                        json_plot_data[json_backbone_name]['JOD_scale_matrix'] = JOD_scale_matrix.tolist()
                    else:
                        json_plot_data[json_backbone_name]['L1_similarity_matrix'] = L1_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name]['L2_similarity_matrix'] = L2_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name]['cos_similarity_matrix'] = cos_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name][
                            'arccos_cos_similarity_matrix'] = arccos_cos_similarity_matrix.tolist()
                    json_plot_data[json_backbone_name]['rho_list'] = self.rho_list.tolist()
                    json_plot_data[json_backbone_name]['multiplier_list'] = self.multiplier_list.tolist()
                    json_plot_data[json_backbone_name]['Spearman_matrix_score'] = Spearman_matrix_score.tolist()
            with open(os.path.join(save_root_path,
                                   f'{model_general_name}_test_{self.test_short_name}.json'),
                      'w') as fp:
                json.dump(json_plot_data, fp)

class Contrast_Detection_SpF_Gabor_RG:
    def __init__(self, sample_num):
        self.W = 224
        self.H = 224
        self.sample_num = sample_num
        self.R = 1
        self.rho_list = np.logspace(np.log10(0.5), np.log10(32), self.sample_num)
        self.contrast_list = np.logspace(np.log10(0.001), np.log10(0.2), self.sample_num)
        self.O = 0
        self.L_b = 100
        self.ppd = 60
        self.test_type = 'Contrast Detection - SpF_Gabor_RG'
        self.test_short_name = 'contrast_detection_SpF_Gabor_RG'
        self.multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)
        castleCSF_result_json = r'gt_json/castleCSF_rho_sensitivity_data_RG.json'
        with open(castleCSF_result_json, 'r') as fp:
            castleCSF_result_data = json.load(fp)
        self.castleCSF_result_rho_list = castleCSF_result_data['rho_list']
        self.castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

    def test_models(self, model_class_instance_list):
        print(self.test_type)
        for model_class in model_class_instance_list:
            backbone_list = model_class.backbone_list
            model_general_name = model_class.name
            print(model_general_name)
            save_root_path = f'data_logs/test_{self.test_short_name}/{model_general_name}'
            os.makedirs(save_root_path, exist_ok=True)
            json_plot_data = {}
            json_plot_data['json_backbone_name_list'] = []
            for backbone_name in tqdm(backbone_list):
                if isinstance(backbone_name, tuple):
                    json_backbone_name = "_".join(map(str, backbone_name))
                else:
                    json_backbone_name = backbone_name.split('/')[-1]
                json_plot_data[json_backbone_name] = {}
                json_plot_data['json_backbone_name_list'].append(json_backbone_name)
                backbone_model = model_class.load_pretrained(backbone_name)

                # Compute the Spearman Correlation
                Spearman_matrix_score = np.zeros([len(self.rho_list), len(self.multiplier_list)])
                for rho_index, rho_value in enumerate(self.rho_list):
                    S_gt = np.interp(rho_value, self.castleCSF_result_rho_list, self.castleCSF_result_sensitivity_list)
                    for multiplier_index, multiplier_value in enumerate(self.multiplier_list):
                        S_test = multiplier_value * S_gt
                        T_L_array, R_L_array = generate_gabor_patch(W=self.W, H=self.H, R=self.R,
                                                                    rho=rho_value, O=self.O, L_b=self.L_b,
                                                                    contrast=1 / S_test, ppd=self.ppd,
                                                                    color_direction='rg')
                        if model_general_name.startswith('cvvdp'):
                            JOD_score = float(model_class.compute_score(T_L_array, R_L_array).cpu())
                            Spearman_score = 1 - JOD_score / 10
                        else:
                            test_feature = model_class.forward_feature(backbone_model, T_L_array)
                            reference_feature = model_class.forward_feature(backbone_model, R_L_array)
                            cos_similarity = float(
                                F.cosine_similarity(test_feature.reshape(1, -1),
                                                    reference_feature.reshape(1, -1)).cpu())
                            if cos_similarity > 1:
                                cos_similarity = 1
                            elif cos_similarity < -1:
                                cos_similarity = -1
                            Spearman_score = np.arccos(cos_similarity) / np.arccos(-1)
                        Spearman_matrix_score[rho_index, multiplier_index] = Spearman_score

                # Compute All Scores
                rho_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                contrast_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                if model_general_name.startswith('cvvdp'):
                    JOD_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                    JOD_scale_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                else:
                    L1_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                    L2_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                    cos_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                    arccos_cos_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                for rho_index in tqdm(range(len(self.rho_list))):
                    rho_value = self.rho_list[rho_index]
                    for contrast_index in range(len(self.contrast_list)):
                        contrast_value = self.contrast_list[contrast_index]
                        T_L_array, R_L_array = generate_gabor_patch(W=self.W, H=self.H, R=self.R, rho=rho_value,
                                                                    O=self.O,
                                                                    L_b=self.L_b, contrast=contrast_value, ppd=self.ppd,
                                                                    color_direction='rg')
                        rho_matrix[rho_index, contrast_index] = rho_value
                        contrast_matrix[rho_index, contrast_index] = contrast_value
                        if model_general_name.startswith('cvvdp'):
                            JOD_score = float(model_class.compute_score(T_L_array, R_L_array).cpu())
                            JOD_matrix[rho_index, contrast_index] = JOD_score
                            JOD_scale_matrix[rho_index, contrast_index] = 1 - JOD_score / 10
                        else:
                            test_feature = model_class.forward_feature(backbone_model, T_L_array)
                            reference_feature = model_class.forward_feature(backbone_model, R_L_array)
                            L1_similarity = float(torch.norm(test_feature - reference_feature, p=1).cpu())
                            L2_similarity = float(torch.norm(test_feature - reference_feature, p=2).cpu())
                            cos_similarity = float(
                                F.cosine_similarity(test_feature.reshape(1, -1),
                                                    reference_feature.reshape(1, -1)).cpu())
                            if cos_similarity > 1:
                                cos_similarity = 1
                            elif cos_similarity < -1:
                                cos_similarity = -1
                            L1_similarity_matrix[rho_index, contrast_index] = L1_similarity
                            L2_similarity_matrix[rho_index, contrast_index] = L2_similarity
                            cos_similarity_matrix[rho_index, contrast_index] = cos_similarity
                            arccos_cos_similarity_matrix[rho_index, contrast_index] = np.arccos(
                                cos_similarity) / np.arccos(-1)
                    json_plot_data[json_backbone_name]['rho_matrix'] = rho_matrix.tolist()
                    json_plot_data[json_backbone_name]['contrast_matrix'] = contrast_matrix.tolist()
                    if model_general_name.startswith('cvvdp'):
                        json_plot_data[json_backbone_name]['JOD_matrix'] = JOD_matrix.tolist()
                        json_plot_data[json_backbone_name]['JOD_scale_matrix'] = JOD_scale_matrix.tolist()
                    else:
                        json_plot_data[json_backbone_name]['L1_similarity_matrix'] = L1_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name]['L2_similarity_matrix'] = L2_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name]['cos_similarity_matrix'] = cos_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name][
                            'arccos_cos_similarity_matrix'] = arccos_cos_similarity_matrix.tolist()
                    json_plot_data[json_backbone_name]['rho_list'] = self.rho_list.tolist()
                    json_plot_data[json_backbone_name]['multiplier_list'] = self.multiplier_list.tolist()
                    json_plot_data[json_backbone_name]['Spearman_matrix_score'] = Spearman_matrix_score.tolist()
            with open(os.path.join(save_root_path,
                                   f'{model_general_name}_test_{self.test_short_name}.json'),
                      'w') as fp:
                json.dump(json_plot_data, fp)

class Contrast_Detection_SpF_Gabor_YV:
    def __init__(self, sample_num):
        self.W = 224
        self.H = 224
        self.sample_num = sample_num
        self.R = 1
        self.rho_list = np.logspace(np.log10(0.5), np.log10(32), self.sample_num)
        self.contrast_list = np.logspace(np.log10(0.001), np.log10(0.2), self.sample_num)
        self.O = 0
        self.L_b = 100
        self.ppd = 60
        self.test_type = 'Contrast Detection - SpF_Gabor_YV'
        self.test_short_name = 'contrast_detection_SpF_Gabor_YV'
        self.multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)
        castleCSF_result_json = r'gt_json/castleCSF_rho_sensitivity_data_YV.json'
        with open(castleCSF_result_json, 'r') as fp:
            castleCSF_result_data = json.load(fp)
        self.castleCSF_result_rho_list = castleCSF_result_data['rho_list']
        self.castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

    def test_models(self, model_class_instance_list):
        print(self.test_type)
        for model_class in model_class_instance_list:
            backbone_list = model_class.backbone_list
            model_general_name = model_class.name
            print(model_general_name)
            save_root_path = f'data_logs/test_{self.test_short_name}/{model_general_name}'
            os.makedirs(save_root_path, exist_ok=True)
            json_plot_data = {}
            json_plot_data['json_backbone_name_list'] = []
            for backbone_name in tqdm(backbone_list):
                if isinstance(backbone_name, tuple):
                    json_backbone_name = "_".join(map(str, backbone_name))
                else:
                    json_backbone_name = backbone_name.split('/')[-1]
                json_plot_data[json_backbone_name] = {}
                json_plot_data['json_backbone_name_list'].append(json_backbone_name)
                backbone_model = model_class.load_pretrained(backbone_name)

                # Compute the Spearman Correlation
                rho_change_list = self.rho_list[self.rho_list < 16]
                Spearman_matrix_score = np.zeros([len(rho_change_list), len(self.multiplier_list)])
                for rho_index, rho_value in enumerate(rho_change_list):
                    S_gt = np.interp(rho_value, self.castleCSF_result_rho_list, self.castleCSF_result_sensitivity_list)
                    for multiplier_index, multiplier_value in enumerate(self.multiplier_list):
                        S_test = multiplier_value * S_gt
                        T_L_array, R_L_array = generate_gabor_patch(W=self.W, H=self.H, R=self.R,
                                                                    rho=rho_value, O=self.O, L_b=self.L_b,
                                                                    contrast=1 / S_test, ppd=self.ppd,
                                                                    color_direction='yv')
                        if model_general_name.startswith('cvvdp'):
                            JOD_score = float(model_class.compute_score(T_L_array, R_L_array).cpu())
                            Spearman_score = 1 - JOD_score / 10
                        else:
                            test_feature = model_class.forward_feature(backbone_model, T_L_array)
                            reference_feature = model_class.forward_feature(backbone_model, R_L_array)
                            cos_similarity = float(
                                F.cosine_similarity(test_feature.reshape(1, -1),
                                                    reference_feature.reshape(1, -1)).cpu())
                            if cos_similarity > 1:
                                cos_similarity = 1
                            elif cos_similarity < -1:
                                cos_similarity = -1
                            Spearman_score = np.arccos(cos_similarity) / np.arccos(-1)
                        Spearman_matrix_score[rho_index, multiplier_index] = Spearman_score

                # Compute All Scores
                rho_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                contrast_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                if model_general_name.startswith('cvvdp'):
                    JOD_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                    JOD_scale_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                else:
                    L1_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                    L2_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                    cos_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                    arccos_cos_similarity_matrix = np.zeros([len(self.rho_list), len(self.contrast_list)])
                for rho_index in tqdm(range(len(self.rho_list))):
                    rho_value = self.rho_list[rho_index]
                    for contrast_index in range(len(self.contrast_list)):
                        contrast_value = self.contrast_list[contrast_index]
                        T_L_array, R_L_array = generate_gabor_patch(W=self.W, H=self.H, R=self.R, rho=rho_value,
                                                                    O=self.O,
                                                                    L_b=self.L_b, contrast=contrast_value,
                                                                    ppd=self.ppd,
                                                                    color_direction='yv')
                        rho_matrix[rho_index, contrast_index] = rho_value
                        contrast_matrix[rho_index, contrast_index] = contrast_value
                        if model_general_name.startswith('cvvdp'):
                            JOD_score = float(model_class.compute_score(T_L_array, R_L_array).cpu())
                            JOD_matrix[rho_index, contrast_index] = JOD_score
                            JOD_scale_matrix[rho_index, contrast_index] = 1 - JOD_score / 10
                        else:
                            test_feature = model_class.forward_feature(backbone_model, T_L_array)
                            reference_feature = model_class.forward_feature(backbone_model, R_L_array)
                            L1_similarity = float(torch.norm(test_feature - reference_feature, p=1).cpu())
                            L2_similarity = float(torch.norm(test_feature - reference_feature, p=2).cpu())
                            cos_similarity = float(
                                F.cosine_similarity(test_feature.reshape(1, -1),
                                                    reference_feature.reshape(1, -1)).cpu())
                            if cos_similarity > 1:
                                cos_similarity = 1
                            elif cos_similarity < -1:
                                cos_similarity = -1
                            L1_similarity_matrix[rho_index, contrast_index] = L1_similarity
                            L2_similarity_matrix[rho_index, contrast_index] = L2_similarity
                            cos_similarity_matrix[rho_index, contrast_index] = cos_similarity
                            arccos_cos_similarity_matrix[rho_index, contrast_index] = np.arccos(
                                cos_similarity) / np.arccos(-1)
                    json_plot_data[json_backbone_name]['rho_matrix'] = rho_matrix.tolist()
                    json_plot_data[json_backbone_name]['contrast_matrix'] = contrast_matrix.tolist()
                    if model_general_name.startswith('cvvdp'):
                        json_plot_data[json_backbone_name]['JOD_matrix'] = JOD_matrix.tolist()
                        json_plot_data[json_backbone_name]['JOD_scale_matrix'] = JOD_scale_matrix.tolist()
                    else:
                        json_plot_data[json_backbone_name]['L1_similarity_matrix'] = L1_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name]['L2_similarity_matrix'] = L2_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name]['cos_similarity_matrix'] = cos_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name][
                            'arccos_cos_similarity_matrix'] = arccos_cos_similarity_matrix.tolist()
                    json_plot_data[json_backbone_name]['rho_list'] = self.rho_list.tolist()
                    json_plot_data[json_backbone_name]['multiplier_list'] = self.multiplier_list.tolist()
                    json_plot_data[json_backbone_name]['Spearman_matrix_score'] = Spearman_matrix_score.tolist()
            with open(os.path.join(save_root_path,
                                   f'{model_general_name}_test_{self.test_short_name}.json'),
                      'w') as fp:
                json.dump(json_plot_data, fp)

class Contrast_Masking_Phase_Coherent:
    def __init__(self, sample_num):
        self.W = 224
        self.H = 224
        self.sample_num = sample_num
        self.rho = 2
        self.O = 0
        self.L_b = 32
        self.contrast_mask_list = np.logspace(np.log10(0.005), np.log10(0.5), self.sample_num)
        self.contrast_test_list = np.logspace(np.log10(0.01), np.log10(0.5), self.sample_num)
        self.ppd = 60
        self.R = 0.5
        self.test_type = 'Contrast Masking - Phase-Coherent Masking'
        self.test_short_name = 'contrast_masking_phase_coherent_masking'
        self.multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)
        foley_result_json = r'gt_json/foley_contrast_masking_data_gabor.json'
        with open(foley_result_json, 'r') as fp:
            foley_result_data = json.load(fp)
        foley_result_x_mask_contrast_list = np.array(foley_result_data['mask_contrast_list'])
        foley_result_y_test_contrast_list = np.array(foley_result_data['test_contrast_list'])
        valid_gt_indices = [index for index, value in enumerate(foley_result_x_mask_contrast_list) if
                            value > self.contrast_mask_list.min() and value < 0.25]
        self.gt_x_mask_C = foley_result_x_mask_contrast_list[valid_gt_indices]
        self.gt_y_test_C = foley_result_y_test_contrast_list[valid_gt_indices]

    def test_models(self, model_class_instance_list):
        print(self.test_type)
        for model_class in model_class_instance_list:
            backbone_list = model_class.backbone_list
            model_general_name = model_class.name
            print(model_general_name)
            save_root_path = f'data_logs/test_{self.test_short_name}/{model_general_name}'
            os.makedirs(save_root_path, exist_ok=True)
            json_plot_data = {}
            json_plot_data['json_backbone_name_list'] = []
            for backbone_name in tqdm(backbone_list):
                if isinstance(backbone_name, tuple):
                    json_backbone_name = "_".join(map(str, backbone_name))
                else:
                    json_backbone_name = backbone_name.split('/')[-1]
                json_plot_data[json_backbone_name] = {}
                json_plot_data['json_backbone_name_list'].append(json_backbone_name)
                backbone_model = model_class.load_pretrained(backbone_name)

                # Compute the Spearman Correlation
                Spearman_matrix_score = np.zeros([len(self.gt_x_mask_C), len(self.multiplier_list)])
                for contrast_mask_index in tqdm(range(len(self.gt_x_mask_C))):
                    contrast_mask_value = self.gt_x_mask_C[contrast_mask_index]
                    contrast_test_value = self.gt_y_test_C[contrast_mask_index]
                    for multiplier_index, multiplier_value in enumerate(self.multiplier_list):
                        C_test = contrast_test_value * multiplier_value
                        T_L_array, R_L_array = generate_contrast_masking(W=self.W, H=self.H, rho=self.rho, O=self.O,
                                                     L_b=self.L_b, contrast_mask=contrast_mask_value,
                                                     contrast_test=C_test, ppd=self.ppd,
                                                     gabor_radius=self.R)
                        T_L_array = np.stack([T_L_array] * 3, axis=-1)
                        R_L_array = np.stack([R_L_array] * 3, axis=-1)
                        if model_general_name.startswith('cvvdp'):
                            JOD_score = float(model_class.compute_score(T_L_array, R_L_array).cpu())
                            Spearman_score = 1 - JOD_score / 10
                        else:
                            test_feature = model_class.forward_feature(backbone_model, T_L_array)
                            reference_feature = model_class.forward_feature(backbone_model, R_L_array)
                            cos_similarity = float(
                                F.cosine_similarity(test_feature.reshape(1, -1),
                                                    reference_feature.reshape(1, -1)).cpu())
                            if cos_similarity > 1:
                                cos_similarity = 1
                            elif cos_similarity < -1:
                                cos_similarity = -1
                            Spearman_score = np.arccos(cos_similarity) / np.arccos(-1)
                        Spearman_matrix_score[contrast_mask_index, multiplier_index] = Spearman_score

                # Compute All Scores
                contrast_mask_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
                contrast_test_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
                if model_general_name.startswith('cvvdp'):
                    JOD_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
                    JOD_scale_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
                else:
                    L1_similarity_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
                    L2_similarity_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
                    cos_similarity_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
                    arccos_cos_similarity_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
                for contrast_mask_index in range(len(self.contrast_mask_list)):
                    contrast_mask_value = self.contrast_mask_list[contrast_mask_index]
                    for contrast_test_index in range(len(self.contrast_test_list)):
                        contrast_test_value = self.contrast_test_list[contrast_test_index]
                        T_L_array, R_L_array = generate_contrast_masking(W=self.W, H=self.H, rho=self.rho, O=self.O,
                                                                         L_b=self.L_b, contrast_mask=contrast_mask_value,
                                                                         contrast_test=contrast_test_value, ppd=self.ppd,
                                                                         gabor_radius=self.R)
                        T_L_array = np.stack([T_L_array] * 3, axis=-1)
                        R_L_array = np.stack([R_L_array] * 3, axis=-1)
                        contrast_mask_matrix[contrast_mask_index, contrast_test_index] = contrast_mask_value
                        contrast_test_matrix[contrast_mask_index, contrast_test_index] = contrast_test_value
                        if model_general_name.startswith('cvvdp'):
                            JOD_score = float(model_class.compute_score(T_L_array, R_L_array).cpu())
                            JOD_matrix[contrast_mask_index, contrast_test_index] = JOD_score
                            JOD_scale_matrix[contrast_mask_index, contrast_test_index] = 1 - JOD_score / 10
                        else:
                            test_feature = model_class.forward_feature(backbone_model, T_L_array)
                            reference_feature = model_class.forward_feature(backbone_model, R_L_array)
                            L1_similarity = float(torch.norm(test_feature - reference_feature, p=1).cpu())
                            L2_similarity = float(torch.norm(test_feature - reference_feature, p=2).cpu())
                            cos_similarity = float(
                                F.cosine_similarity(test_feature.reshape(1, -1),
                                                    reference_feature.reshape(1, -1)).cpu())
                            if cos_similarity > 1:
                                cos_similarity = 1
                            elif cos_similarity < -1:
                                cos_similarity = -1
                            L1_similarity_matrix[contrast_mask_index, contrast_test_index] = L1_similarity
                            L2_similarity_matrix[contrast_mask_index, contrast_test_index] = L2_similarity
                            cos_similarity_matrix[contrast_mask_index, contrast_test_index] = cos_similarity
                            arccos_cos_similarity_matrix[contrast_mask_index, contrast_test_index] = np.arccos(
                                cos_similarity) / np.arccos(-1)
                    json_plot_data[json_backbone_name]['contrast_mask_matrix'] = contrast_mask_matrix.tolist()
                    json_plot_data[json_backbone_name]['contrast_test_matrix'] = contrast_test_matrix.tolist()
                    if model_general_name.startswith('cvvdp'):
                        json_plot_data[json_backbone_name]['JOD_matrix'] = JOD_matrix.tolist()
                        json_plot_data[json_backbone_name]['JOD_scale_matrix'] = JOD_scale_matrix.tolist()
                    else:
                        json_plot_data[json_backbone_name]['L1_similarity_matrix'] = L1_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name]['L2_similarity_matrix'] = L2_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name]['cos_similarity_matrix'] = cos_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name][
                            'arccos_cos_similarity_matrix'] = arccos_cos_similarity_matrix.tolist()
                    json_plot_data[json_backbone_name]['multiplier_list'] = self.multiplier_list.tolist()
                    json_plot_data[json_backbone_name]['Spearman_matrix_score'] = Spearman_matrix_score.tolist()
            with open(os.path.join(save_root_path,
                                   f'{model_general_name}_test_{self.test_short_name}.json'),
                      'w') as fp:
                json.dump(json_plot_data, fp)

class Contrast_Masking_Phase_Incoherent:
    def __init__(self, sample_num):
        self.W = 224
        self.H = 224
        self.sample_num = sample_num
        self.Mask_upper_frequency = 12
        self.L_b = 37
        self.contrast_mask_list = np.logspace(np.log10(0.005), np.log10(0.5), self.sample_num)
        self.contrast_test_list = np.logspace(np.log10(0.01), np.log10(0.5), self.sample_num)
        self.ppd = 60
        self.R = 0.8
        self.rho_test = 1.2
        self.test_type = 'Contrast Masking - Phase-Incoherent Masking'
        self.test_short_name = 'contrast_masking_phase_incoherent_masking'
        self.multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)
        foley_result_json = r'gt_json/contrast_masking_data_gabor_on_noise.json'
        with open(foley_result_json, 'r') as fp:
            foley_result_data = json.load(fp)
        foley_result_x_mask_contrast_list = np.array(foley_result_data['mask_contrast_list'])
        foley_result_y_test_contrast_list = np.array(foley_result_data['test_contrast_list'])
        valid_gt_indices = [index for index, value in enumerate(foley_result_x_mask_contrast_list) if
                            value > self.contrast_mask_list.min() and value < 0.25]
        self.gt_x_mask_C = foley_result_x_mask_contrast_list[valid_gt_indices]
        self.gt_y_test_C = foley_result_y_test_contrast_list[valid_gt_indices]

    def test_models(self, model_class_instance_list):
        print(self.test_type)
        for model_class in model_class_instance_list:
            backbone_list = model_class.backbone_list
            model_general_name = model_class.name
            print(model_general_name)
            save_root_path = f'data_logs/test_{self.test_short_name}/{model_general_name}'
            os.makedirs(save_root_path, exist_ok=True)
            json_plot_data = {}
            json_plot_data['json_backbone_name_list'] = []
            for backbone_name in tqdm(backbone_list):
                if isinstance(backbone_name, tuple):
                    json_backbone_name = "_".join(map(str, backbone_name))
                else:
                    json_backbone_name = backbone_name.split('/')[-1]
                json_plot_data[json_backbone_name] = {}
                json_plot_data['json_backbone_name_list'].append(json_backbone_name)
                backbone_model = model_class.load_pretrained(backbone_name)

                # Compute the Spearman Correlation
                Spearman_matrix_score = np.zeros([len(self.gt_x_mask_C), len(self.multiplier_list)])
                for contrast_mask_index in tqdm(range(len(self.gt_x_mask_C))):
                    contrast_mask_value = self.gt_x_mask_C[contrast_mask_index]
                    contrast_test_value = self.gt_y_test_C[contrast_mask_index]
                    for multiplier_index, multiplier_value in enumerate(self.multiplier_list):
                        C_test = contrast_test_value * multiplier_value
                        T_L_array, R_L_array = generate_contrast_masking_gabor_on_noise(W=self.W, H=self.H,
                                                                    sigma=self.R,
                                                                    rho=self.rho_test,
                                                                    Mask_upper_frequency=self.Mask_upper_frequency,
                                                                    L_b=self.L_b,
                                                                    contrast_mask=contrast_mask_value,
                                                                    contrast_test=C_test,
                                                                    ppd=self.ppd)
                        T_L_array = np.stack([T_L_array] * 3, axis=-1)
                        R_L_array = np.stack([R_L_array] * 3, axis=-1)
                        if model_general_name.startswith('cvvdp'):
                            JOD_score = float(model_class.compute_score(T_L_array, R_L_array).cpu())
                            Spearman_score = 1 - JOD_score / 10
                        else:
                            test_feature = model_class.forward_feature(backbone_model, T_L_array)
                            reference_feature = model_class.forward_feature(backbone_model, R_L_array)
                            cos_similarity = float(
                                F.cosine_similarity(test_feature.reshape(1, -1),
                                                    reference_feature.reshape(1, -1)).cpu())
                            if cos_similarity > 1:
                                cos_similarity = 1
                            elif cos_similarity < -1:
                                cos_similarity = -1
                            Spearman_score = np.arccos(cos_similarity) / np.arccos(-1)
                        Spearman_matrix_score[contrast_mask_index, multiplier_index] = Spearman_score

                # Compute All Scores
                contrast_mask_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
                contrast_test_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
                if model_general_name.startswith('cvvdp'):
                    JOD_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
                    JOD_scale_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
                else:
                    L1_similarity_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
                    L2_similarity_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
                    cos_similarity_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
                    arccos_cos_similarity_matrix = np.zeros([len(self.contrast_mask_list), len(self.contrast_test_list)])
                for contrast_mask_index in range(len(self.contrast_mask_list)):
                    contrast_mask_value = self.contrast_mask_list[contrast_mask_index]
                    for contrast_test_index in range(len(self.contrast_test_list)):
                        contrast_test_value = self.contrast_test_list[contrast_test_index]
                        T_L_array, R_L_array = generate_contrast_masking_gabor_on_noise(W=self.W, H=self.H,
                                                                    sigma=self.R,
                                                                    rho=self.rho_test,
                                                                    Mask_upper_frequency=self.Mask_upper_frequency,
                                                                    L_b=self.L_b,
                                                                    contrast_mask=contrast_mask_value,
                                                                    contrast_test=contrast_test_value,
                                                                    ppd=self.ppd)
                        T_L_array = np.stack([T_L_array] * 3, axis=-1)
                        R_L_array = np.stack([R_L_array] * 3, axis=-1)
                        contrast_mask_matrix[contrast_mask_index, contrast_test_index] = contrast_mask_value
                        contrast_test_matrix[contrast_mask_index, contrast_test_index] = contrast_test_value
                        if model_general_name.startswith('cvvdp'):
                            JOD_score = float(model_class.compute_score(T_L_array, R_L_array).cpu())
                            JOD_matrix[contrast_mask_index, contrast_test_index] = JOD_score
                            JOD_scale_matrix[contrast_mask_index, contrast_test_index] = 1 - JOD_score / 10
                        else:
                            test_feature = model_class.forward_feature(backbone_model, T_L_array)
                            reference_feature = model_class.forward_feature(backbone_model, R_L_array)
                            L1_similarity = float(torch.norm(test_feature - reference_feature, p=1).cpu())
                            L2_similarity = float(torch.norm(test_feature - reference_feature, p=2).cpu())
                            cos_similarity = float(
                                F.cosine_similarity(test_feature.reshape(1, -1),
                                                    reference_feature.reshape(1, -1)).cpu())
                            if cos_similarity > 1:
                                cos_similarity = 1
                            elif cos_similarity < -1:
                                cos_similarity = -1
                            L1_similarity_matrix[contrast_mask_index, contrast_test_index] = L1_similarity
                            L2_similarity_matrix[contrast_mask_index, contrast_test_index] = L2_similarity
                            cos_similarity_matrix[contrast_mask_index, contrast_test_index] = cos_similarity
                            arccos_cos_similarity_matrix[contrast_mask_index, contrast_test_index] = np.arccos(
                                cos_similarity) / np.arccos(-1)
                    json_plot_data[json_backbone_name]['contrast_mask_matrix'] = contrast_mask_matrix.tolist()
                    json_plot_data[json_backbone_name]['contrast_test_matrix'] = contrast_test_matrix.tolist()
                    if model_general_name.startswith('cvvdp'):
                        json_plot_data[json_backbone_name]['JOD_matrix'] = JOD_matrix.tolist()
                        json_plot_data[json_backbone_name]['JOD_scale_matrix'] = JOD_scale_matrix.tolist()
                    else:
                        json_plot_data[json_backbone_name]['L1_similarity_matrix'] = L1_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name]['L2_similarity_matrix'] = L2_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name]['cos_similarity_matrix'] = cos_similarity_matrix.tolist()
                        json_plot_data[json_backbone_name][
                            'arccos_cos_similarity_matrix'] = arccos_cos_similarity_matrix.tolist()
                    json_plot_data[json_backbone_name]['multiplier_list'] = self.multiplier_list.tolist()
                    json_plot_data[json_backbone_name]['Spearman_matrix_score'] = Spearman_matrix_score.tolist()
            with open(os.path.join(save_root_path,
                                   f'{model_general_name}_test_{self.test_short_name}.json'),
                      'w') as fp:
                json.dump(json_plot_data, fp)


class Contrast_Matching_cos_scale_solve_no_scaler:
    def __init__(self, sample_num):
        self.W = 224
        self.H = 224
        self.sample_num = sample_num
        self.R = 1
        self.rho_referenece = 5
        rho_test_list_gt = [0.25, 0.5, 1, 2, 5, 10, 15, 20, 25]
        rho_test_list_2 = np.logspace(np.log10(0.25), np.log10(25), 20).tolist()
        rho_test_list_gt = [round(x, 3) for x in rho_test_list_gt]
        rho_test_list_2 = [round(x, 3) for x in rho_test_list_2]
        self.rho_test_list = sorted(set(rho_test_list_gt + rho_test_list_2))
        self.O = 0
        self.L_b = 10
        self.ppd = 60
        self.test_type = 'Contrast Matching Cos Scale Solve No scaler'
        self.test_short_name = 'contrast_matching_cos_scale_solve_no_scaler'
        with open(r'gt_json/contrast_constancy_sin_5_cpd.json', 'r') as fp:
            json_data = json.load(fp)
        self.reference_contrast_list = json_data['average_reference_contrast']
        self.generate_pre_figures()

    def generate_pre_figures(self):
        self.R_L_array_matrix = np.zeros([len(self.reference_contrast_list), self.H, self.W, 3])
        for reference_contrast_index in range(len(self.reference_contrast_list)):
            reference_contrast_value = self.reference_contrast_list[reference_contrast_index]
            R_L_array = generate_sinusoidal_grating(W=self.W, H=self.H, spatial_frequency=self.rho_referenece,
                                                    orientation=self.O, L_b=self.L_b,
                                                    contrast=reference_contrast_value, ppd=self.ppd)
            R_L_array = np.stack([R_L_array] * 3, axis=-1)
            self.R_L_array_matrix[reference_contrast_index, ...] = R_L_array
        R_L_array_Con_0 = generate_sinusoidal_grating(W=self.W, H=self.H,
                                                      spatial_frequency=self.rho_referenece,
                                                      orientation=self.O, L_b=self.L_b,
                                                      contrast=0, ppd=self.ppd)
        self.R_L_array_Con_0 = np.stack([R_L_array_Con_0] * 3, axis=-1)

    def T_optimize_target(self, contrast, rho, model_class, backbone_model, R_feature_Con_0, contrast_matching_scale):
        T_L_array = generate_sinusoidal_grating(W=self.W, H=self.H,
                                                spatial_frequency=rho,
                                                orientation=self.O, L_b=self.L_b,
                                                contrast=contrast, ppd=self.ppd)
        T_L_array = np.stack([T_L_array] * 3, axis=-1)
        T_feature = model_class.forward_feature(backbone_model, T_L_array)
        T_score = max(min(float(F.cosine_similarity(R_feature_Con_0.reshape(1, -1), T_feature.reshape(1, -1)).cpu()), 1), -1)
        T_score_scale = np.arccos(T_score) / np.arccos(-1)
        T_matching_scale = T_score_scale
        return T_matching_scale - contrast_matching_scale

    def T_optimize_target_cvvdp(self, contrast, rho, model_class, R_L_array_Con_0, contrast_matching_scale):
        T_L_array = generate_sinusoidal_grating(W=self.W, H=self.H,
                                                spatial_frequency=rho,
                                                orientation=self.O, L_b=self.L_b,
                                                contrast=contrast, ppd=self.ppd)
        T_L_array = np.stack([T_L_array] * 3, axis=-1)
        T_score = float(model_class.compute_score(R_L_array_Con_0, T_L_array).cpu())
        T_score_scale = 10 - T_score
        T_matching_scale = T_score_scale / 10
        return T_matching_scale - contrast_matching_scale

    def test_models(self, model_class_instance_list):
        print(self.test_type)
        for model_class in model_class_instance_list:
            backbone_list = model_class.backbone_list
            model_general_name = model_class.name
            print(model_general_name)
            save_root_path = f'data_logs/test_{self.test_short_name}/{model_general_name}'
            os.makedirs(save_root_path, exist_ok=True)
            json_plot_data = {}
            json_plot_data['json_backbone_name_list'] = []
            json_plot_data['reference_contrast_list'] = self.reference_contrast_list
            json_plot_data['rho_test_list'] = self.rho_test_list
            for backbone_name in tqdm(backbone_list):
                if isinstance(backbone_name, tuple):
                    json_backbone_name = "_".join(map(str, backbone_name))
                else:
                    json_backbone_name = backbone_name.split('/')[-1]
                json_plot_data[json_backbone_name] = {}
                json_plot_data['json_backbone_name_list'].append(json_backbone_name)
                backbone_model = model_class.load_pretrained(backbone_name)
                if not model_general_name.startswith('cvvdp'):
                    R_L_array_Con_0_feature = model_class.forward_feature(backbone_model, self.R_L_array_Con_0)
                for reference_contrast_index in tqdm(range(len(self.reference_contrast_list))):
                    reference_contrast_value = self.reference_contrast_list[reference_contrast_index]
                    R_L_array = self.R_L_array_matrix[reference_contrast_index]
                    if model_general_name.startswith('cvvdp'):
                        R_score = float(model_class.compute_score(self.R_L_array_Con_0, R_L_array).cpu())
                        R_score_scale = (10 - R_score) / 10
                    else:
                        R_feature = model_class.forward_feature(backbone_model, R_L_array)
                        R_score = max(min(float(F.cosine_similarity(R_L_array_Con_0_feature.reshape(1, -1), R_feature.reshape(1, -1)).cpu()), 1), -1)
                        R_score_scale = np.arccos(R_score) / np.arccos(-1)
                    contrast_matching_scale = R_score_scale
                    contrast_matching_scale = max(min(contrast_matching_scale, 1), 0)
                    match_test_contrast_list = []
                    for rho_test_index in range(len(self.rho_test_list)):
                        rho_test_value = self.rho_test_list[rho_test_index]
                        bounds = [0.001, 1]
                        if model_general_name.startswith('cvvdp'):
                            target_function = lambda test_contrast: self.T_optimize_target_cvvdp(
                                contrast=test_contrast, rho=rho_test_value, model_class=model_class, R_L_array_Con_0=self.R_L_array_Con_0, contrast_matching_scale=contrast_matching_scale)
                        else:
                            target_function = lambda test_contrast: self.T_optimize_target(
                                contrast=test_contrast, rho=rho_test_value, model_class=model_class, R_feature_Con_0=R_L_array_Con_0_feature,
                                backbone_model=backbone_model, contrast_matching_scale=contrast_matching_scale)
                        if (target_function(bounds[0]) < 0) and (target_function(bounds[1]) < 0):
                            test_contrast_optimization_value = 1
                        elif (target_function(bounds[0]) > 0) and (target_function(bounds[1]) > 0):
                            test_contrast_optimization_value = 0.001
                        else:
                            test_contrast_optimization_result = root_scalar(target_function, bracket=bounds, xtol=1e-5)
                            test_contrast_optimization_value = float(test_contrast_optimization_result.root)
                        match_test_contrast_list.append(test_contrast_optimization_value)
                    json_plot_data[json_backbone_name][
                        f'ref_contrast_{reference_contrast_value}_test_contrast_list'] = match_test_contrast_list
            # json_plot_data = convert_numpy_to_python(json_plot_data)
            with open(os.path.join(save_root_path, f'{model_general_name}_test_{self.test_short_name}.json'), 'w') as fp:
                json.dump(json_plot_data, fp)

