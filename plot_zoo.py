import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import json
from tqdm import tqdm

class plot_visual:
    def __init__(self, figure_size=(4.3,3), dpi=300, level_num=50):
        self.plot_classes = ['contrast_detection_SpF_Gabor_ach', 'contrast_detection_SpF_Noise_ach',
                             'contrast_detection_SpF_Gabor_RG', 'contrast_detection_SpF_Gabor_YV',
                             'contrast_detection_luminance', 'contrast_detection_area',
                             'contrast_masking_phase_coherent_masking', 'contrast_masking_phase_incoherent_masking',
                             'contrast_matching_cos_scale_solve']
        self.figure_size = figure_size
        self.dpi = dpi
        self.level_num = level_num
        self.data_save_dir = 'data_logs'
        self.plt_save_dir = 'plot_picts'
        self.gt_json_dict = {
            'contrast_detection_SpF_Gabor_ach': ['gt_json/castleCSF_rho_sensitivity_data.json', 'rho_list', 'sensitivity_list'],
            'contrast_detection_SpF_Noise_ach': ['gt_json/castleCSF_rho_sensitivity_data_band_lim_noise.json', 'rho_list', 'sensitivity_list'],
            'contrast_detection_SpF_Gabor_RG': ['gt_json/castleCSF_rho_sensitivity_data_RG.json', 'rho_list', 'sensitivity_list'],
            'contrast_detection_SpF_Gabor_YV': ['gt_json/castleCSF_rho_sensitivity_data_YV.json', 'rho_list', 'sensitivity_list'],
            'contrast_detection_luminance': ['gt_json/castleCSF_luminance_sensitivity_data.json', 'luminance_list', 'sensitivity_list'],
            'contrast_detection_area': ['gt_json/castleCSF_area_sensitivity_data.json', 'area_list', 'sensitivity_list'],
            'contrast_masking_phase_coherent_masking': ['gt_json/foley_contrast_masking_data_gabor.json', 'mask_contrast_list', 'test_contrast_list'],
            'contrast_masking_phase_incoherent_masking': ['gt_json/contrast_masking_data_gabor_on_noise.json', 'mask_contrast_list', 'test_contrast_list'],
            'contrast_matching_cos_scale_solve_no_scaler': 'gt_json/contrast_constancy_sin_5_cpd.json',
        }
        self.plot_json_dict = {
            'contrast_detection_SpF_Gabor_ach': ['rho_matrix', 'contrast_matrix'],
            'contrast_detection_SpF_Noise_ach': ['rho_matrix', 'contrast_matrix'],
            'contrast_detection_SpF_Gabor_RG': ['rho_matrix', 'contrast_matrix'],
            'contrast_detection_SpF_Gabor_YV': ['rho_matrix', 'contrast_matrix'],
            'contrast_detection_luminance': ['L_b_matrix', 'contrast_matrix'],
            'contrast_detection_area': ['area_matrix', 'contrast_matrix'],
            'contrast_masking_phase_coherent_masking': ['contrast_mask_matrix', 'contrast_test_matrix'],
            'contrast_masking_phase_incoherent_masking': ['contrast_mask_matrix', 'contrast_test_matrix'],
        }
        self.plot_ticks_label_dict = {
            'contrast_detection_SpF_Gabor_ach': [[0.5, 1, 2, 4, 8, 16, 32], [1, 10, 100, 1000], 'Spatial Frequency (cpd)', 'Sensitivity'],
            'contrast_detection_SpF_Noise_ach': [[0.5, 1, 2, 4, 8, 16, 32], [1, 10, 100, 1000], 'Spatial Frequency (cpd)', 'Sensitivity'],
            'contrast_detection_SpF_Gabor_RG': [[0.5, 1, 2, 4, 8, 16, 32], [5, 10, 100, 1000], 'Spatial Frequency (cpd)', 'Sensitivity'],
            'contrast_detection_SpF_Gabor_YV': [[0.5, 1, 2, 4, 8, 16, 32], [5, 10, 100, 1000], 'Spatial Frequency (cpd)', 'Sensitivity'],
            'contrast_detection_luminance': [[0.1, 1, 10, 100], [1, 10, 100, 1000], 'Luminance (cd/m$^2$)', 'Sensitivity'],
            'contrast_detection_area': [[0.1, 1], [1, 10, 100, 1000], 'Area (degree$^2$)', 'Sensitivity'],
            'contrast_masking_phase_coherent_masking': [[0.01, 0.1], [0.01, 0.1], 'Mask Contrast', 'Test Contrast'],
            'contrast_masking_phase_incoherent_masking': [[0.01, 0.1], [0.01, 0.1], 'Mask Contrast', 'Test Contrast'],
            'contrast_matching_cos_scale_solve_no_scaler': [[0.25, 0.5, 1, 2, 4, 8, 16], [0.001, 0.01, 0.1, 1], 'Test Spatial Frequency (cpd)', 'Test Contrast'],
        }
        self.plot_metrics_name_list = [['arccos_cos_similarity_matrix'],
                                  ['JOD_scale_matrix']]
        os.makedirs(self.plt_save_dir, exist_ok=True)
        self.colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'indigo', 'violet']

    def plot_all_models(self, plot_tests_list):
        for plot_test in tqdm(plot_tests_list):
            if plot_test not in self.plot_classes:
                raise ValueError('The input plot test list is not correct!')
            elif plot_test.startswith('contrast_detection') or plot_test.startswith('contrast_masking'):
                test_sub_dir = os.path.join(self.data_save_dir, f'test_{plot_test}')
                all_model_list = os.listdir(test_sub_dir)
                gt_json_list = self.gt_json_dict[plot_test]
                gt_file_path = gt_json_list[0]
                gt_x_name = gt_json_list[1]
                gt_y_name = gt_json_list[2]
                with open(gt_file_path, 'r') as fp:
                    gt_result_data = json.load(fp)
                gt_result_X_list = gt_result_data[gt_x_name]
                gt_result_Y_list = gt_result_data[gt_y_name]
                plot_ticks_label_list = self.plot_ticks_label_dict[plot_test]
                X_ticks = plot_ticks_label_list[0]
                Y_ticks = plot_ticks_label_list[1]
                X_label = plot_ticks_label_list[2]
                Y_label = plot_ticks_label_list[3]

                for model_name in tqdm(all_model_list):
                    json_file_path = os.path.join(test_sub_dir, model_name, f'{model_name}_test_{plot_test}.json')
                    with open(json_file_path, 'r') as fp:
                        plot_json_data = json.load(fp)
                    backbone_name_list = plot_json_data['json_backbone_name_list']
                    plot_json_list = self.plot_json_dict[plot_test]
                    plot_x_name = plot_json_list[0]
                    plot_y_name = plot_json_list[1]
                    for backbone_name in backbone_name_list:
                        plot_X_matrix = np.array(plot_json_data[backbone_name][plot_x_name])
                        plot_Y_matrix = np.array(plot_json_data[backbone_name][plot_y_name])
                        if plot_test.startswith('contrast_detection'):
                            plot_Y_matrix = 1 / plot_Y_matrix
                        if model_name.startswith('cvvdp'):
                            metrics_list = self.plot_metrics_name_list[1]
                        else:
                            metrics_list = self.plot_metrics_name_list[0]
                        for metric_name in metrics_list:
                            plot_score_matrix = np.array(plot_json_data[backbone_name][metric_name])
                            plot_figure_name = f'{backbone_name}-{metric_name}'
                            plt.figure(figsize=self.figure_size, dpi=self.dpi)
                            levels = np.linspace(0, 1, self.level_num)
                            plt.contourf(plot_X_matrix, plot_Y_matrix, plot_score_matrix,
                                         levels=levels, cmap='rainbow', alpha=0.3)
                            plt.contour(plot_X_matrix, plot_Y_matrix, plot_score_matrix,
                                        levels=levels, cmap='rainbow', linewidths=1)
                            if plot_test.startswith('contrast_detection'):
                                plt.plot(gt_result_X_list, gt_result_Y_list, 'k', linestyle='--', linewidth=2,
                                         label='castleCSF prediction')
                            elif plot_test.startswith('contrast_masking'):
                                plt.plot(gt_result_X_list, gt_result_Y_list, 'k', linestyle='--', linewidth=2,
                                         label='Human Results', marker='o')

                            plt.xlim([plot_X_matrix.min(), plot_X_matrix.max()])
                            plt.ylim([plot_Y_matrix.min(), plot_Y_matrix.max()])
                            plt.xlabel(X_label, fontsize=12)
                            plt.ylabel(Y_label, fontsize=12)
                            plt.xscale('log')
                            plt.yscale('log')
                            plt.xticks(X_ticks, X_ticks)
                            plt.yticks(Y_ticks, Y_ticks)
                            plt.tight_layout()
                            plt.legend(loc='lower right')
                            save_figure_dir = os.path.join(self.plt_save_dir, f'test_{plot_test}', model_name)
                            os.makedirs(save_figure_dir, exist_ok=True)
                            plt.savefig(os.path.join(save_figure_dir, plot_figure_name+'.png'), dpi=self.dpi,
                                        bbox_inches='tight', pad_inches=0.02)
                            plt.close()
            elif plot_test.startswith('contrast_matching'):
                test_sub_dir = os.path.join(self.data_save_dir, f'test_{plot_test}')
                all_model_list = os.listdir(test_sub_dir)
                gt_file_path = self.gt_json_dict[plot_test]
                with open(gt_file_path, 'r') as fp:
                    human_result_data = json.load(fp)
                plot_ticks_label_list = self.plot_ticks_label_dict[plot_test]
                rho_gt_list = [0.25, 0.5, 1, 2, 5, 10, 15, 20, 25]
                X_ticks = plot_ticks_label_list[0]
                Y_ticks = plot_ticks_label_list[1]
                X_label = plot_ticks_label_list[2]
                Y_label = plot_ticks_label_list[3]
                for model_name in tqdm(all_model_list):
                    json_file_path = os.path.join(test_sub_dir, model_name, f'{model_name}_test_{plot_test}.json')
                    try:
                        with open(json_file_path, 'r') as fp:
                            plot_json_data = json.load(fp)
                    except:
                        continue
                    backbone_name_list = plot_json_data['json_backbone_name_list']
                    reference_contrast_list = plot_json_data['reference_contrast_list']
                    rho_test_list = plot_json_data['rho_test_list']
                    for backbone_index in tqdm(range(len(backbone_name_list))):
                        backbone_name = backbone_name_list[backbone_index]
                        plt.figure(figsize=self.figure_size, dpi=self.dpi)
                        legend_OK = 0
                        for reference_contrast_index in range(len(reference_contrast_list)):
                            reference_contrast_value = reference_contrast_list[reference_contrast_index]
                            test_contrast_list = plot_json_data[backbone_name][
                                f'ref_contrast_{reference_contrast_value}_test_contrast_list']
                            human_gt_test_contrast_list =  human_result_data[f'ref_contrast_index_{reference_contrast_index}'][
                                'y_test_contrast_average']
                            if not legend_OK:
                                plt.plot(rho_test_list, test_contrast_list, color=self.colors[reference_contrast_index],
                                         linestyle='-', linewidth=2, label='Model Prediction')
                                plt.plot(rho_gt_list, human_gt_test_contrast_list,
                                         color=self.colors[reference_contrast_index], linestyle='--',
                                         linewidth=2, marker='o', label='Human Results')
                                legend_OK = 1
                            else:
                                plt.plot(rho_test_list, test_contrast_list, color=self.colors[reference_contrast_index],
                                         linestyle='-', linewidth=2)
                                plt.plot(rho_gt_list, human_gt_test_contrast_list,
                                         color=self.colors[reference_contrast_index], linestyle='--', linewidth=2,
                                         marker='o')
                        plt.legend()
                        plt.legend(loc='lower left')
                        plt.xlabel(X_label, fontsize=12)
                        plt.ylabel(Y_label, fontsize=12)
                        plt.xlim([0.25, 25])
                        plt.ylim([0.001, 1])
                        plt.xscale('log')
                        plt.yscale('log')
                        plt.xticks(X_ticks, X_ticks)
                        plt.yticks(Y_ticks, Y_ticks)
                        plt.tight_layout()
                        save_figure_dir = os.path.join(self.plt_save_dir, f'test_{plot_test}', model_name)
                        os.makedirs(save_figure_dir, exist_ok=True)
                        plot_figure_name = f'{backbone_name}-cos'
                        plt.savefig(os.path.join(save_figure_dir, plot_figure_name+'.png'), dpi=self.dpi,
                                    bbox_inches='tight', pad_inches=0.02)
                        plt.close()

            else:
                raise ValueError('The input plot test list is not correct!')











