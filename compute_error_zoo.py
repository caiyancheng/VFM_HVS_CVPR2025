import numpy as np
import os
import torch
from tqdm import tqdm
import json
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr, spearmanr
class compute_error:
    def __init__(self):
        self.plot_classes = ['contrast_detection_SpF_Gabor_ach', 'contrast_detection_SpF_Noise_ach',
                             'contrast_detection_SpF_Gabor_RG', 'contrast_detection_SpF_Gabor_YV',
                             'contrast_detection_luminance', 'contrast_detection_area',
                             'contrast_masking_phase_coherent_masking', 'contrast_masking_phase_incoherent_masking',
                             'contrast_matching_cos_scale_solve_no_scaler']
        self.gt_json_path = 'gt_json/contrast_constancy_sin_5_cpd.json'
        self.data_save_dir = 'data_logs'
        self.rho_gt_list = [0.25, 0.5, 1, 2, 5, 10, 15, 20, 25]
        self.gt_rho_index = [0, 3, 7, 11, 16, 20, 22, 25, 26]
    def compute_all_models_error(self, compute_tests_list):
        for plot_test in tqdm(compute_tests_list):
            if plot_test not in self.plot_classes:
                raise ValueError('The input plot test list is not correct!')
            elif plot_test.startswith('contrast_detection') or plot_test.startswith('contrast_masking'):
                test_sub_dir = os.path.join(self.data_save_dir, f'test_{plot_test}')
                all_model_list = os.listdir(test_sub_dir)
                for model_name in tqdm(all_model_list):
                    json_file_path = os.path.join(test_sub_dir, model_name, f'{model_name}_test_{plot_test}.json')
                    with open(json_file_path, 'r') as fp:
                        plot_json_data = json.load(fp)

                    backbone_name_list = plot_json_data['json_backbone_name_list']
                    for backbone_name in backbone_name_list:
                        multiplier_list = plot_json_data[backbone_name]['multiplier_list']
                        Spearman_matrix_score = plot_json_data[backbone_name]['Spearman_matrix_score']
                        X_Spearman_multiplier = []
                        Y_Spearman_Score = []
                        for index in range(len(Spearman_matrix_score)):
                            X_Spearman_multiplier += multiplier_list
                            Y_Spearman_Score += Spearman_matrix_score[index]
                        correlation, p_value = spearmanr(X_Spearman_multiplier, Y_Spearman_Score)
                        print(plot_test, model_name, backbone_name)
                        if plot_test.startswith('contrast_masking'):
                            print("Spearman Correlation:", round(correlation, 4))
                        elif plot_test.startswith('contrast_detection'):
                            print("Spearman Correlation:", -round(correlation, 4))
            elif plot_test.startswith('contrast_matching'):
                human_result_json_path = self.gt_json_path
                with open(human_result_json_path, 'r') as fp:
                    human_result_data = json.load(fp)
                reference_contrast_list = human_result_data['average_reference_contrast']
                human_result_array = np.zeros([len(reference_contrast_list), len(self.rho_gt_list)])
                for reference_contrast_index, reference_contrast_value in enumerate(reference_contrast_list):
                    human_result_array[reference_contrast_index, :] = \
                    human_result_data[f'ref_contrast_index_{reference_contrast_index}']['y_test_contrast_average']

                test_sub_dir = os.path.join(self.data_save_dir, f'test_{plot_test}')
                all_model_list = os.listdir(test_sub_dir)
                for model_name in tqdm(all_model_list):
                    json_file_path = os.path.join(test_sub_dir, model_name, f'{model_name}_test_{plot_test}.json')
                    with open(json_file_path, 'r') as fp:
                        test_result = json.load(fp)

                    backbone_name_list = test_result['json_backbone_name_list']
                    for backbone_name in backbone_name_list:
                        model_prediction_result_array = np.zeros([len(reference_contrast_list), len(self.rho_gt_list)])
                        for reference_contrast_index, reference_contrast_value in enumerate(reference_contrast_list):
                            test_result_full_list = test_result[backbone_name][
                                f'ref_contrast_{reference_contrast_value}_test_contrast_list']
                            model_prediction_result_array[reference_contrast_index, :] = [test_result_full_list[i] for i
                                                                                          in self.gt_rho_index]
                        log_RMSE_loss = root_mean_squared_error(np.log10(human_result_array),
                                                                np.log10(model_prediction_result_array))
                        print(plot_test, model_name, backbone_name)
                        print('RMSE: ', round(log_RMSE_loss, 4))


            else:
                raise ValueError('The input plot test list is not correct!')

