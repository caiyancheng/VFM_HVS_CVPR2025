# RMSE for contrast matching and Spearman Correlation for contrast detection and contrast masking
from compute_error_zoo import *

# The Nine Tests
# ['contrast_detection_SpF_Gabor_ach', 'contrast_detection_SpF_Noise_ach',
#  'contrast_detection_SpF_Gabor_RG', 'contrast_detection_SpF_Gabor_YV',
#  'contrast_detection_luminance', 'contrast_detection_area',
#  'contrast_masking_phase_coherent_masking', 'contrast_masking_phase_incoherent_masking',
#  'contrast_matching_cos_scale_solve']

compute_tests_list = ['contrast_detection_SpF_Gabor_ach', 'contrast_detection_SpF_Noise_ach',
                      'contrast_detection_SpF_Gabor_RG', 'contrast_detection_SpF_Gabor_YV',
                      'contrast_detection_luminance', 'contrast_detection_area',
                      'contrast_masking_phase_coherent_masking', 'contrast_masking_phase_incoherent_masking',
                      'contrast_matching_cos_scale_solve_no_scaler']
compute_class = compute_error()
compute_class.compute_all_models_error(compute_tests_list)
X = 1

