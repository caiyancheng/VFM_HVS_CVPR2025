import numpy as np
import os
from model_zoo import *
from test_zoo import *
from tqdm import tqdm
import torch
import gc
import argparse

def main(model_classes, test_classes):
    model_class_list = [globals()[model_class] for model_class in model_classes]
    test_class_list = [globals()[test_class] for test_class in test_classes]

    model_class_instance_list = []
    for model_class in model_class_list:
        print(model_class)
        model_instance = model_class()
        model_class_instance_list.append(model_instance)

    for test_class in tqdm(test_class_list):
        test_instance = test_class(sample_num=20)
        test_instance.test_models(model_class_instance_list=model_class_instance_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify model and test class lists.")
    parser.add_argument("--model_classes", type=str, nargs="+", help="List of model classes to use.")
    parser.add_argument("--test_classes", type=str, nargs="+", help="List of test classes to use.")
    args = parser.parse_args()

    model_classes = args.model_classes if args.model_classes else [
        "no_encoder_tools", "dino_tools", "dinov2_tools", "mae_tools", "openclip_tools", "sam_float_tools", "vae_tools", "cvvdp_tools"
    ] # We don't provide the code for SAM-2 as it is still updating, please don't use the cvvdp_tools if you cannot download it.
    test_classes = args.test_classes if args.test_classes else [
        "Contrast_Detection_Area", "Contrast_Detection_Luminance", "Contrast_Detection_SpF_Gabor_Ach",
        "Contrast_Detection_SpF_Noise_Ach", "Contrast_Detection_SpF_Gabor_RG", "Contrast_Detection_SpF_Gabor_YV",
        "Contrast_Masking_Phase_Coherent", "Contrast_Masking_Phase_Incoherent", "Contrast_Matching_cos_scale_solve_no_scaler"
    ]

    main(model_classes, test_classes)
