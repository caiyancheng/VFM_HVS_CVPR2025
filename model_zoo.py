import numpy as np
import torch
torch.hub.set_dir(r'./Torch_hub')
from display_encoding import display_encode
display_encode_tool = display_encode(400)
from transformers import AutoImageProcessor, ViTMAEForPreTraining
import open_clip
from SAM_repo.SAM import SAMFeatureExtractor
from diffusers import AutoencoderKL
import pycvvdp
import torch.nn.functional as F

def check_input(input, aim_shape=(224,224,3)):
    if input.shape == aim_shape:
        return 1
    else:
        raise ValueError('Input does not have the shape of (224, 224, 3)')

class no_encoder_tools:
    def __init__(self):
        self.name = 'no_encoder'
        self.backbone_list = [self.name]

    def load_pretrained(self, backbone_name):
        X = 1

    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = display_encode_tool.L2C_sRGB(image_L_array)
        image_C_tensor = torch.tensor(image_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
        feature = image_C_tensor
        return feature

class cvvdp_tools:
    def __init__(self):
        self.name = 'cvvdp_hdr'
        disp_photo = pycvvdp.vvdp_display_photo_eotf(400, contrast=1000000, source_colorspace='BT.709', EOTF="linear", E_ambient=0)
        self.metric = pycvvdp.cvvdp(display_name='standard_hdr_linear', display_photometry=disp_photo)
        self.backbone_list = [self.name]

    def load_pretrained(self, backbone_name):
        X = 1

    def compute_score(self, T_L_array, R_L_array): #直接输入Luminance
        check_input(T_L_array)
        check_input(R_L_array)
        T_L_tensor = torch.tensor(T_L_array, dtype=torch.float32)
        R_L_tensor = torch.tensor(R_L_array, dtype=torch.float32)
        JOD, m_stats = self.metric.predict(T_L_tensor, R_L_tensor, dim_order="HWC")
        return JOD

class dino_tools:
    def __init__(self):
        self.name = 'dino'
        self.backbone_list = ['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8', 'dino_xcit_small_12_p16',
                     'dino_xcit_small_12_p8', 'dino_xcit_medium_24_p16', 'dino_xcit_medium_24_p8', 'dino_resnet50']
    def load_pretrained(self, backbone_name):
        backbone_model = torch.hub.load('facebookresearch/dino:main', backbone_name)
        backbone_model.eval()
        backbone_model.cuda()
        return backbone_model

    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = display_encode_tool.L2C_sRGB(image_L_array)
        image_C_tensor = torch.tensor(image_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
        feature = backbone_model(image_C_tensor)
        return feature

class dinov2_tools:
    def __init__(self):
        self.name = 'dinov2'
        self.backbone_list = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
                         'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg']
    def load_pretrained(self, backbone_name):
        backbone_model = torch.hub.load('facebookresearch/dinov2', backbone_name)
        backbone_model.eval()
        backbone_model.cuda()
        return backbone_model

    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = display_encode_tool.L2C_sRGB(image_L_array)
        image_C_tensor = torch.tensor(image_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
        feature = backbone_model(image_C_tensor)
        return feature

class mae_tools:
    def __init__(self):
        self.name = 'mae'
        self.backbone_list = ['vit-mae-base', 'vit-mae-large', 'vit-mae-huge']
        torch.manual_seed(8)
        self.default_noise = torch.rand(1, 196)
    def load_pretrained(self, backbone_name):
        processor = AutoImageProcessor.from_pretrained(f'facebook/{backbone_name}')
        model = ViTMAEForPreTraining.from_pretrained(f'facebook/{backbone_name}')
        model.eval()
        return (processor, model)
    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = display_encode_tool.L2C_sRGB(image_L_array)
        processor, model = backbone_model
        input_dict = processor(images=image_C_array.astype(np.float32), return_tensors="pt", do_resize=False, do_rescale=False,
                               do_normalize=False)
        input_dict['noise'] = self.default_noise
        feature = model.vit(**input_dict).last_hidden_state
        return feature

class openclip_tools:
    def __init__(self):
        self.name = 'openclip'
        clip_model_list = [('RN50', 'openai'), ('RN50', 'yfcc15m'), ('RN101', 'openai'), ('RN101', 'yfcc15m'),
                       ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion2b_s34b_b79k'), ('ViT-B-16', 'openai'),
                       ('ViT-B-16', 'laion2b_s34b_b88k'),
                       ('ViT-L-14', 'openai'), ('ViT-L-14', 'laion2b_s32b_b82k'),
                       ('convnext_base_w', 'laion2b_s13b_b82k'), ('convnext_base_w', 'laion2b_s13b_b82k_augreg'),
                       ('convnext_large_d', 'laion2b_s26b_b102k_augreg'),
                       ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg')]
        self.backbone_list = clip_model_list

    def load_pretrained(self, backbone_name):
        clip_model_name, clip_model_trainset = backbone_name
        model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name,
                                                                     pretrained=clip_model_trainset,
                                                                     cache_dir=r'./Openclip_cache')
        model.eval()
        model.cuda()
        return (model, preprocess)

    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = display_encode_tool.L2C_sRGB(image_L_array)
        image_C_tensor = torch.tensor(image_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
        model, preprocess = backbone_model
        feature = model.encode_image(image_C_tensor)
        return feature

class sam_tools:
    def __init__(self):
        self.name = 'sam'
        self.backbone_list = ['sam_vit_b_01ec64', 'sam_vit_l_0b3195', 'sam_vit_h_4b8939']
        self.sam_vit_list = ['vit_b', 'vit_l', 'vit_h']
    def load_pretrained(self, backbone_name):
        model_type = backbone_name.split('_')[1]+'_'+backbone_name.split('_')[2]
        backbone_model = SAMFeatureExtractor(
            model_type=model_type,
            checkpoint_path=rf"./SAM_repo/{backbone_name}.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return backbone_model.cuda()

    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = (display_encode_tool.L2C_sRGB(image_L_array) * 255).astype(np.uint8).cuda()
        feature = backbone_model.extract_features_from_numpy(image_C_array)
        return feature

class sam_float_tools:
    def __init__(self):
        self.name = 'sam_float'
        self.backbone_list = ['sam_vit_b_01ec64', 'sam_vit_l_0b3195', 'sam_vit_h_4b8939']
        self.sam_vit_list = ['vit_b', 'vit_l', 'vit_h']
    def load_pretrained(self, backbone_name):
        model_type = backbone_name.split('_')[1]+'_'+backbone_name.split('_')[2]
        backbone_model = SAMFeatureExtractor(
            model_type=model_type,
            checkpoint_path=rf"./SAM_repo/{backbone_name}.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return backbone_model

    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = display_encode_tool.L2C_sRGB(image_L_array).astype(np.float32)
        feature = backbone_model.extract_features_from_numpy_float32(image_C_array)
        return feature

class vae_tools:
    def __init__(self):
        self.name = 'vae'
        self.backbone_list = ['stable-diffusion-v1-5/stable-diffusion-v1-5', 'stabilityai/stable-diffusion-xl-base-1.0']
    def load_pretrained(self, backbone_name):
        backbone_model = AutoencoderKL.from_pretrained(backbone_name, subfolder="vae").cuda()
        return backbone_model
    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = display_encode_tool.L2C_sRGB(image_L_array)
        image_C_tensor = torch.tensor(image_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
        norm_image_C_tensor = (image_C_tensor - 0.5) * 2
        latent = backbone_model.encode(norm_image_C_tensor)
        feature = latent.latent_dist.sample()
        return feature