import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from segment_anything import sam_model_registry, SamPredictor
import torch.nn.functional as F

class SAMFeatureExtractor:
    def __init__(self, model_type="vit_h", checkpoint_path="path/to/sam_vit_h_4b8939.pth", device="cuda"):
        self.device = device
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.model.to(device)
        self.model.eval()

        self.predictor = SamPredictor(self.model)

        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0)

    @torch.no_grad()
    def extract_features(self, image_path, return_tensors=True):
        image = np.array(Image.open(image_path).convert("RGB")) #0-255, H,W,3
        self.predictor.set_image(image)
        features = self.predictor.get_image_embedding()
        return features

    @torch.no_grad()
    def extract_features_from_numpy(self, image_array):
        self.predictor.set_image(image_array)
        features = self.predictor.get_image_embedding()
        return features

    @torch.no_grad()
    def extract_features_from_numpy_float32(self, image_array):
        # self.predictor.set_image(image_array)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)[None,...].cuda()
        image_tensor = F.interpolate(image_tensor, size=(1024, 1024), mode='bilinear', align_corners=False)
        features = self.predictor.model.image_encoder(image_tensor)
        return features

    def extract_features_from_tensor(self, image_tensor):
        image_tensor = F.interpolate(image_tensor, size=(1024, 1024), mode='bilinear', align_corners=False)
        features = self.predictor.model.image_encoder(image_tensor)
        return features

    def extract_features_batch(self, image_paths, batch_size=4):
        all_features = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_features = []

            for path in batch_paths:
                features = self.extract_features(path)
                batch_features.append(features)

            batch_features = torch.cat(batch_features, dim=0)
            all_features.append(batch_features)

        return torch.cat(all_features, dim=0)