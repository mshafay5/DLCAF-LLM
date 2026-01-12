#!/usr/bin/env python3
"""
Complete Experiment Runner for All 8 Vision-Language Experiments

This script can run all 8 experiments by simply changing the CONFIG section:
- ViT-B16/B32 models
- MLP vs T11+MLP architectures  
- LSCL+MILNCE vs GDDA-only losses

Just change the CONFIG dictionary and run!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import random
from tqdm import tqdm
import json
from transformers import CLIPModel, CLIPTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
from datetime import datetime

try:
    from models import GlobalModel, GlobalLoss, GDDA_Loss, SemanticSimCLRLoss
except ImportError:
    print("Warning: Could not import GlobalModel")

# CLIP-style disease descriptions
# =============================================================================
# 7 Descriptions

# disease_visual_descriptions = {
#     "Tomato_Early_blight_leaf": [
#         "a photo of a tomato leaf with brown spots and target-like rings",
#         "a photo of a leaf showing dark circular spots with concentric patterns",
#         "a photo of a diseased tomato leaf with brown, bull's-eye shaped lesions",
#         "a photo of a tomato leaf with dark, concentric patches and ring-like spots",
#         "a photo of a tomato leaf infected with early blight disease",
#         "a photo of a tomato leaf with expanding brown lesions showing distinct concentric zones",
#         "a photo of a leaf displaying necrotic spots with characteristic target-shaped patterns and surrounding yellowing"
#     ],
#     "Tomato_leaf": [
#         "a photo of a healthy green tomato leaf",
#         "a photo of a normal tomato leaf without any disease symptoms",
#         "a photo of a fresh, vibrant green tomato leaf",
#         "a photo of an undamaged tomato leaf in good condition",
#         "a photo of a clean, healthy leaf from a tomato plant",
#         "a photo of a tomato leaf with uniform green color and smooth surface texture",
#         "a photo of an intact tomato leaf showing no discoloration or lesions"
#     ],
#     "Tomato_leaf_bacterial_spot": [
#         "a photo of a tomato leaf with small dark bacterial spots",
#         "a photo of a diseased tomato leaf with tiny black lesions",
#         "a photo of a leaf showing scattered small dark spots from bacterial infection",
#         "a photo of a tomato leaf with numerous small, round bacterial spots",
#         "a photo of a leaf with dark specks caused by bacterial spot disease",
#         "a photo of a tomato leaf covered with pinpoint-sized dark lesions with yellow halos",
#         "a photo of a leaf displaying multiple small angular spots with water-soaked margins"
#     ],
#     "Tomato_leaf_late_blight": [
#         "a photo of a tomato leaf with large, water-soaked dark spots",
#         "a photo of a diseased tomato leaf with brown, blotchy patches",
#         "a photo of a leaf with dark, spreading areas from late blight",
#         "a photo of a tomato leaf showing signs of late blight disease",
#         "a photo of a wilted tomato leaf with brown-black spots",
#         "a photo of a tomato leaf with irregular dark lesions and white fungal growth on undersides",
#         "a photo of a leaf showing rapid tissue collapse with greasy-appearing dark brown areas"
#     ],
#     "Tomato_leaf_mosaic_virus": [
#         "a photo of a tomato leaf with light and dark green mosaic patterns",
#         "a photo of a diseased tomato leaf showing mottled color variations",
#         "a photo of a leaf with uneven green coloring and mosaic-like patterns",
#         "a photo of a tomato leaf infected with mosaic virus showing patchy colors",
#         "a photo of a tomato leaf with viral mosaic disease patterns",
#         "a photo of a tomato leaf displaying alternating light and dark green irregular patches",
#         "a photo of a leaf with distorted surface and characteristic mosaic mottling pattern"
#     ],
#     "Tomato_leaf_yellow_virus": [
#         "a photo of a tomato leaf with yellow patches and discoloration",
#         "a photo of a diseased tomato leaf showing yellowing areas",
#         "a photo of a leaf with yellow spots and viral infection symptoms",
#         "a photo of a tomato leaf infected with yellow leaf curl virus",
#         "a photo of a tomato leaf turning yellow due to viral disease",
#         "a photo of a tomato leaf with progressive yellowing from margins inward",
#         "a photo of a leaf showing bright yellow discoloration and upward curling edges"
#     ],
#     "Tomato_mold_leaf": [
#         "a photo of a tomato leaf with gray fuzzy mold growth",
#         "a photo of a diseased tomato leaf covered in mold patches",
#         "a photo of a leaf with soft, moldy areas and fungal growth",
#         "a photo of a tomato leaf showing mold infection with fuzzy texture",
#         "a photo of a tomato leaf with white or gray mold covering",
#         "a photo of a tomato leaf with velvety gray fungal colonies and spore masses",
#         "a photo of a leaf displaying powdery white to gray mold patches with visible mycelia"
#     ],
#     "Tomato_Septoria_leaf_spot": [
#         "a photo of a tomato leaf with small, round spots and dark borders",
#         "a photo of a diseased tomato leaf showing Septoria leaf spot lesions",
#         "a photo of a leaf with circular spots having gray centers and dark edges",
#         "a photo of a tomato leaf with small, dark-rimmed lesions",
#         "a photo of a leaf displaying characteristic Septoria leaf spot disease",
#         "a photo of a tomato leaf with numerous small circular lesions featuring tan centers and dark brown margins",
#         "a photo of a leaf showing pepper-like spots with visible black fruiting bodies in gray centers"
#     ]
# }


# =============================================================================
# 10 Descriptions
# disease_visual_descriptions = {
#     "Tomato_Early_blight_leaf": [
#         "a photo of a tomato leaf with brown spots and target-like rings",
#         "a photo of a leaf showing dark circular spots with concentric patterns",
#         "a photo of a diseased tomato leaf with brown, bull's-eye shaped lesions",
#         "a photo of a tomato leaf with dark, concentric patches and ring-like spots",
#         "a photo of a tomato leaf infected with early blight disease",
#         "a photo of a tomato leaf with expanding brown lesions showing distinct concentric zones",
#         "a photo of a leaf displaying necrotic spots with characteristic target-shaped patterns and surrounding yellowing",
#         "a photo of a tomato leaf with multiple target spot lesions and progressive tissue death",
#         "a photo of a tomato leaf with alternating dark and light brown concentric rings",
#         "a photo of early blight symptoms featuring circular lesions with defined margins"
#     ],
#     "Tomato_leaf": [
#         "a photo of a healthy green tomato leaf",
#         "a photo of a normal tomato leaf without any disease symptoms",
#         "a photo of a fresh, vibrant green tomato leaf",
#         "a photo of an undamaged tomato leaf in good condition",
#         "a photo of a clean, healthy leaf from a tomato plant",
#         "a photo of a tomato leaf with uniform green color and smooth surface texture",
#         "a photo of an intact tomato leaf showing no discoloration or lesions",
#         "a photo of a tomato leaf with normal green pigmentation and structure",
#         "a photo of a pristine tomato leaf with no signs of disease or damage",
#         "a photo of a healthy tomato leaf with consistent color throughout"
#     ],
#     "Tomato_leaf_bacterial_spot": [
#         "a photo of a tomato leaf with small dark bacterial spots",
#         "a photo of a diseased tomato leaf with tiny black lesions",
#         "a photo of a leaf showing scattered small dark spots from bacterial infection",
#         "a photo of a tomato leaf with numerous small, round bacterial spots",
#         "a photo of a leaf with dark specks caused by bacterial spot disease",
#         "a photo of a tomato leaf covered with pinpoint-sized dark lesions with yellow halos",
#         "a photo of a leaf displaying multiple small angular spots with water-soaked margins",
#         "a photo of bacterial leaf spot showing raised dark lesions on leaf surface",
#         "a photo of a tomato leaf with densely packed small necrotic spots",
#         "a photo of bacterial infection with characteristic greasy-looking small spots"
#     ],
#     "Tomato_leaf_late_blight": [
#         "a photo of a tomato leaf with large, water-soaked dark spots",
#         "a photo of a diseased tomato leaf with brown, blotchy patches",
#         "a photo of a leaf with dark, spreading areas from late blight",
#         "a photo of a tomato leaf showing signs of late blight disease",
#         "a photo of a wilted tomato leaf with brown-black spots",
#         "a photo of a tomato leaf with irregular dark lesions and white fungal growth on undersides",
#         "a photo of a leaf showing rapid tissue collapse with greasy-appearing dark brown areas",
#         "a photo of late blight featuring large necrotic zones with undefined borders",
#         "a photo of a tomato leaf with advancing brown lesions and wilting tissue",
#         "a photo of severe late blight infection with extensive leaf destruction"
#     ],
#     "Tomato_leaf_mosaic_virus": [
#         "a photo of a tomato leaf with light and dark green mosaic patterns",
#         "a photo of a diseased tomato leaf showing mottled color variations",
#         "a photo of a leaf with uneven green coloring and mosaic-like patterns",
#         "a photo of a tomato leaf infected with mosaic virus showing patchy colors",
#         "a photo of a tomato leaf with viral mosaic disease patterns",
#         "a photo of a tomato leaf displaying alternating light and dark green irregular patches",
#         "a photo of a leaf with distorted surface and characteristic mosaic mottling pattern",
#         "a photo of mosaic virus symptoms showing variegated green patterns",
#         "a photo of a tomato leaf with interveinal mosaic discoloration",
#         "a photo of viral infection featuring distinct light and dark green zones"
#     ],
#     "Tomato_leaf_yellow_virus": [
#         "a photo of a tomato leaf with yellow patches and discoloration",
#         "a photo of a diseased tomato leaf showing yellowing areas",
#         "a photo of a leaf with yellow spots and viral infection symptoms",
#         "a photo of a tomato leaf infected with yellow leaf curl virus",
#         "a photo of a tomato leaf turning yellow due to viral disease",
#         "a photo of a tomato leaf with progressive yellowing from margins inward",
#         "a photo of a leaf showing bright yellow discoloration and upward curling edges",
#         "a photo of yellow virus infection with chlorotic leaf tissue",
#         "a photo of a tomato leaf with extensive yellowing and reduced green pigmentation",
#         "a photo of viral yellowing with characteristic leaf curl symptoms"
#     ],
#     "Tomato_mold_leaf": [
#         "a photo of a tomato leaf with gray fuzzy mold growth",
#         "a photo of a diseased tomato leaf covered in mold patches",
#         "a photo of a leaf with soft, moldy areas and fungal growth",
#         "a photo of a tomato leaf showing mold infection with fuzzy texture",
#         "a photo of a tomato leaf with white or gray mold covering",
#         "a photo of a tomato leaf with velvety gray fungal colonies and spore masses",
#         "a photo of a leaf displaying powdery white to gray mold patches with visible mycelia",
#         "a photo of fungal mold infection with extensive hyphal networks",
#         "a photo of a tomato leaf with dense mold growth and tissue decay",
#         "a photo of advanced mold colonization with cottony fungal structures"
#     ],
#     "Tomato_Septoria_leaf_spot": [
#         "a photo of a tomato leaf with small, round spots and dark borders",
#         "a photo of a diseased tomato leaf showing Septoria leaf spot lesions",
#         "a photo of a leaf with circular spots having gray centers and dark edges",
#         "a photo of a tomato leaf with small, dark-rimmed lesions",
#         "a photo of a leaf displaying characteristic Septoria leaf spot disease",
#         "a photo of a tomato leaf with numerous small circular lesions featuring tan centers and dark brown margins",
#         "a photo of a leaf showing pepper-like spots with visible black fruiting bodies in gray centers",
#         "a photo of Septoria infection with distinctive gray-centered lesions",
#         "a photo of a tomato leaf with multiple small spots containing pycnidia",
#         "a photo of Septoria leaf spot featuring circular lesions with prominent dark borders"
#     ]
# }
# 



# =============================================================================
## 5 Descriptions
# =============================================================================
disease_visual_descriptions = {
    "Tomato_Early_blight_leaf": [
        "a photo of a tomato leaf with brown spots and target-like rings",
        "a photo of a leaf showing dark circular spots with concentric patterns",
        "a photo of a diseased tomato leaf with brown, bull's-eye shaped lesions",
       "a photo of a tomato leaf with dark, concentric patches and ring-like spots",
        "a photo of a tomato leaf infected with early blight disease"
    ],
    "Tomato_leaf": [
        "a photo of a healthy green tomato leaf",
        "a photo of a normal tomato leaf without any disease symptoms",
        "a photo of a fresh, vibrant green tomato leaf",
        "a photo of an undamaged tomato leaf in good condition",
        "a photo of a clean, healthy leaf from a tomato plant"
    ],
    "Tomato_leaf_bacterial_spot": [
        "a photo of a tomato leaf with small dark bacterial spots",
        "a photo of a diseased tomato leaf with tiny black lesions",
       "a photo of a leaf showing scattered small dark spots from bacterial infection",
       "a photo of a tomato leaf with numerous small, round bacterial spots",
       "a photo of a leaf with dark specks caused by bacterial spot disease"
    ],
    "Tomato_leaf_late_blight": [
        "a photo of a tomato leaf with large, water-soaked dark spots",
        "a photo of a diseased tomato leaf with brown, blotchy patches",
        "a photo of a leaf with dark, spreading areas from late blight",
        "a photo of a tomato leaf showing signs of late blight disease"
        "a photo of a wilted tomato leaf with brown-black spots"
    ],
    "Tomato_leaf_mosaic_virus": [
       "a photo of a tomato leaf with light and dark green mosaic patterns",
        "a photo of a diseased tomato leaf showing mottled color variations",
        "a photo of a leaf with uneven green coloring and mosaic-like patterns",
       "a photo of a tomato leaf infected with mosaic virus showing patchy colors",
        "a photo of a tomato leaf with viral mosaic disease patterns"
    ],
    "Tomato_leaf_yellow_virus": [
        "a photo of a tomato leaf with yellow patches and discoloration",
        "a photo of a diseased tomato leaf showing yellowing areas",
        "a photo of a leaf with yellow spots and viral infection symptoms",
        "a photo of a tomato leaf infected with yellow leaf curl virus"
        "a photo of a tomato leaf turning yellow due to viral disease"
    ],
    "Tomato_mold_leaf": [
        "a photo of a tomato leaf with gray fuzzy mold growth",
        "a photo of a diseased tomato leaf covered in mold patches",
        "a photo of a leaf with soft, moldy areas and fungal growth",
        "a photo of a tomato leaf showing mold infection with fuzzy texture",
        "a photo of a tomato leaf with white or gray mold covering"
    ],
    "Tomato_Septoria_leaf_spot": [
        "a photo of a tomato leaf with small, round spots and dark borders",
        "a photo of a diseased tomato leaf showing Septoria leaf spot lesions",
        "a photo of a leaf with circular spots having gray centers and dark edges",
        "a photo of a tomato leaf with small, dark-rimmed lesions",
        "a photo of a leaf displaying characteristic Septoria leaf spot disease"
    ]
}
# 
# =============================================================================

# =============================================================================

# =============================================================================
# EXPERIMENT CONFIGURATIONS - CHANGE THIS SECTION TO RUN DIFFERENT EXPERIMENTS
# =============================================================================

# EXPERIMENT 1: ViT-B16 + MLP + LSCL+MILNCE
CONFIG_VB16_MLP_BOTH = {
    'full_images_dir': 'dataset/Full_Images',
    'bbox_images_dir': 'dataset/Cropped_Images',
    'output_dir': 'experiments/ViT-B16/MLP_SimCLR+MILNCE',
    'model_name': 'openai/clip-vit-base-patch16',
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 5e-5,
    'lscl_weight': 0.5,  # Enable LSCL
    'temperature_mil': 0.05,
    'temperature_lscl': 0.07,
    'num_workers': 4,
    'fixed_augmentations': 5,
    'label_smoothing': 0.1,
    'dropout_rate': 0.1,
    'early_stopping_patience': 10,
    'min_delta': 0.001,
    'save_every_n_epochs': 5,
    'unfreeze_last_n_blocks': 0,  # Only MLP
}

# EXPERIMENT 2: ViT-B16 + MLP + GDDA-only
CONFIG_VB16_MLP_MILNCE = {
    'full_images_dir': 'dataset/Full_Images',
    'bbox_images_dir': 'dataset/Cropped_Images',
    'output_dir': 'experiments/ViT-B16/MLP_MILNCE-only',
    'model_name': 'openai/clip-vit-base-patch16',
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 5e-5,
    'lscl_weight': 0.0,  # Disable LSCL
    'temperature_mil': 0.05,
    'temperature_lscl': 0.07,
    'num_workers': 4,
    'fixed_augmentations': 5,
    'label_smoothing': 0.1,
    'dropout_rate': 0.1,
    'early_stopping_patience': 10,
    'min_delta': 0.001,
    'save_every_n_epochs': 5,
    'unfreeze_last_n_blocks': 0,  # Only MLP
}

# EXPERIMENT 3: ViT-B16 + T11+MLP + LSCL+MILNCE
CONFIG_VB16_T11_BOTH = {
    'full_images_dir': 'dataset/Full_Images',
    'bbox_images_dir': 'dataset/Cropped_Images',
    'output_dir': 'experiments/ViT-B16/T11+MLP_SimCLR+MILNCE',
    'model_name': 'openai/clip-vit-base-patch16',
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 5e-5,
    'lscl_weight': 0.5,  # Enable LSCL
    'temperature_mil': 0.05,
    'temperature_lscl': 0.07,
    'num_workers': 4,
    'fixed_augmentations': 5,
    'label_smoothing': 0.1,
    'dropout_rate': 0.1,
    'early_stopping_patience': 15,
    'min_delta': 0.001,
    'save_every_n_epochs': 5,
    'unfreeze_last_n_blocks': 1,  # T11 + MLP
}

# EXPERIMENT 4: ViT-B16 + T11+MLP + GDDA-only
CONFIG_VB16_T11_MILNCE = {
    'full_images_dir': 'dataset/Full_Images',
    'bbox_images_dir': 'dataset/Cropped_Images',
    'output_dir': 'experiments/ViT-B16/T11+MLP_MILNCE-only',
    'model_name': 'openai/clip-vit-base-patch16',
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 5e-5,
    'lscl_weight': 0.0,  # Disable LSCL
    'temperature_mil': 0.05,
    'temperature_lscl': 0.07,
    'num_workers': 4,
    'fixed_augmentations': 5,
    'label_smoothing': 0.1,
    'dropout_rate': 0.1,
    'early_stopping_patience': 10,
    'min_delta': 0.001,
    'save_every_n_epochs': 5,
    'unfreeze_last_n_blocks': 1,  # T11 + MLP
}

# EXPERIMENT 5: ViT-B32 + MLP + LSCL+MILNCE
CONFIG_VB32_MLP_BOTH = {
    'full_images_dir': 'dataset/Full_Images',
    'bbox_images_dir': 'dataset/Cropped_Images',
    'output_dir': 'experiments/ViT-B32/MLP_SimCLR+MILNCE',
    'model_name': 'openai/clip-vit-base-patch32',
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 5e-5,
    'lscl_weight': 0.5,  # Enable LSCL
    'temperature_mil': 0.05,
    'temperature_lscl': 0.07,
    'num_workers': 4,
    'fixed_augmentations': 5,
    'label_smoothing': 0.1,
    'dropout_rate': 0.1,
    'early_stopping_patience': 10,
    'min_delta': 0.001,
    'save_every_n_epochs': 5,
    'unfreeze_last_n_blocks': 0,  # Only MLP
}

# EXPERIMENT 6: ViT-B32 + MLP + GDDA-only
CONFIG_VB32_MLP_MILNCE = {
    'full_images_dir': 'dataset/Full_Images',
    'bbox_images_dir': 'dataset/Cropped_Images',
    'output_dir': 'experiments/ViT-B32/MLP_MILNCE-only',
    'model_name': 'openai/clip-vit-base-patch32',
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 5e-5,
    'lscl_weight': 0.0,  # Disable LSCL
    'temperature_mil': 0.05,
    'temperature_lscl': 0.07,
    'num_workers': 4,
    'fixed_augmentations': 5,
    'label_smoothing': 0.1,
    'dropout_rate': 0.1,
    'early_stopping_patience': 10,
    'min_delta': 0.001,
    'save_every_n_epochs': 5,
    'unfreeze_last_n_blocks': 0,  # Only MLP
}

# EXPERIMENT 7: ViT-B32 + T11+MLP + LSCL+MILNCE
CONFIG_VB32_T11_BOTH = {
    'full_images_dir': 'dataset/Full_Images',
    'bbox_images_dir': 'dataset/Cropped_Images',
    'output_dir': 'experiments/ViT-B32/T11+MLP_SimCLR+MILNCE',
    'model_name': 'openai/clip-vit-base-patch32',
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 5e-5,
    'lscl_weight': 0.5,  # Enable LSCL
    'temperature_mil': 0.05,
    'temperature_lscl': 0.07,
    'num_workers': 4,
    'fixed_augmentations': 5,
    'label_smoothing': 0.1,
    'dropout_rate': 0.1,
    'early_stopping_patience': 10,
    'min_delta': 0.001,
    'save_every_n_epochs': 5,
    'unfreeze_last_n_blocks': 1,  # T11 + MLP
}

# EXPERIMENT 8: ViT-B32 + T11+MLP + GDDA-only
CONFIG_VB32_T11_MILNCE = {
    'full_images_dir': 'dataset/Full_Images',
    'bbox_images_dir': 'dataset/Cropped_Images',
    'output_dir': 'experiments/ViT-B32/T11+MLP_MILNCE-only',
    'model_name': 'openai/clip-vit-base-patch32',
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 5e-5,
    'lscl_weight': 0.0,  # Disable LSCL
    'temperature_mil': 0.05,
    'temperature_lscl': 0.07,
    'num_workers': 4,
    'fixed_augmentations': 5,
    'label_smoothing': 0.1,
    'dropout_rate': 0.1,
    'early_stopping_patience': 10,
    'min_delta': 0.001,
    'save_every_n_epochs': 5,
    'unfreeze_last_n_blocks': 1,  # T11 + MLP
}

# =============================================================================
# CHOOSE YOUR EXPERIMENT HERE - CHANGE THIS LINE TO RUN DIFFERENT EXPERIMENTS
# =============================================================================

# CHANGE THIS LINE TO SWITCH BETWEEN EXPERIMENTS:
CONFIG = CONFIG_VB16_T11_BOTH  # <-- CHANGE THIS TO RUN DIFFERENT EXPERIMENTS

# Available options:
# CONFIG = CONFIG_VB16_MLP_BOTH     # Experiment 1 (completed), 12.32 it/s
# CONFIG = CONFIG_VB16_MLP_MILNCE   # Experiment 2 (completed) , 12.5 it/sec
# CONFIG = CONFIG_VB16_T11_BOTH     # Experiment 3 (completed)
# CONFIG = CONFIG_VB16_T11_MILNCE   # Experiment 4 (completed), 12.08 it/sec
# CONFIG = CONFIG_VB32_MLP_BOTH     # Experiment 5 (completed)
# CONFIG = CONFIG_VB32_MLP_MILNCE   # Experiment 6 (completed), 18.43 it/sec
# CONFIG = CONFIG_VB32_T11_BOTH     # Experiment 7 (running)
# CONFIG = CONFIG_VB32_T11_MILNCE   # Experiment 8 (completed), 18.82 it/s

# =============================================================================
# CORE IMPLEMENTATION (NO CHANGES NEEDED BELOW)
# =============================================================================

class SimpleBoundingBoxProcessor:
    """Fixed augmentation processor - always return exactly 5 augmentations"""
    
    def __init__(self, target_augmentations=5):
        self.target_augmentations = target_augmentations
        
    def process_bbox_list(self, bbox_images, full_image=None):
        """Process bbox list to get exactly target_augmentations"""
        if len(bbox_images) >= self.target_augmentations:
            # If we have enough bbox crops, randomly select target_augmentations
            selected_indices = random.sample(range(len(bbox_images)), self.target_augmentations)
            return [bbox_images[i] for i in selected_indices]
        elif len(bbox_images) > 0:
            # If we have some bbox crops but not enough, use them + LSCL augmentations
            needed_augmentations = self.target_augmentations - len(bbox_images)
            LSCL_augmentations = self._create_simclr_augmentations(full_image, needed_augmentations)
            return bbox_images + LSCL_augmentations
        else:
            # No bbox crops, create all 5 as LSCL augmentations
            return self._create_simclr_augmentations(full_image, self.target_augmentations)
    
    def _create_simclr_augmentations(self, full_image, num_augmentations):
        """Create LSCL-style augmentations from full image"""
        augmentations = []
        
        for _ in range(num_augmentations):
            # LSCL-style random crop and augmentation
            augmented_image = self._apply_simclr_augmentation(full_image)
            augmentations.append(augmented_image)
        
        return augmentations
    
    def _apply_simclr_augmentation(self, image):
        """Apply single LSCL-style augmentation"""
        w, h = image.size
        
        # Random crop (LSCL uses random crops between 0.08 and 1.0 of the area)
        crop_area_ratio = random.uniform(0.08, 1.0)
        crop_aspect_ratio = random.uniform(0.75, 1.33)  # 3/4 to 4/3
        
        crop_area = crop_area_ratio * w * h
        crop_w = int(round((crop_area * crop_aspect_ratio) ** 0.5))
        crop_h = int(round((crop_area / crop_aspect_ratio) ** 0.5))
        
        # Ensure crop dimensions don't exceed image dimensions
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        
        # Random position for crop
        left = random.randint(0, max(0, w - crop_w))
        top = random.randint(0, max(0, h - crop_h))
        
        # Crop the image
        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        
        return cropped

class LeafDiseaseDataset(Dataset):
    """Dataset with dynamic bbox count based on available crops"""
    
    def __init__(self, full_images_dir, bbox_images_dir, disease_descriptions, 
                 transform=None, train=True, fixed_augmentations=5):
        self.full_images_dir = full_images_dir
        self.bbox_images_dir = bbox_images_dir
        self.disease_descriptions = disease_descriptions
        self.transform = transform
        self.train = train
        self.fixed_augmentations = fixed_augmentations
        
        # Fixed augmentation processor
        self.bbox_processor = SimpleBoundingBoxProcessor(target_augmentations=fixed_augmentations)
        
        # Discover classes
        self.classes = sorted([d for d in os.listdir(full_images_dir) 
                              if os.path.isdir(os.path.join(full_images_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"Discovered {len(self.classes)} classes: {self.classes}")
        
        # Build dataset
        self.samples = self._build_samples()
        print(f"Built dataset with {len(self.samples)} unique images")
        
        # Calculate and print detailed statistics
        self._calculate_and_print_statistics()
    
    def _build_samples(self):
        """Build samples list"""
        samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.full_images_dir, class_name)
            
            for img_file in os.listdir(class_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                full_img_path = os.path.join(class_dir, img_file)
                img_name = os.path.splitext(img_file)[0]
                bbox_imgs = self._find_bbox_images(class_name, img_name)
                
                sample = {
                    'full_image_path': full_img_path,
                    'bbox_image_paths': bbox_imgs,
                    'class_name': class_name,
                    'class_idx': self.class_to_idx[class_name],
                    'descriptions': self.disease_descriptions.get(class_name, [f"a photo of a {class_name}"])
                }
                samples.append(sample)
        
        return samples
    
    def _find_bbox_images(self, class_name, img_name):
        """Find bounding box images"""
        bbox_class_dir = os.path.join(self.bbox_images_dir, class_name)
        if not os.path.exists(bbox_class_dir):
            return []
        
        bbox_imgs = []
        img_name_lower = img_name.lower()
        
        for bbox_file in os.listdir(bbox_class_dir):
            bbox_file_lower = bbox_file.lower()
            if (bbox_file_lower.startswith(img_name_lower) and 
                '_bb' in bbox_file_lower and 
                bbox_file_lower.endswith(('.jpg', '.jpeg', '.png'))):
                bbox_imgs.append(os.path.join(bbox_class_dir, bbox_file))
        
        return sorted(bbox_imgs)
    
    def _calculate_and_print_statistics(self):
        """Calculate and print detailed dataset statistics with fixed augmentations"""
        print(f"\n{'='*60}")
        print("DETAILED DATASET STATISTICS (FIXED 5 AUGMENTATIONS)")
        print(f"{'='*60}")
        
        # Initialize counters
        total_unique_images = len(self.samples)
        total_full_image_samples = 0
        total_augmentation_samples = 0
        total_bbox_files_found = 0
        total_descriptions = 0
        images_with_enough_bbox = 0
        images_with_some_bbox = 0
        images_without_bbox = 0
        
        class_stats = {}
        bbox_count_distribution = {}
        
        # Calculate per-class statistics
        for sample in self.samples:
            class_name = sample['class_name']
            num_descriptions = len(sample['descriptions'])
            num_bbox_files = len(sample['bbox_image_paths'])
            
            # Track bbox count distribution
            if num_bbox_files not in bbox_count_distribution:
                bbox_count_distribution[num_bbox_files] = 0
            bbox_count_distribution[num_bbox_files] += 1
            
            # Categorize images by bbox availability
            if num_bbox_files >= self.fixed_augmentations:
                images_with_enough_bbox += 1
            elif num_bbox_files > 0:
                images_with_some_bbox += 1
            else:
                images_without_bbox += 1
            
            if class_name not in class_stats:
                class_stats[class_name] = {
                    'unique_images': 0,
                    'descriptions_per_class': num_descriptions,
                    'total_bbox_files': 0,
                    'images_with_enough_bbox': 0,
                    'images_with_some_bbox': 0,
                    'images_without_bbox': 0,
                    'full_image_samples': 0,
                    'augmentation_samples': 0
                }
            
            # Count unique images per class
            class_stats[class_name]['unique_images'] += 1
            
            # Count bbox files found
            class_stats[class_name]['total_bbox_files'] += num_bbox_files
            total_bbox_files_found += num_bbox_files
            
            # Categorize by bbox availability
            if num_bbox_files >= self.fixed_augmentations:
                class_stats[class_name]['images_with_enough_bbox'] += 1
            elif num_bbox_files > 0:
                class_stats[class_name]['images_with_some_bbox'] += 1
            else:
                class_stats[class_name]['images_without_bbox'] += 1
            
            if self.train:
                # Training mode: multiple descriptions per image
                full_image_samples_for_this_image = num_descriptions
                augmentation_samples_for_this_image = self.fixed_augmentations * num_descriptions
                
                class_stats[class_name]['full_image_samples'] += full_image_samples_for_this_image
                class_stats[class_name]['augmentation_samples'] += augmentation_samples_for_this_image
                
                total_full_image_samples += full_image_samples_for_this_image
                total_augmentation_samples += augmentation_samples_for_this_image
                total_descriptions += num_descriptions
            else:
                # Test mode: one sample per image
                class_stats[class_name]['full_image_samples'] += 1
                class_stats[class_name]['augmentation_samples'] += self.fixed_augmentations
                
                total_full_image_samples += 1
                total_augmentation_samples += self.fixed_augmentations
                total_descriptions += 1
        
        # Print overall statistics
        print(f"Mode: {'TRAINING' if self.train else 'TESTING'}")
        print(f"Unique images: {total_unique_images:,}")
        print(f"Images with ≥{self.fixed_augmentations} bbox crops: {images_with_enough_bbox:,}")
        print(f"Images with some bbox crops: {images_with_some_bbox:,}")
        print(f"Images without bbox crops: {images_without_bbox:,}")
        print(f"Fixed augmentations per image: {self.fixed_augmentations}")
        print(f"Total bbox crop files found: {total_bbox_files_found:,}")
        print(f"Total descriptions used: {total_descriptions:,}")
        
        # Print bbox count distribution
        print(f"\n📊 BBOX COUNT DISTRIBUTION:")
        for bbox_count in sorted(bbox_count_distribution.keys()):
            count = bbox_count_distribution[bbox_count]
            strategy = "use all bbox" if bbox_count >= self.fixed_augmentations else \
                      f"bbox + {self.fixed_augmentations - bbox_count} LSCL" if bbox_count > 0 else \
                      f"all {self.fixed_augmentations} LSCL"
            print(f"  {bbox_count} bbox crops: {count:,} images ({strategy})")
        
        print(f"\n📊 TRAINING SAMPLES BREAKDOWN:")
        print(f"  Full image samples (GDDA): {total_full_image_samples:,}")
        print(f"  Augmentation samples (LSCL): {total_augmentation_samples:,}")
        
        total_training_samples = total_full_image_samples + total_augmentation_samples
        print(f"  📈 TOTAL TRAINING SAMPLES: {total_training_samples:,}")
        
        if total_unique_images > 0:
            multiplication_factor = total_training_samples / total_unique_images
            print(f"  Multiplication factor: {multiplication_factor:.1f}x")
        
        # Print per-class breakdown
        print(f"\n📋 PER-CLASS BREAKDOWN:")
        print(f"{'Class':<25} {'Images':<8} {'≥5 BB':<8} {'Some BB':<8} {'No BB':<8} {'BB Files':<8} {'Full Samp':<10} {'Aug Samp':<10} {'Total':<8}")
        print("-" * 110)
        
        for class_name in sorted(class_stats.keys()):
            stats = class_stats[class_name]
            total_class_samples = stats['full_image_samples'] + stats['augmentation_samples']
            
            print(f"{class_name:<25} {stats['unique_images']:<8} "
                  f"{stats['images_with_enough_bbox']:<8} {stats['images_with_some_bbox']:<8} "
                  f"{stats['images_without_bbox']:<8} {stats['total_bbox_files']:<8} "
                  f"{stats['full_image_samples']:<10} {stats['augmentation_samples']:<10} "
                  f"{total_class_samples:<8}")
        
        # Set total_items for __len__
        if self.train:
            self.total_items = total_full_image_samples
        else:
            self.total_items = len(self.samples)
        
        print(f"\n🔄 Dataset __len__() returns: {self.total_items:,}")
        print(f"{'='*60}\n")
    
    def __len__(self):
        """Return the total number of items in the dataset"""
        return self.total_items
    
    def __getitem__(self, idx):
        if self.train:
            # Calculate which sample and description to use
            sample_idx = 0
            desc_idx = 0
            current_idx = idx
            
            for i, sample in enumerate(self.samples):
                num_descriptions = len(sample['descriptions'])
                if current_idx < num_descriptions:
                    sample_idx = i
                    desc_idx = current_idx
                    break
                current_idx -= num_descriptions
            
            sample = self.samples[sample_idx]
            description = sample['descriptions'][desc_idx]
        else:
            sample = self.samples[idx]
            description = sample['descriptions'][0]
        
        # Load full image
        full_image = Image.open(sample['full_image_path']).convert('RGB')
        
        # Load bounding box images
        bbox_images_pil = []
        for bbox_path in sample['bbox_image_paths']:
            try:
                bbox_img = Image.open(bbox_path).convert('RGB')
                bbox_images_pil.append(bbox_img)
            except Exception as e:
                print(f"Warning: Error loading {bbox_path}: {e}")
        
        # Process bbox images - use actual crops or create fallback crops
        bbox_images_processed = self.bbox_processor.process_bbox_list(
            bbox_images_pil, full_image=full_image
        )
        
        # Apply transforms
        if self.transform:
            full_image_transformed = self.transform(full_image)
            bbox_images_transformed = [self.transform(img) for img in bbox_images_processed]
        else:
            full_image_transformed = full_image
            bbox_images_transformed = bbox_images_processed
        
        return {
            'full_image': full_image_transformed,
            'bbox_images': torch.stack(bbox_images_transformed),
            'description': description,
            'class_idx': sample['class_idx'],
            'class_name': sample['class_name'],
            'image_path': sample['full_image_path'],
            'num_bbox_crops': len(bbox_images_processed)  # Track actual number of crops
        }
    
def collate_fn(batch):
    """Collate function with fixed 5 augmentations per image"""
    full_images = torch.stack([item['full_image'] for item in batch])
    
    # All images now have exactly 5 augmentations
    batch_size = len(batch)
    fixed_augmentations = 5
    
    bbox_images = torch.zeros(batch_size, fixed_augmentations, 3, 224, 224)
    
    for i, item in enumerate(batch):
        bbox_images[i] = item['bbox_images']  # Should be exactly [5, 3, 224, 224]
    
    class_indices = torch.tensor([item['class_idx'] for item in batch])
    
    return {
        'full_images': full_images,
        'bbox_images': bbox_images,
        'descriptions': [item['description'] for item in batch],
        'class_indices': class_indices,
        'class_names': [item['class_name'] for item in batch],
        'image_paths': [item['image_path'] for item in batch]
    }


def train_model(model, train_loader, loss_fn, optimizer, scheduler, device, num_epochs, save_dir, 
                early_stopping_patience=10, min_delta=0.001, save_every_n_epochs=5):
    """Training loop with early stopping and comprehensive logging"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Early stopping variables
    best_loss = float('inf')
    epochs_without_improvement = 0
    early_stopped = False
    
    # Training history
    history = {
        'epoch': [], 
        'total_loss': [], 
        'gdda_loss': [], 
        'lscl_loss': [], 
        'lr': [],
        'early_stopped': False,
        'best_epoch': 0
    }
    
    # Training start time
    start_time = time.time()
    
    print(f"\n🚀 TRAINING STARTED")
    print(f"   Max epochs: {num_epochs}")
    print(f"   Early stopping patience: {early_stopping_patience}")
    print(f"   Min improvement delta: {min_delta}")
    print(f"   Checkpoint every: {save_every_n_epochs} epochs")
    print(f"   Save directory: {save_dir}")
    print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for epoch in range(num_epochs):
        model.train()
        running_losses = {'total': 0, 'global_mil': 0, 'simclr': 0}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch)
            
            # Compute loss
            loss_dict = loss_fn(
                outputs['global_image_features'],
                outputs['semantic_features_list'],
                outputs['text_features'],
                batch['class_indices']
            )
            
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update losses
            running_losses['total'] += total_loss.item()
            running_losses['global_mil'] += loss_dict['gdda_loss'].item()
            running_losses['simclr'] += loss_dict['lscl_loss'].item()
            num_batches += 1
            
            # Update progress bar with early stopping info
            pbar.set_postfix({
                'Total': f"{total_loss.item():.4f}",
                'Global': f"{loss_dict['gdda_loss'].item():.4f}",
                'SimCLR': f"{loss_dict['lscl_loss'].item():.4f}",
                'Best': f"{best_loss:.4f}",
                'Patience': f"{epochs_without_improvement}/{early_stopping_patience}"
            })
        
        # Update scheduler and calculate average losses
        scheduler.step()
        avg_losses = {k: v / num_batches for k, v in running_losses.items()}
        current_lr = scheduler.get_last_lr()[0]
        
        # Update history
        history['epoch'].append(epoch + 1)
        history['total_loss'].append(avg_losses['total'])
        history['gdda_loss'].append(avg_losses['global_mil'])
        history['lscl_loss'].append(avg_losses['simclr'])
        history['lr'].append(current_lr)
        
        # Early stopping logic
        improvement = best_loss - avg_losses['total']
        
        if improvement > min_delta:
            # Significant improvement found
            best_loss = avg_losses['total']
            epochs_without_improvement = 0
            history['best_epoch'] = epoch + 1
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'history': history,
                'config': CONFIG,
                'early_stopping_info': {
                    'patience': early_stopping_patience,
                    'min_delta': min_delta,
                    'epochs_without_improvement': epochs_without_improvement
                }
            }, os.path.join(save_dir, 'best_model.pt'))
            
            print(f"✅ Epoch {epoch+1}: NEW BEST! Total: {avg_losses['total']:.4f} "
                  f"(↓{improvement:.4f}), Global: {avg_losses['global_mil']:.4f}, "
                  f"SimCLR: {avg_losses['simclr']:.4f}, LR: {current_lr:.6f}")
        else:
            # No significant improvement
            epochs_without_improvement += 1
            print(f"⏳ Epoch {epoch+1}: Total: {avg_losses['total']:.4f}, "
                  f"Global: {avg_losses['global_mil']:.4f}, "
                  f"SimCLR: {avg_losses['simclr']:.4f}, LR: {current_lr:.6f} "
                  f"[No improvement: {epochs_without_improvement}/{early_stopping_patience}]")
        
        # Save periodic checkpoints
        if (epoch + 1) % save_every_n_epochs == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_losses['total'],
                'history': history,
                'config': CONFIG,
                'early_stopping_info': {
                    'patience': early_stopping_patience,
                    'min_delta': min_delta,
                    'epochs_without_improvement': epochs_without_improvement
                }
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
            print(f"💾 Checkpoint saved at epoch {epoch+1}")
        
        # Check for early stopping
        if epochs_without_improvement >= early_stopping_patience:
            early_stopped = True
            history['early_stopped'] = True
            print(f"\n🛑 EARLY STOPPING triggered!")
            print(f"   No improvement for {early_stopping_patience} consecutive epochs")
            print(f"   Best loss: {best_loss:.4f} at epoch {history['best_epoch']}")
            print(f"   Stopped at epoch: {epoch + 1}")
            break
    
    # Calculate training time
    end_time = time.time()
    training_time = end_time - start_time
    
    # Final summary
    print(f"\n🏁 TRAINING COMPLETED")
    if not early_stopped:
        print(f"   Completed all {num_epochs} epochs")
    else:
        print(f"   Early stopped after {epoch + 1} epochs")
    print(f"   Best loss: {best_loss:.4f} at epoch {history['best_epoch']}")
    print(f"   Total training time: {training_time/3600:.2f} hours")
    print(f"   End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save final training state
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_losses['total'],
        'history': history,
        'config': CONFIG,
        'training_completed': not early_stopped,
        'training_time_hours': training_time/3600,
        'early_stopping_info': {
            'patience': early_stopping_patience,
            'min_delta': min_delta,
            'epochs_without_improvement': epochs_without_improvement,
            'early_stopped': early_stopped
        }
    }, os.path.join(save_dir, 'final_model.pt'))
    
    return model, history

def print_experiment_info(config):
    """Print detailed experiment information"""
    print(f"\n{'='*80}")
    print("EXPERIMENT CONFIGURATION")
    print(f"{'='*80}")
    
    # Determine experiment type
    model_type = "ViT-B/16" if "patch16" in config['model_name'] else "ViT-B/32"
    architecture = "MLP only" if config['unfreeze_last_n_blocks'] == 0 else f"T{12-config['unfreeze_last_n_blocks']}+MLP"
    loss_type = "SimCLR+MILNCE" if config['lscl_weight'] > 0 else "MILNCE only"
    
    print(f"📋 Experiment Details:")
    print(f"   Model: {model_type}")
    print(f"   Architecture: {architecture}")
    print(f"   Loss Functions: {loss_type}")
    print(f"   Output Directory: {config['output_dir']}")
    
    print(f"\n🔧 Training Configuration:")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Max Epochs: {config['num_epochs']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   LSCL Weight: {config['lscl_weight']}")
    print(f"   Early Stopping Patience: {config['early_stopping_patience']}")
    print(f"   Dropout Rate: {config['dropout_rate']}")
    
    print(f"\n🎯 Model Architecture:")
    print(f"   Base Model: {config['model_name']}")
    print(f"   Unfrozen Blocks: {config['unfreeze_last_n_blocks']}")
    print(f"   Text Encoder: Fully Frozen")
    vision_arch = 'MLP only' if CONFIG['unfreeze_last_n_blocks'] == 0 else f'Last {CONFIG["unfreeze_last_n_blocks"]} block(s) + MLP'
    print(f"   Vision Encoder: {vision_arch}")
    
def main():
    """Main function - runs the selected experiment"""
    
    # Print experiment information
    print_experiment_info(CONFIG)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print(f"📁 Output directory: {CONFIG['output_dir']}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # Create dataset
    print(f"\n📦 Creating dataset...")
    full_dataset = LeafDiseaseDataset(
        full_images_dir=CONFIG['full_images_dir'],
        bbox_images_dir=CONFIG['bbox_images_dir'],
        disease_descriptions=disease_visual_descriptions,
        transform=transform,
        train=False,
        fixed_augmentations=CONFIG['fixed_augmentations']
    )
    
    # Use all data for training (no test split)
    train_samples = full_dataset.samples
    train_dataset = LeafDiseaseDataset(
        full_images_dir=CONFIG['full_images_dir'],
        bbox_images_dir=CONFIG['bbox_images_dir'],
        disease_descriptions=disease_visual_descriptions,
        transform=transform,
        train=True,
        fixed_augmentations=CONFIG['fixed_augmentations']
    )
    train_dataset.samples = train_samples
    train_dataset.total_items = sum(len(disease_visual_descriptions.get(sample['class_name'], [''])) 
                                   for sample in train_samples)
    
    # Data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"✅ Dataset created successfully")
    print(f"   Training samples: {len(train_samples)} unique images")
    print(f"   Training items: {train_dataset.total_items:,}")
    print(f"   Classes: {len(train_dataset.classes)}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Batches per epoch: {len(train_loader)}")
    
    # Load model
    print(f"\n🤖 Loading model: {CONFIG['model_name']}")
    model = GlobalModel(
        CONFIG['model_name'], 
        dropout_rate=CONFIG['dropout_rate'],
        unfreeze_last_n_blocks=CONFIG['unfreeze_last_n_blocks']
    ).to(device)
    
    # Loss function
    loss_fn = GlobalLoss(
        lscl_weight=CONFIG['lscl_weight'],
        temperature_mil=CONFIG['temperature_mil'],
        temperature_lscl=CONFIG['temperature_lscl'],
        label_smoothing=CONFIG['label_smoothing']
    )
    
    # Count parameters
    print(f"\n📊 Parameter Analysis:")
    trainable_params = []
    trainable_param_count = 0
    frozen_param_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            trainable_param_count += param.numel()
            print(f"   ✅ Trainable: {name} - {param.numel():,} parameters")
        else:
            frozen_param_count += param.numel()
    
    total_params = trainable_param_count + frozen_param_count
    print(f"\n📈 Parameter Summary:")
    print(f"   Trainable parameters: {trainable_param_count:,}")
    print(f"   Frozen parameters: {frozen_param_count:,}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable percentage: {100 * trainable_param_count / total_params:.2f}%")
    
    # Optimizer for only trainable parameters
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=CONFIG['learning_rate'], 
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
    
    print(f"\n⚙️  Optimizer: AdamW")
    print(f"   Learning rate: {CONFIG['learning_rate']}")
    print(f"   Weight decay: 0.01")
    print(f"   Scheduler: CosineAnnealingLR")
    
    # Training
    print(f"\n🎯 Training Strategy:")
    print(f"   GDDA: Full images ↔ Disease descriptions")
    if CONFIG['lscl_weight'] > 0:
        print(f"   LSCL: 5 crops per image consistency learning (weight: {CONFIG['lscl_weight']})")
    else:
        print(f"   LSCL: DISABLED (weight: 0.0)")
    print(f"   Dynamic bbox strategy: Use actual crops or fallback to LSCL augmentations")
    print(f"   Text encoder: Fully frozen")
    vision_desc = 'MLP Only' if CONFIG['unfreeze_last_n_blocks'] == 0 else f'Last {CONFIG["unfreeze_last_n_blocks"]} Block(s) + MLP'
    print(f"   Vision Encoder: {vision_desc}")
    
    # Start training
    # Start training
    model, history = train_model(
    model=model,
    train_loader=train_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    num_epochs=CONFIG['num_epochs'],
    save_dir=CONFIG['output_dir'],
    early_stopping_patience=CONFIG['early_stopping_patience'],
    min_delta=CONFIG['min_delta'],
    save_every_n_epochs=CONFIG['save_every_n_epochs']
)
    # Print training summary
    print(f"\n📊 TRAINING SUMMARY:")
    if history['early_stopped']:
        print(f"   Status: Early stopped after {len(history['epoch'])} epochs")
    else:
        print(f"   Status: Completed all {len(history['epoch'])} epochs")
    print(f"   Best epoch: {history['best_epoch']}")
    print(f"   Best loss: {min(history['total_loss']):.4f}")
    print(f"   Final loss: {history['total_loss'][-1]:.4f}")
    
    # Save training history
    print(f"\n💾 Saving training artifacts...")
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(CONFIG['output_dir'], 'training_history.csv'), index=False)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['epoch'], history['total_loss'], 'b-', linewidth=2)
    plt.title('Total Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    if history['best_epoch'] in history['epoch']:
        best_idx = history['epoch'].index(history['best_epoch'])
        plt.axvline(x=history['best_epoch'], color='red', linestyle='--', alpha=0.7, label=f'Best: {history["best_epoch"]}')
        plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['epoch'], history['gdda_loss'], 'g-', linewidth=2, label='GDDA')
    if CONFIG['lscl_weight'] > 0:
        plt.plot(history['epoch'], history['lscl_loss'], 'r-', linewidth=2, label='LSCL')
    plt.title('Individual Losses', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['epoch'], history['lr'], 'purple', linewidth=2)
    plt.title('Learning Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save experiment info
    experiment_info = {
        'experiment_details': {
            'model_type': "ViT-B/16" if "patch16" in CONFIG['model_name'] else "ViT-B/32",
            'architecture': "MLP only" if CONFIG['unfreeze_last_n_blocks'] == 0 else f"T{12-CONFIG['unfreeze_last_n_blocks']}+MLP",
            'loss_functions': "SimCLR+MILNCE" if CONFIG['lscl_weight'] > 0 else "MILNCE only",
            'output_directory': CONFIG['output_dir']
        },
        'dataset_info': {
            'classes': train_dataset.classes,
            'class_to_idx': train_dataset.class_to_idx,
            'num_classes': len(train_dataset.classes),
            'total_unique_images': len(train_samples),
            'total_training_items': train_dataset.total_items,
            'descriptions': disease_visual_descriptions
        },
        'model_info': {
            'base_model': CONFIG['model_name'],
            'architecture': 'GDDA + LSCL',
            'text_encoder': 'Fully frozen',
            'vision_encoder': f"Frozen except final projection + last {CONFIG['unfreeze_last_n_blocks']} block(s)" if CONFIG['unfreeze_last_n_blocks'] > 0 else "Frozen except final projection",
            'trainable_parameters': trainable_param_count,
            'frozen_parameters': frozen_param_count,
            'total_parameters': total_params,
            'trainable_percentage': 100 * trainable_param_count / total_params
        },
        'training_config': CONFIG,
        'training_results': {
            'completed_epochs': len(history['epoch']),
            'early_stopped': history['early_stopped'],
            'best_epoch': history['best_epoch'],
            'best_loss': min(history['total_loss']),
            'final_loss': history['total_loss'][-1],
            'training_time_info': 'See final_model.pt for training time'
        },
        'files_saved': {
            'best_model': 'best_model.pt',
            'final_model': 'final_model.pt',
            'training_history_csv': 'training_history.csv',
            'training_history_plot': 'training_history.png',
            'experiment_info': 'experiment_info.json',
            'tokenizer': 'tokenizer/'
        }
    }
    
    with open(os.path.join(CONFIG['output_dir'], 'experiment_info.json'), 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    # Save tokenizer
    tokenizer_path = os.path.join(CONFIG['output_dir'], 'tokenizer')
    model.tokenizer.save_pretrained(tokenizer_path)
    
    # Create README for the experiment
    readme_content = f"""# Experiment Results

## Experiment Configuration
- **Model**: {experiment_info['experiment_details']['model_type']}
- **Architecture**: {experiment_info['experiment_details']['architecture']}
- **Loss Functions**: {experiment_info['experiment_details']['loss_functions']}
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset
- **Classes**: {experiment_info['dataset_info']['num_classes']}
- **Unique Images**: {experiment_info['dataset_info']['total_unique_images']:,}
- **Training Items**: {experiment_info['dataset_info']['total_training_items']:,}

## Model Parameters
- **Trainable**: {experiment_info['model_info']['trainable_parameters']:,} ({experiment_info['model_info']['trainable_percentage']:.2f}%)
- **Frozen**: {experiment_info['model_info']['frozen_parameters']:,}
- **Total**: {experiment_info['model_info']['total_parameters']:,}

## Training Results
- **Epochs Completed**: {experiment_info['training_results']['completed_epochs']}
- **Early Stopped**: {'Yes' if experiment_info['training_results']['early_stopped'] else 'No'}
- **Best Epoch**: {experiment_info['training_results']['best_epoch']}
- **Best Loss**: {experiment_info['training_results']['best_loss']:.4f}
- **Final Loss**: {experiment_info['training_results']['final_loss']:.4f}

## Files Saved
- `best_model.pt` - Best performing model checkpoint
- `final_model.pt` - Final training state
- `training_history.csv` - Training metrics per epoch
- `training_history.png` - Training curves visualization
- `experiment_info.json` - Complete experiment metadata
- `tokenizer/` - Saved tokenizer files

## How to Load the Model
```python
import torch
from transformers import CLIPTokenizer

# Load the best model
checkpoint = torch.load('best_model.pt')
model_state_dict = checkpoint['model_state_dict']
config = checkpoint['config']

# Load tokenizer
tokenizer = CLIPTokenizer.from_pretrained('tokenizer/')

# Use the model for inference...
```
"""
    
    with open(os.path.join(CONFIG['output_dir'], 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Training history saved: training_history.csv")
    print(f"✅ Training plots saved: training_history.png")  
    print(f"✅ Experiment info saved: experiment_info.json")
    print(f"✅ Tokenizer saved: tokenizer/")
    print(f"✅ README created: README.md")
    
    print(f"\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"📁 All results saved to: {CONFIG['output_dir']}")
    print(f"🏆 Best model: {CONFIG['output_dir']}/best_model.pt")
    
    # Show next steps
    print(f"\n🔄 NEXT STEPS:")
    print(f"1. To run another experiment, change the CONFIG line and run again")
    print(f"2. Available configs: CONFIG_VB16_MLP_BOTH, CONFIG_VB16_MLP_MILNCE, etc.")
    print(f"3. To evaluate this model, use the best_model.pt file")
    print(f"4. Check the README.md file for detailed results and usage instructions")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        print(f"📁 Partial results may be saved in: {CONFIG['output_dir']}")
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print(f"📁 Check output directory: {CONFIG['output_dir']}")
        raise

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================

"""
HOW TO USE THIS SCRIPT:

1. **Choose your experiment** by changing this line at the top:
   CONFIG = CONFIG_VB16_MLP_BOTH  # Change this line
   
   Available options:
   - CONFIG_VB16_MLP_BOTH     # ViT-B16 + MLP + LSCL+MILNCE  
   - CONFIG_VB16_MLP_MILNCE   # ViT-B16 + MLP + GDDA only
   - CONFIG_VB16_T11_BOTH     # ViT-B16 + T11+MLP + LSCL+MILNCE
   - CONFIG_VB16_T11_MILNCE   # ViT-B16 + T11+MLP + GDDA only
   - CONFIG_VB32_MLP_BOTH     # ViT-B32 + MLP + LSCL+MILNCE
   - CONFIG_VB32_MLP_MILNCE   # ViT-B32 + MLP + GDDA only  
   - CONFIG_VB32_T11_BOTH     # ViT-B32 + T11+MLP + LSCL+MILNCE
   - CONFIG_VB32_T11_MILNCE   # ViT-B32 + T11+MLP + GDDA only

2. **Run the script**:
   python experiment_runner.py

3. **Monitor progress**:
   - Training progress with early stopping info
   - Best model automatically saved
   - Checkpoints saved every 5 epochs
   - Training curves plotted and saved

4. **Check results**:
   - All outputs saved to organized folder structure
   - README.md created with experiment summary
   - JSON file with complete metadata
   - Training history CSV for analysis

5. **Run next experiment**:
   - Change CONFIG line
   - Run script again
   - Results saved to different folder automatically

EXAMPLE WORKFLOW:
1. CONFIG = CONFIG_VB16_MLP_BOTH -> Run -> Check results
2. CONFIG = CONFIG_VB16_MLP_MILNCE -> Run -> Check results  
3. CONFIG = CONFIG_VB16_T11_BOTH -> Run -> Check results
... and so on for all 8 experiments

The script handles everything automatically:
✅ Model loading and configuration
✅ Architecture setup (MLP vs T11+MLP)
✅ Loss function setup (SimCLR+MILNCE vs GDDA only)
✅ Training with early stopping
✅ Automatic checkpointing
✅ Results organization and saving
✅ Comprehensive logging and documentation
"""
