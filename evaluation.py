#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
from transformers import CLIPModel, CLIPTokenizer

# Load model architecture
try:
    from SLIP_MILNCE_1layer_VT import GlobalSimCLRModel
except ImportError:
    print("Error: SLIP_MILNCE_1layer_VT.py not found.")

# =============================================================================
# RESTORED ORIGINAL PROMPT MAPPINGS (EXACTLY AS PROVIDED)
# =============================================================================
CORRECTED_STRATEGY4_PROMPTS = {

    "Brown Spots": ["a photo of a bacterial spot", "a close-up image of bacterial spot", "botanical documentation of bacterial spot", "agricultural photo showing bacterial spot", "plant pathology image of bacterial spot"],
    "Early Blight": ["a photo of a early blight disease", "a close-up image of early blight disease", "botanical documentation of early blight disease", "agricultural photo showing early blight disease", "plant pathology image of early blight disease"],
    "Healthy": ["a photo of a healthy tomato leaf", "a close-up image of healthy tomato leaf", "botanical documentation of healthy tomato leaf", "agricultural photo showing healthy tomato leaf", "plant pathology image of healthy tomato leaf"],
    "Mosaic Virus": ["a photo of a mosaic virus disease", "a close-up image of mosaic virus disease", "botanical documentation of mosaic virus disease", "agricultural photo showing mosaic virus disease", "plant pathology image of mosaic virus disease"],
    "Yellow Leaf Curl": ["a photo of a yellow leaf curl virus", "a close-up image of yellow leaf curl virus", "botanical documentation of yellow leaf curl virus", "agricultural photo showing yellow leaf curl virus", "plant pathology image of yellow leaf curl virus"],
    "Bacterial Spot": ["a photo of a bacterial spot", "a close-up image of bacterial spot", "botanical documentation of bacterial spot", "agricultural photo showing bacterial spot", "plant pathology image of bacterial spot"],
    "Late Blight": ["a photo of a late blight disease", "a close-up image of late blight disease", "botanical documentation of late blight disease", "agricultural photo showing late blight disease", "plant pathology image of late blight disease"],
    "Leaf Mold": ["a photo of a leaf mold disease", "a close-up image of leaf mold disease", "botanical documentation of leaf mold disease", "agricultural photo showing leaf mold disease", "plant pathology image of leaf mold disease"],
    "Septoria Leaf Spot": ["a photo of a septoria leaf spot disease", "a close-up image of septoria leaf spot disease", "botanical documentation of septoria leaf spot disease", "agricultural photo showing septoria leaf spot disease", "plant pathology image of septoria leaf spot disease"],
    "Spider Mites": ["a photo of a tomato leaf with small dark spots and stippling damage", "a close-up image of tomato leaf with tiny feeding damage spots", "botanical documentation of tomato leaf with small scattered lesions", "agricultural photo showing tomato leaf with fine spotted damage", "plant pathology image of tomato leaf with minute dark specks"],
    "Target Spot": ["a photo of a tomato leaf with brown spots and target-like rings", "a close-up image of tomato leaf showing dark circular spots with concentric patterns", "botanical documentation of tomato leaf with brown, bull's-eye shaped lesions", "agricultural photo showing tomato leaf with dark, concentric patches", "plant pathology image of tomato leaf infected with early blight disease"],
    "Black Mold": ["a photo of a black mold disease", "a close-up image of black mold disease", "botanical documentation of black mold disease", "agricultural photo showing black mold disease", "plant pathology image of black mold disease"],
    "Gray Spot": ["a photo of a gray spot disease", "a close-up image of gray spot disease", "botanical documentation of gray spot disease", "agricultural photo showing gray spot disease", "plant pathology image of gray spot disease"],
    "Powdery Mildew": ["a photo of a powdery mildew", "a close-up image of powdery mildew", "botanical documentation of powdery mildew", "agricultural photo showing powdery mildew", "plant pathology image of powdery mildew"],
    "Leaf Miner": ["a photo of a leaf miner damage", "a close-up image of leaf miner damage", "botanical documentation of leaf miner damage", "agricultural photo showing leaf miner damage", "plant pathology image of leaf miner damage"],
    "Nutrient Deficiency": ["a photo of a tomato leaf with yellow patches and discoloration", "a close-up image of tomato leaf showing yellowing areas", "botanical documentation of tomato leaf with yellow spots and viral symptoms", "agricultural photo showing tomato leaf with nutrient deficiency yellowing", "plant pathology image of tomato leaf with uniform color changes"],
    "Spotted Wilt Virus": ["a photo of a spotted wilt virus", "a close-up image of spotted wilt virus", "botanical documentation of spotted wilt virus", "agricultural photo showing spotted wilt virus", "plant pathology image of spotted wilt virus"],
    "Brown Spot": ["a photo of a bacterial spot", "a close-up image of bacterial spot", "botanical documentation of bacterial spot", "agricultural photo showing bacterial spot", "plant pathology image of bacterial spot"],
    "Mosaic_Viral": ["a photo of a mosaic virus disease", "a close-up image of mosaic virus disease", "botanical documentation of mosaic virus disease"],
    "Rings_Blight": ["a photo of a tomato leaf with brown spots and target-like rings", "a close-up image of tomato leaf showing dark circular spots with concentric patterns"]
}

EVAL_CONFIG = {
    'model_path': '/home/pc-4/Desktop/Shafay/8th July/PlantDoc-Object-Detection-Dataset-master/experiments/ViT-B16/T11+MLP_SimCLR+MILNCE/best_model.pt',
    'model_name': 'openai/clip-vit-base-patch16', 
    'eval_dir': '/home/pc-4/Desktop/Shafay/8th July/PlantDoc-Object-Detection-Dataset-master/Eval',
    'unfreeze_last_n_blocks': 1,
}

class EvaluationDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        for class_name in self.classes:
            class_dir = os.path.join(dataset_dir, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append({'image_path': os.path.join(class_dir, img_file), 'class_name': class_name})
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = self.transform(Image.open(sample['image_path']).convert('RGB'))
        return image, sample['class_name']

@torch.no_grad()
def run_clean_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = GlobalSimCLRModel(EVAL_CONFIG['model_name'], dropout_rate=0.1, unfreeze_last_n_blocks=EVAL_CONFIG['unfreeze_last_n_blocks'])
    checkpoint = torch.load(EVAL_CONFIG['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])
    ])

    subfolders = ["taiwan_tomato", "PV_ss", "FP", "AD"]
    
    print(f"{'Dataset':<20} | {'Accuracy':<10} | {'Weighted F1':<10}")
    print("-" * 45)

    for sub in subfolders:
        dataset_path = os.path.join(EVAL_CONFIG['eval_dir'], sub)
        if not os.path.exists(dataset_path): continue
        
        ds = EvaluationDataset(dataset_path, transform=transform)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
        
        # Pre-encode text features for this dataset's classes
        text_features_list = []
        valid_classes = []
        for cls in ds.classes:
            prompts = CORRECTED_STRATEGY4_PROMPTS.get(cls, [f"a photo of {cls}"])
            tokens = model.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
            feats = model.clip_model.get_text_features(**tokens)
            text_features_list.append(F.normalize(feats.mean(dim=0, keepdim=True), p=2, dim=-1))
            valid_classes.append(cls)
        
        text_tensor = torch.cat(text_features_list, dim=0)
        
        all_preds, all_gts = [], []
        for imgs, labels in dl:
            img_feats = F.normalize(model.clip_model.get_image_features(imgs.to(device)), p=2, dim=-1)
            logits = torch.matmul(img_feats, text_tensor.T)
            preds_idx = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend([valid_classes[i] for i in preds_idx])
            all_gts.extend(labels)

        acc = accuracy_score(all_gts, all_preds)
        f1 = f1_score(all_gts, all_preds, average='weighted', zero_division=0)
        print(f"{sub:<20} | {acc:.4f}     | {f1:.4f}")

if __name__ == "__main__":
    run_clean_eval()
