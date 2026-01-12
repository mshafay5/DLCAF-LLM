#!/usr/bin/env python3
"""
FIXED Evaluation Script with Corrected Prompt Mappings
Based on your original working Strategy4_Ensemble_Multiple approach
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import CLIPModel, CLIPTokenizer

# Load your model architecture
try:
    from models import GlobalModel
except ImportError:
    print("Warning: Could not import GlobalModel")

# =============================================================================
# CORRECTED PROMPT MAPPINGS - FIXED ALL PROBLEMATIC CASES
# =============================================================================

# This contains the corrected prompts that fix all the mapping issues
STRATEGY4_PROMPTS = {

    "Brown Spots": [
        "a photo of a bacterial spot",
        "a close-up image of bacterial spot", 
        "botanical documentation of bacterial spot",
        "agricultural photo showing bacterial spot",
        "plant pathology image of bacterial spot"
    ],
    "Early Blight": [
        "a photo of a early blight disease",
        "a close-up image of early blight disease",
        "botanical documentation of early blight disease", 
        "agricultural photo showing early blight disease",
        "plant pathology image of early blight disease"
    ],
    "Healthy": [
        "a photo of a healthy tomato leaf",
        "a close-up image of healthy tomato leaf",
        "botanical documentation of healthy tomato leaf",
        "agricultural photo showing healthy tomato leaf", 
        "plant pathology image of healthy tomato leaf"
    ],
    "Mosaic Virus": [
        "a photo of a mosaic virus disease",
        "a close-up image of mosaic virus disease",
        "botanical documentation of mosaic virus disease",
        "agricultural photo showing mosaic virus disease",
        "plant pathology image of mosaic virus disease"
    ],
    "Yellow Leaf Curl": [
        "a photo of a yellow leaf curl virus",
        "a close-up image of yellow leaf curl virus",
        "botanical documentation of yellow leaf curl virus",
        "agricultural photo showing yellow leaf curl virus",
        "plant pathology image of yellow leaf curl virus"
    ],
    
    # Plant Village Dataset 
    "Bacterial Spot": [
        "a photo of a bacterial spot",
        "a close-up image of bacterial spot",
        "botanical documentation of bacterial spot",
        "agricultural photo showing bacterial spot",
        "plant pathology image of bacterial spot"
    ],
    "Early Blight": [
        "a photo of a early blight disease",
        "a close-up image of early blight disease",
        "botanical documentation of early blight disease",
        "agricultural photo showing early blight disease", 
        "plant pathology image of early blight disease"
    ],
    "Healthy": [
        "a photo of a healthy tomato leaf",
        "a close-up image of healthy tomato leaf",
        "botanical documentation of healthy tomato leaf",
        "agricultural photo showing healthy tomato leaf",
        "plant pathology image of healthy tomato leaf"
    ],
    "Late Blight": [
        "a photo of a late blight disease",
        "a close-up image of late blight disease", 
        "botanical documentation of late blight disease",
        "agricultural photo showing late blight disease",
        "plant pathology image of late blight disease"
    ],
    "Leaf Mold": [ 
        "a photo of a leaf mold disease",
        "a close-up image of leaf mold disease",
        "botanical documentation of leaf mold disease", 
        "agricultural photo showing leaf mold disease",
        "plant pathology image of leaf mold disease"
    ],
    "Mosaic Virus": [
        "a photo of a mosaic virus disease",
        "a close-up image of mosaic virus disease",
        "botanical documentation of mosaic virus disease",
        "agricultural photo showing mosaic virus disease",
        "plant pathology image of mosaic virus disease"
    ],
    "Septoria Leaf Spot": [ 
        "a photo of a septoria leaf spot disease",
        "a close-up image of septoria leaf spot disease", 
        "botanical documentation of septoria leaf spot disease",
        "agricultural photo showing septoria leaf spot disease",
        "plant pathology image of septoria leaf spot disease"
    ],
    "Spider Mites": [
        "a photo of a tomato leaf with small dark spots and stippling damage",
        "a close-up image of tomato leaf with tiny feeding damage spots",
        "botanical documentation of tomato leaf with small scattered lesions",
        "agricultural photo showing tomato leaf with fine spotted damage",
        "plant pathology image of tomato leaf with minute dark specks"
    ],

    "Target Spot": [
        "a photo of a tomato leaf with brown spots and target-like rings",
        "a close-up image of tomato leaf showing dark circular spots with concentric patterns", 
        "botanical documentation of tomato leaf with brown, bull's-eye shaped lesions",
        "agricultural photo showing tomato leaf with dark, concentric patches",
        "plant pathology image of tomato leaf infected with early blight disease"
    ],
    "Yellow Leaf Curl": [
        "a photo of a yellow leaf curl virus",
        "a close-up image of yellow leaf curl virus",
        "botanical documentation of yellow leaf curl virus",
        "agricultural photo showing yellow leaf curl virus",
        "plant pathology image of yellow leaf curl virus"
    ],
    
    # Taiwan Dataset - 
    "Bacterial Spot": [
        "a photo of a bacterial spot",
        "a close-up image of bacterial spot",
        "botanical documentation of bacterial spot", 
        "agricultural photo showing bacterial spot",
        "plant pathology image of bacterial spot"
    ],
    "Black Mold": [  
        "a photo of a black mold disease",
        "a close-up image of black mold disease",
        "botanical documentation of black mold disease",
        "agricultural photo showing black mold disease", 
        "plant pathology image of black mold disease"
    ],
    "Gray Spot": [  
        "a photo of a gray spot disease",
        "a close-up image of gray spot disease",
        "botanical documentation of gray spot disease",
        "agricultural photo showing gray spot disease",
        "plant pathology image of gray spot disease"
    ],
    "Healthy": [
        "a photo of a healthy tomato leaf",
        "a close-up image of healthy tomato leaf",
        "botanical documentation of healthy tomato leaf",
        "agricultural photo showing healthy tomato leaf",
        "plant pathology image of healthy tomato leaf"
    ],
    "Late Blight": [
        "a photo of a late blight disease",
        "a close-up image of late blight disease",
        "botanical documentation of late blight disease",
        "agricultural photo showing late blight disease",
        "plant pathology image of late blight disease"
    ],
    "Powdery Mildew": [
        "a photo of a powdery mildew",
        "a close-up image of powdery mildew",
        "botanical documentation of powdery mildew",
        "agricultural photo showing powdery mildew",
        "plant pathology image of powdery mildew"
    ],
    
}

# Configuration for your model
EVAL_CONFIG = {
    'model_path': 'experiments/ViT-B16/T11+MLP_SimCLR+MILNCE/best_model.pt',
    'model_name': 'openai/clip-vit-base-patch16', 
    'eval_dir': '/Eval',
    'output_dir': '/experiments/ViT-B16/T11+MLP_SimCLR+MILNCE/evaluation',
    'experiment_name': 'ViT-B16_T11+MLP',
    'unfreeze_last_n_blocks': 1,
}

class EvaluationDataset(Dataset):
    """Dataset class for evaluation - same as your original"""
    
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        
        self.classes = sorted([d for d in os.listdir(dataset_dir) 
                              if os.path.isdir(os.path.join(dataset_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = self._build_samples()
        print(f"Dataset: {len(self.classes)} classes, {len(self.samples)} samples")
        
    def _build_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.dataset_dir, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    samples.append({
                        'image_path': os.path.join(class_dir, img_file),
                        'class_name': class_name,
                        'class_idx': self.class_to_idx[class_name]
                    })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {sample['image_path']}: {e}")
            image = torch.zeros(3, 224, 224)
        
        return {
            'image': image,
            'class_name': sample['class_name'],
            'class_idx': sample['class_idx'],
            'image_path': sample['image_path']
        }
    

def load_trained_model(model_path, model_name, unfreeze_last_n_blocks, device):
    """Load your trained model"""
    print(f"Loading model from {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = GlobalModel(
        model_name=model_name, 
        dropout_rate=0.1,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Training loss: {checkpoint.get('loss', 'unknown'):.4f}")
    
    return model

def evaluate_dataset_fixed_prompts(model, dataset_path, device, dataset_name):
    """Evaluate using your original Strategy4 methodology but with CORRECTED prompts"""
    
    print(f"\n{'='*60}")
    print(f"EVALUATING: {dataset_name}")
    print(f"Using CORRECTED Strategy4_Ensemble_Multiple approach")
    print(f"{'='*60}")
    
    # Transform (same as your original)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # Create dataset and dataloader
    dataset = EvaluationDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    print(f"Classes found: {dataset.classes}")
    
    # STEP 1: Pre-encode all class prompts (same as your Strategy4)
    print(f"Pre-encoding corrected prompts for all classes...")
    strategy_text_features = {}
    
    for class_name in dataset.classes:
        if class_name in STRATEGY4_PROMPTS:
            prompts = STRATEGY4_PROMPTS[class_name]
            print(f"  {class_name}: Using corrected prompts")
        else:
            # Fallback for classes not in corrected mapping
            prompts = [
                f"a photo of a {class_name.lower()}",
                f"a close-up image of {class_name.lower()}",
                f"botanical documentation of {class_name.lower()}",
                f"agricultural photo showing {class_name.lower()}",
                f"plant pathology image of {class_name.lower()}"
            ]
            print(f"  {class_name}: Using fallback prompts")
        
        # Encode prompts (same as Strategy4)
        text_inputs = model.tokenizer(
            prompts, padding=True, truncation=True, 
            max_length=77, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            class_text_features = model.clip_model.get_text_features(**text_inputs)
            # Average the embeddings (ensemble approach)
            class_text_features = F.normalize(class_text_features.mean(dim=0, keepdim=True), p=2, dim=-1)
            strategy_text_features[class_name] = class_text_features
    
    # Stack all text features in class order
    text_features_tensor = torch.cat([strategy_text_features[cls] for cls in dataset.classes], dim=0)
    print(f"Encoded {text_features_tensor.shape[0]} class embeddings")
    
    # STEP 2: Evaluate images (same as your approach)
    all_predictions = []
    all_labels = []
    all_confidences = []
    sample_results = []
    
    print(f"Evaluating {len(dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            images = batch['image'].to(device)
            labels = batch['class_idx']
            
            # Get image features
            image_features = model.clip_model.get_image_features(images)
            image_features = F.normalize(image_features, p=2, dim=-1)
            
            # Compute similarities with all class embeddings
            similarities = torch.matmul(image_features, text_features_tensor.T)
            predictions = torch.argmax(similarities, dim=1)
            confidences = torch.max(similarities, dim=1)[0]
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())
            
            # Store detailed results
            for i in range(len(batch['image'])):
                sample_results.append({
                    'image_path': batch['image_path'][i],
                    'true_class': batch['class_name'][i],
                    'true_idx': labels[i].item(),
                    'pred_idx': predictions[i].item(),
                    'pred_class': dataset.classes[predictions[i].item()],
                    'confidence': confidences[i].item(),
                    'correct': predictions[i].item() == labels[i].item()
                })
    
    # STEP 3: Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    
    # Classification report
    report = classification_report(
        all_labels, all_predictions, 
        target_names=dataset.classes, 
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # STEP 4: Print results (compare with your previous results)
    print(f"\n{dataset_name} CORRECTED Results:")
    print(f"accuracy\t{accuracy:.4f}")
    print(f"f1_weighted\t{f1_weighted:.4f}")  
    print(f"f1_macro\t{f1_macro:.4f}")
    
    print(f"\nPer-class breakdown:")
    for class_name in dataset.classes:
        if class_name in report:
            class_report = report[class_name]
            print(f"{class_name}")
            print(f"precision\t{class_report['precision']:.4f}")
            print(f"recall\t{class_report['recall']:.4f}")
            print(f"f1-score\t{class_report['f1-score']:.4f}")
            print(f"support\t{int(class_report['support'])}")
    
    return {
        'dataset_name': dataset_name,
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'classification_report': report,
        'confusion_matrix': cm,
        'sample_results': sample_results,
        'corrected_prompts_used': {cls: STRATEGY4_PROMPTS.get(cls, 'fallback') 
                                  for cls in dataset.classes}
    }

def main():
    """Main evaluation with corrected prompts"""
    
    print(f"Model: {EVAL_CONFIG['experiment_name']}")
    print(f"This fixes all the prompt mapping issues identified in your results")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_trained_model(
        EVAL_CONFIG['model_path'], 
        EVAL_CONFIG['model_name'],
        EVAL_CONFIG['unfreeze_last_n_blocks'], 
        device
    )
    
    # Find evaluation datasets
    eval_datasets = []
    if os.path.exists(EVAL_CONFIG['eval_dir']):
        for item in os.listdir(EVAL_CONFIG['eval_dir']):
            dataset_path = os.path.join(EVAL_CONFIG['eval_dir'], item)
            if os.path.isdir(dataset_path):
                eval_datasets.append((item, dataset_path))
    
    print(f"Found {len(eval_datasets)} datasets: {[name for name, _ in eval_datasets]}")
    
    # Evaluate all datasets
    all_results = []
    for dataset_name, dataset_path in eval_datasets:
        try:
            result = evaluate_dataset_fixed_prompts(model, dataset_path, device, dataset_name)
            all_results.append(result)
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            continue
    
    # Create comparison table
    if all_results:
        print(f"\n{'='*80}")
        print("COMPARISON: Original vs Corrected Results")
        print(f"{'='*80}")
        
        # Your original results for comparison
        original_results = {
            "FieldPlant": {"accuracy": 77.99, "f1_weighted": 78.51},
            "Plant Village": {"accuracy": 55.75, "f1_weighted": 55.39}, 
            "Taiwan": {"accuracy": 36.33, "f1_weighted": 29.72},
            "Tomato Village": {"accuracy": 21.49, "f1_weighted": 16.46}
        }
        
        print(f"{'Dataset':<15} {'Original Acc':<12} {'Fixed Acc':<12} {'Improvement':<12} {'Original F1':<12} {'Fixed F1':<12} {'F1 Improve':<12}")
        print("-" * 90)
        
        summary_data = []
        for result in all_results:
            dataset_name = result['dataset_name']
            fixed_acc = result['accuracy'] * 100
            fixed_f1 = result['f1_weighted'] * 100
            
            if dataset_name in original_results:
                orig_acc = original_results[dataset_name]['accuracy']
                orig_f1 = original_results[dataset_name]['f1_weighted']
                acc_improvement = fixed_acc - orig_acc
                f1_improvement = fixed_f1 - orig_f1
                
                print(f"{dataset_name:<15} {orig_acc:<12.1f} {fixed_acc:<12.1f} {acc_improvement:<12.1f} "
                      f"{orig_f1:<12.1f} {fixed_f1:<12.1f} {f1_improvement:<12.1f}")
                
                summary_data.append({
                    'Dataset': dataset_name,
                    'Original_Accuracy': orig_acc,
                    'Fixed_Accuracy': fixed_acc,
                    'Accuracy_Improvement': acc_improvement,
                    'Original_F1': orig_f1,
                    'Fixed_F1': fixed_f1,
                    'F1_Improvement': f1_improvement
                })
        
        # Save results
        os.makedirs(EVAL_CONFIG['output_dir'], exist_ok=True)
        
        # Save comparison table
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_csv(os.path.join(EVAL_CONFIG['output_dir'], 'corrected_results_comparison.csv'), index=False)
        
        # Save detailed results
        for result in all_results:
            dataset_name = result['dataset_name']
            
            # Save detailed classification report
            df_report = pd.DataFrame(result['classification_report']).T
            df_report.to_csv(os.path.join(EVAL_CONFIG['output_dir'], f'{dataset_name}_classification_report.csv'))
            
            # Save confusion matrix plot  
            plt.figure(figsize=(10, 8))
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', 
                       xticklabels=list(result['classification_report'].keys())[:-3],  # Remove avg keys
                       yticklabels=list(result['classification_report'].keys())[:-3],
                       cmap='Blues')
            plt.title(f'Confusion Matrix - {dataset_name} (Corrected)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(EVAL_CONFIG['output_dir'], f'{dataset_name}_confusion_matrix.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\nResults saved to: {EVAL_CONFIG['output_dir']}")
        
if __name__ == "__main__":
    main()
