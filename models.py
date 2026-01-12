
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


class GlobalModel(nn.Module):
    """Simplified model: GDDA + LSCL with configurable ViT layers"""
    
    def __init__(self, model_name, dropout_rate=0.1, unfreeze_last_n_blocks=1):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Freeze text encoder completely
        self._setup_text_encoder_freezing()
        
        # Freeze vision encoder except final projection and last N blocks
        self._setup_vision_encoder_freezing(unfreeze_last_n_blocks=unfreeze_last_n_blocks)
    
    def _setup_text_encoder_freezing(self):
        """Fully freeze text encoder"""
        # Freeze all text encoder parameters
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.text_projection.parameters():
            param.requires_grad = False
        
        print("✅ Fully froze text encoder (no layers unfrozen)")
    
    def _setup_vision_encoder_freezing(self, unfreeze_last_n_blocks=1):
        """
        Freeze vision encoder except for the final projection layer and last N transformer blocks
        
        Args:
            unfreeze_last_n_blocks (int): Number of transformer blocks to unfreeze from the end
                                         (in addition to the projection layer)
        """
        # Freeze all vision model parameters first
        for param in self.clip_model.vision_model.parameters():
            param.requires_grad = False
        
        # Always unfreeze the final vision projection layer
        for param in self.clip_model.visual_projection.parameters():
            param.requires_grad = True
        print("✅ Unfroze final vision projection layer")
        
        # Unfreeze the last N transformer blocks
        if unfreeze_last_n_blocks > 0 and hasattr(self.clip_model.vision_model, 'encoder'):
            # For ViT models, the encoder contains the transformer blocks
            encoder_layers = self.clip_model.vision_model.encoder.layers
            total_layers = len(encoder_layers)
            
            # Unfreeze the last N blocks
            for i in range(max(0, total_layers - unfreeze_last_n_blocks), total_layers):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True
                print(f"✅ Unfroze transformer block {i} (layer {i+1} of {total_layers})")
            
            # Also unfreeze the final layer norm if it exists
            if hasattr(self.clip_model.vision_model, 'post_layernorm'):
                for param in self.clip_model.vision_model.post_layernorm.parameters():
                    param.requires_grad = True
                print("✅ Unfroze post layer normalization")
            
            print(f"🔧 Vision encoder: Frozen except final projection + last {unfreeze_last_n_blocks} transformer block(s)")
        else:
            print(f"🔧 Vision encoder: Frozen except final projection layer only")
    
    def forward(self, batch):
        full_images = batch['full_images']
        bbox_images = batch['bbox_images']  # Shape: [batch_size, 5, 3, 224, 224]
        descriptions = batch['descriptions']
        
        batch_size = full_images.size(0)
        
        # Global image features (for GDDA)
        global_image_features = self.clip_model.get_image_features(full_images)
        global_image_features = self.dropout(global_image_features)
        
        # Semantic features for LSCL - handle fixed 5 crops per image
        semantic_features_list = []
        for batch_idx in range(batch_size):
            # Get all 5 crops for this image
            image_crops = bbox_images[batch_idx]  # Shape: [5, 3, 224, 224]
            crop_features = self.clip_model.get_image_features(image_crops)
            crop_features = self.dropout(crop_features)
            semantic_features_list.append(crop_features)
        
        # Text features (fully frozen)
        text_inputs = self.tokenizer(
            descriptions, padding=True, truncation=True, 
            max_length=77, return_tensors="pt"
        ).to(full_images.device)
        text_features = self.clip_model.get_text_features(**text_inputs)
        text_features = self.dropout(text_features)
        
        return {
            'global_image_features': global_image_features,
            'semantic_features_list': semantic_features_list,  
            'text_features': text_features,
        }

class GlobalLoss(nn.Module):
    """Simplified loss: GDDA+LSCL with configurable weights"""
    
    def __init__(self, lscl_weight=0.5, temperature_mil=0.05, 
                 temperature_lscl=0.07, label_smoothing=0.1):
        super().__init__()
        self.lscl_weight = lscl_weight
        self.gdda_loss = GDDA_Loss(temperature_mil, label_smoothing)
        self.semantic_lscl_loss = SemanticSimCLRLoss(temperature_lscl)
        
        print(f"🔧 Loss configuration:")
        print(f"   LSCL weight: {lscl_weight}")
        print(f"   {'✅ LSCL ENABLED' if lscl_weight > 0 else '❌ LSCL DISABLED'}")
        print(f"   O2M Loss temperature: {temperature_mil}")
        print(f"   LSCL temperature: {temperature_lscl}")
        print(f"   Label smoothing: {label_smoothing}")
    
    def forward(self, global_image_features, semantic_features_list, 
                text_features, labels):
        
        # GDDA: Full images ↔ Text descriptions
        gdda_loss = self.gdda_loss(global_image_features, text_features, labels)
        
        # LSCL: Fixed 5 bbox crop consistency
        if self.lscl_weight > 0:
            lscl_loss = self.semantic_lscl_loss(semantic_features_list, labels)
        else:
            lscl_loss = torch.tensor(0.0, device=global_image_features.device, requires_grad=True)
        
        # Combined loss
        total_loss = gdda_loss + self.lscl_weight * lscl_loss
        
        return {
            'total_loss': total_loss,
            'gdda_loss': gdda_loss,
            'lscl_loss': lscl_loss
        }

class GDDA_Loss(nn.Module):
    """GDDA loss with label smoothing"""
    
    def __init__(self, temperature=0.05, label_smoothing=0.1):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
    
    def forward(self, image_features, text_features, labels):
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # Similarity matrix
        similarity = torch.matmul(image_features, text_features.T) / self.temperature
        
        # Positive mask (same class = positive)
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # Label smoothing
        if self.label_smoothing > 0:
            num_classes = positive_mask.size(0)
            positive_mask = (1 - self.label_smoothing) * positive_mask + \
                           self.label_smoothing / num_classes
        
        # O2M Loss calculation
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        positive_counts = positive_mask.sum(dim=1)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_counts.clamp(min=1e-8)
        
        valid_samples = (positive_counts > 0).float()
        if valid_samples.sum() > 0:
            loss = -((mean_log_prob_pos * valid_samples).sum() / valid_samples.sum())
        else:
            loss = F.cross_entropy(similarity, torch.arange(similarity.size(0), device=similarity.device))
        
        return loss

class SemanticSimCLRLoss(nn.Module):
    """LSCL loss for fixed 5 bbox crops per image"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, semantic_features_list, labels):
        """
        Args:
            semantic_features_list: List of tensors, each with shape [5, feature_dim] (5 crops per image)
            labels: Tensor with shape [batch_size]
        """
        if len(semantic_features_list) == 0:
            return torch.tensor(0.0, device=labels.device, requires_grad=True)
        
        # Collect all features and their corresponding labels
        all_features = []
        all_labels = []
        
        for batch_idx, crop_features in enumerate(semantic_features_list):
            # crop_features shape: [5, feature_dim]
            num_crops = crop_features.size(0)
            
            # Add all crops from this image
            all_features.append(crop_features)
            
            # Repeat the label for all crops from this image  
            crop_labels = labels[batch_idx].repeat(num_crops)
            all_labels.append(crop_labels)
        
        # Concatenate all features and labels
        if len(all_features) == 0:
            return torch.tensor(0.0, device=labels.device, requires_grad=True)
            
        features = torch.cat(all_features, dim=0)  # [total_crops, feature_dim]
        crop_labels = torch.cat(all_labels, dim=0)  # [total_crops]
        
        # Normalize features
        features = F.normalize(features, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive mask (same class = positive)
        positive_mask = (crop_labels.unsqueeze(0) == crop_labels.unsqueeze(1)).float()
        
        # Remove diagonal (self-similarities)
        total_crops = features.size(0)
        diagonal_mask = torch.eye(total_crops, dtype=torch.bool, device=features.device)
        positive_mask = positive_mask[~diagonal_mask].view(total_crops, -1)
        similarity_matrix = similarity_matrix[~diagonal_mask].view(total_crops, -1)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        positive_counts = positive_mask.sum(dim=1)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_counts.clamp(min=1e-8)
        
        valid_samples = (positive_counts > 0).float()
        if valid_samples.sum() > 0:
            loss = -((mean_log_prob_pos * valid_samples).sum() / valid_samples.sum())
        else:
            loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        
        return loss
