"""
STEGO (Self-supervised semantic segmentation by distilling feature correspondences) implementation
Based on the research paper: "Unsupervised Semantic Segmentation by Distilling Feature Correspondences"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from pathlib import Path

try:
    from dinov2.models import build_model_from_cfg
    from dinov2.utils.config import setup_and_build_model
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False
    print("Warning: DINOv2 not available. Please install it for STEGO functionality.")


class STEGOHead(nn.Module):
    """
    STEGO segmentation head that processes DINO features
    """
    def __init__(self, feature_dim: int = 768, num_classes: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # MLP for feature processing
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: DINO features of shape (B, H*W, feature_dim)
        Returns:
            logits: Segmentation logits of shape (B, H*W, num_classes)
        """
        return self.mlp(features)


class STEGOModel(nn.Module):
    """
    Complete STEGO model with DINO backbone and segmentation head
    """
    def __init__(self, 
                 dino_model_name: str = "dinov2_vitb14",
                 num_classes: int = 4,
                 feature_dim: int = 768,
                 hidden_dim: int = 256,
                 device: str = "cuda"):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.feature_dim = feature_dim
        
        # Load DINO backbone
        if DINO_AVAILABLE:
            try:
                self.dino_model = torch.hub.load('facebookresearch/dinov2', dino_model_name, pretrained=True)
                self.dino_model.eval()
                # Freeze DINO parameters
                for param in self.dino_model.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"Error loading DINO model: {e}")
                self.dino_model = None
        else:
            self.dino_model = None
            
        # STEGO segmentation head
        self.stego_head = STEGOHead(feature_dim, num_classes, hidden_dim)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def extract_dino_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features using frozen DINO backbone
        Args:
            images: Input images of shape (B, C, H, W)
        Returns:
            features: DINO features of shape (B, H*W, feature_dim)
        """
        if self.dino_model is None:
            # Fallback: return random features if DINO not available
            B, C, H, W = images.shape
            return torch.randn(B, (H//14)*(W//14), self.feature_dim, device=self.device)
            
        with torch.no_grad():
            # Extract features using DINO
            features = self.dino_model.forward_features(images)
            # Remove CLS token and reshape
            if len(features.shape) == 3:
                features = features[:, 1:, :]  # Remove CLS token
            return features
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through STEGO model
        Args:
            images: Input images of shape (B, C, H, W)
        Returns:
            logits: Segmentation logits of shape (B, H*W, num_classes)
        """
        # Extract DINO features
        features = self.extract_dino_features(images)
        
        # Process through STEGO head
        logits = self.stego_head(features)
        
        return logits
    
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Predict segmentation masks
        Args:
            images: Input images of shape (B, C, H, W)
        Returns:
            predictions: Softmax probabilities of shape (B, num_classes, H, W)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(images)
            B, HW, C = logits.shape
            H = W = int(np.sqrt(HW))
            
            # Reshape and apply softmax
            logits = logits.view(B, H, W, C).permute(0, 3, 1, 2)
            predictions = F.softmax(logits, dim=1)
            
        return predictions
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for STEGO model
        Args:
            image: Input image as numpy array (H, W, C)
        Returns:
            tensor: Preprocessed image tensor (1, C, H, W)
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)


class STEGOTrainer:
    """
    Trainer class for STEGO model
    """
    def __init__(self, 
                 model: STEGOModel,
                 learning_rate: float = 0.0001,
                 device: str = "cuda"):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.stego_head.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Resize masks to match DINO output
            masks = F.interpolate(masks.float().unsqueeze(1), size=(14, 14), mode='nearest').squeeze(1).long()
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(images)
            B, HW, C = logits.shape
            H = W = int(np.sqrt(HW))
            
            # Reshape for loss calculation
            logits = logits.view(B, C, H, W)
            masks_flat = masks.view(B, -1)
            
            # Calculate loss
            loss = self.criterion(logits, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_pixels / total_pixels
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, dataloader) -> Dict[str, float]:
        """
        Validate the model
        """
        self.model.eval()
        total_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Resize masks to match DINO output
                masks = F.interpolate(masks.float().unsqueeze(1), size=(14, 14), mode='nearest').squeeze(1).long()
                
                # Forward pass
                logits = self.model(images)
                B, HW, C = logits.shape
                H = W = int(np.sqrt(HW))
                
                # Reshape for loss calculation
                logits = logits.view(B, C, H, W)
                
                # Calculate loss
                loss = self.criterion(logits, masks)
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                correct_pixels += (predictions == masks).sum().item()
                total_pixels += masks.numel()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_pixels / total_pixels
        
        return {'loss': avg_loss, 'accuracy': accuracy}


def create_stego_model(num_classes: int = 4, device: str = "cuda") -> STEGOModel:
    """
    Create a STEGO model instance
    """
    return STEGOModel(
        dino_model_name="dinov2_vitb14",
        num_classes=num_classes,
        device=device
    )


def load_stego_model(model_path: str, num_classes: int = 4, device: str = "cuda") -> STEGOModel:
    """
    Load a trained STEGO model
    """
    model = create_stego_model(num_classes, device)
    checkpoint = torch.load(model_path, map_location=device)
    model.stego_head.load_state_dict(checkpoint['stego_head_state_dict'])
    return model
