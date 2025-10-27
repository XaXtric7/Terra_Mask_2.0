"""
Simplified STEGO implementation without DINO dependency
Uses a pre-trained ResNet backbone as feature extractor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional


class SimpleSTEGOHead(nn.Module):
    """
    Simplified STEGO segmentation head
    """
    def __init__(self, feature_dim: int = 2048, num_classes: int = 4, hidden_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Feature processing layers
        self.conv1 = nn.Conv2d(feature_dim, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, num_classes, 1)
        
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Backbone features of shape (B, C, H, W)
        Returns:
            logits: Segmentation logits of shape (B, num_classes, H, W)
        """
        x = F.relu(self.bn1(self.conv1(features)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.conv3(x)
        
        return x


class SimpleSTEGOModel(nn.Module):
    """
    Simplified STEGO model with ResNet backbone
    """
    def __init__(self, 
                 backbone_name: str = "resnet50",
                 num_classes: int = 4,
                 pretrained: bool = True,
                 device: str = "cuda"):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        
        # Load backbone
        if backbone_name == "resnet50":
            if pretrained:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet50(weights=None)
            feature_dim = 2048
        elif backbone_name == "resnet34":
            if pretrained:
                self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet34(weights=None)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # STEGO segmentation head
        self.stego_head = SimpleSTEGOHead(feature_dim, num_classes)

        # ✅ Move backbone and head to device
        self.backbone = self.backbone.to(device)
        self.stego_head = self.stego_head.to(device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features using backbone
        Args:
            images: Input images of shape (B, C, H, W)
        Returns:
            features: Backbone features of shape (B, C, H, W)
        """
        with torch.no_grad():
            features = self.backbone(images)
        return features
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through STEGO model
        Args:
            images: Input images of shape (B, C, H, W)
        Returns:
            logits: Segmentation logits of shape (B, num_classes, H, W)
        """
        # Extract backbone features
        features = self.extract_features(images)
        
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


class SimpleSTEGOTrainer:
    """
    Trainer class for simplified STEGO model
    """
    def __init__(self, 
                 model: SimpleSTEGOModel,
                 learning_rate: float = 0.001,
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
            
            # Resize masks to match backbone output (8x8 for ResNet50)
            masks = F.interpolate(masks.float().unsqueeze(1), size=(8, 8), mode='nearest').squeeze(1).long()
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(images)
            
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
                
                # Resize masks to match backbone output
                masks = F.interpolate(masks.float().unsqueeze(1), size=(8, 8), mode='nearest').squeeze(1).long()
                
                # Forward pass
                logits = self.model(images)
                
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


def create_simple_stego_model(num_classes: int = 4, device: str = "cuda") -> SimpleSTEGOModel:
    """
    Create a simplified STEGO model instance
    """
    return SimpleSTEGOModel(
        backbone_name="resnet50",
        num_classes=num_classes,
        pretrained=True,
        device=device
    )


def load_simple_stego_model(model_path: str, num_classes: int = 4, device: str = "cuda") -> SimpleSTEGOModel:
    """
    Load a trained simplified STEGO model
    """
    model = create_simple_stego_model(num_classes, device)
    checkpoint = torch.load(model_path, map_location=device)
    model.stego_head.load_state_dict(checkpoint['stego_head_state_dict'])
    return model
