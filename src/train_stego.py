"""
Training script for STEGO model
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import yaml
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.stego_model import STEGOModel, STEGOTrainer
from utils.constants import Constants
from utils.logger import custom_logger
from utils.root_config import get_root_config
from utils.dataset import SegmentationDataset
from utils.preprocess import get_preprocessing
import segmentation_models_pytorch as smp


class STEGODataset:
    """
    Custom dataset for STEGO training
    """
    def __init__(self, images_dir: str, masks_dir: str, classes: list, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.classes = classes
        self.transform = transform
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
        
        # Create class mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = os.path.join(self.masks_dir, self.image_files[idx])
        mask = cv2.imread(mask_path, 0)
        
        # Convert mask to class indices
        mask_indices = np.zeros_like(mask)
        for cls, idx in self.class_to_idx.items():
            if cls == 'background':
                mask_indices[mask == 0] = idx
            elif cls == 'building':
                mask_indices[mask == 1] = idx
            elif cls == 'woodland':
                mask_indices[mask == 2] = idx
            elif cls == 'water':
                mask_indices[mask == 3] = idx
            elif cls == 'road':
                mask_indices[mask == 4] = idx
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask_indices)
            image = augmented['image']
            mask_indices = augmented['mask']
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_indices = torch.from_numpy(mask_indices).long()
        
        return image, mask_indices


def train_stego_model():
    """
    Main training function for STEGO model
    """
    # Load configuration
    ROOT, slice_config = get_root_config(__file__, Constants)
    
    # Get configuration parameters
    log_level = slice_config['vars']['log_level']
    file_type = slice_config['vars']['file_type']
    patch_size = slice_config['vars']['patch_size']
    batch_size = slice_config['vars']['batch_size']
    epochs = slice_config['vars']['epochs']
    device = slice_config['vars']['device']
    all_classes = slice_config['vars']['all_classes']
    train_classes = slice_config['vars']['train_classes']
    
    # Setup logging
    log_dir = ROOT / slice_config['dirs']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "stego_train.log"
    logger = custom_logger("STEGO Training Logs", log_path.as_posix(), log_level)
    
    # Setup paths
    train_img_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['train_dir'] / slice_config['dirs']['image_dir']
    train_mask_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['train_dir'] / slice_config['dirs']['mask_dir']
    
    # Create patches directory
    patches_dir = train_img_dir.parent / f"patches_{patch_size}"
    patches_img_dir = patches_dir / "images"
    patches_mask_dir = patches_dir / "masks"
    
    patches_img_dir.mkdir(parents=True, exist_ok=True)
    patches_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Create patches if they don't exist
    if not any(patches_img_dir.iterdir()):
        logger.info("Creating image patches...")
        from utils.patching import patching
        patching(str(train_img_dir), str(patches_img_dir), file_type, patch_size)
        
        logger.info("Creating mask patches...")
        patching(str(train_mask_dir), str(patches_mask_dir), file_type, patch_size)
    
    # Create dataset
    from utils.preprocess import get_training_augmentation
    train_dataset = STEGODataset(
        str(patches_img_dir),
        str(patches_mask_dir),
        train_classes,
        transform=get_training_augmentation()
    )
    
    # Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Create STEGO model
    logger.info("Creating STEGO model...")
    model = STEGOModel(
        dino_model_name="dinov2_vitb14",
        num_classes=len(train_classes),
        device=device
    )
    
    # Create trainer
    trainer = STEGOTrainer(model, learning_rate=0.0001, device=device)
    
    # Training loop
    logger.info("Starting STEGO training...")
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Train epoch
        train_metrics = trainer.train_epoch(train_loader, epoch)
        train_losses.append(train_metrics['loss'])
        train_accuracies.append(train_metrics['accuracy'])
        
        logger.info(f"Epoch {epoch+1} - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'stego_head_state_dict': model.stego_head.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy']
            }
            
            model_dir = ROOT / slice_config['dirs']['model_dir']
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, model_dir / f"stego_model_epoch_{epoch+1}.pth")
            logger.info(f"Model saved at epoch {epoch+1}")
    
    # Save final model
    final_checkpoint = {
        'epoch': epochs,
        'stego_head_state_dict': model.stego_head.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'train_accuracy': train_accuracies[-1],
        'classes': train_classes
    }
    
    model_dir = ROOT / slice_config['dirs']['model_dir']
    torch.save(final_checkpoint, model_dir / "stego_model_final.pth")
    logger.info("Final model saved")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(model_dir / "stego_training_curves.png")
    plt.close()
    
    logger.info("STEGO training completed successfully!")


if __name__ == "__main__":
    train_stego_model()
