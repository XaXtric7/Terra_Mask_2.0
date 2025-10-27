"""
Main script to run the complete comparison between U-Net and STEGO models
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.constants import Constants
from utils.logger import custom_logger
from utils.root_config import get_root_config


def run_training():
    """Run training for both models"""
    print("="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    # Train simplified STEGO model
    print("\n1. Training simplified STEGO model...")
    try:
        subprocess.run([sys.executable, "train_simple_stego.py"], check=True)
        print("✓ STEGO model training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ STEGO model training failed: {e}")
        return False
    
    print("\n2. U-Net model is already trained and available.")
    print("✓ Using existing U-Net model for comparison.")
    
    return True


def run_comparison():
    """Run model comparison"""
    print("\n" + "="*60)
    print("COMPARISON PHASE")
    print("="*60)
    
    print("\nRunning comprehensive model comparison...")
    try:
        subprocess.run([sys.executable, "compare_models.py"], check=True)
        print("✓ Model comparison completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Model comparison failed: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run U-Net vs STEGO model comparison")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip training phase and only run comparison")
    parser.add_argument("--skip-comparison", action="store_true",
                       help="Skip comparison phase and only run training")
    
    args = parser.parse_args()
    
    # Load configuration
    ROOT, slice_config = get_root_config(__file__, Constants)
    
    # Setup logging
    log_dir = ROOT / slice_config['dirs']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "comparison_pipeline.log"
    logger = custom_logger("Comparison Pipeline Logs", log_path.as_posix(), "INFO")
    
    logger.info("Starting U-Net vs STEGO comparison pipeline")
    
    success = True
    
    # Training phase
    if not args.skip_training:
        logger.info("Starting training phase...")
        success = run_training()
        if not success:
            logger.error("Training phase failed!")
            return
    else:
        print("Skipping training phase as requested.")
    
    # Comparison phase
    if not args.skip_comparison:
        logger.info("Starting comparison phase...")
        success = run_comparison()
        if not success:
            logger.error("Comparison phase failed!")
            return
    else:
        print("Skipping comparison phase as requested.")
    
    if success:
        print("\n" + "="*60)
        print("COMPARISON PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nResults are available in:")
        print(f"- Comparison plots: {ROOT}/output/model_comparison/")
        print(f"- Detailed metrics: {ROOT}/output/model_comparison/detailed_comparison.csv")
        print(f"- Summary report: {ROOT}/output/model_comparison/summary_report.json")
        print("\nCheck the comparison directory for detailed results and visualizations.")
        logger.info("Comparison pipeline completed successfully!")
    else:
        print("\n" + "="*60)
        print("COMPARISON PIPELINE FAILED!")
        print("="*60)
        logger.error("Comparison pipeline failed!")


if __name__ == "__main__":
    main()
