import torch
from pathlib import Path
import sys
import os
from tqdm import tqdm
import json
import torch.cuda as cuda

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from training.models import MultimodelSentimentAnalyzer, MultiModelTrainer
from training.meld_dataset import prepare_data_loader
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument(
        "--batch-size", type=int, default=16
    )  # Reduced batch size for 6GB GPU
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--data-dir", type=str, default="dataset")
    p.add_argument("--save-dir", type=str, default="saved_models")
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)
    return p.parse_args()


def setup_gpu():
    if torch.cuda.is_available():
        # Set memory growth
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for cudnn
        return torch.device("cuda")
    return torch.device("cpu")


def save_model(model, save_path, config):
    """Save model for AWS deployment"""
    os.makedirs(save_path, exist_ok=True)

    # Save model state
    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))

    # Save model config
    with open(os.path.join(save_path, "model_config.json"), "w") as f:
        json.dump(config, f)


def main():
    args = parse_args()
    device = setup_gpu()
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        print("CUDA Device: ", torch.cuda.get_device_name(0))

    # Setup data paths
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)

    # Load data with progress bar
    print("Loading datasets...")
    train_loader, val_loader, test_loader = prepare_data_loader(
        train_csv=str(data_dir / "train" / "train_sent_emo.csv"),
        train_vid_dir=str(data_dir / "train" / "train_splits"),
        dev_csv=str(data_dir / "dev" / "dev_sent_emo.csv"),
        dev_vid_dir=str(data_dir / "dev" / "dev_splits"),
        test_csv=str(data_dir / "test" / "test_sent_emo.csv"),
        test_vid_dir=str(data_dir / "test" / "test_splits"),
        batch_size=args.batch_size,
    )

    # Initialize model with memory optimization
    model = MultimodelSentimentAnalyzer().to(device)
    if torch.cuda.is_available():
        try:
            # Use memory efficient optimizations
            from torch.cuda.amp import autocast

            scaler = torch.cuda.amp.GradScaler()
            print("Using mixed precision training")
        except:
            scaler = None
            print("Mixed precision training not available")

    # Create trainer with mixed precision support
    trainer = MultiModelTrainer(
        model,
        train_loader,
        val_loader,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        scaler=scaler if torch.cuda.is_available() else None,
    )

    # Training loop with progress bar
    best_val_acc = 0
    print("Starting training...")

    for epoch in range(args.epochs):
        # Training progress
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        epoch_loss = {"total": 0, "emotion": 0, "sentiment": 0}

        for batch in train_iterator:
            loss = trainer.train_step(batch)
            epoch_loss["total"] += loss["total"]
            train_iterator.set_postfix(loss=f"{loss['total']:.4f}")

        # Validation
        val_metrics = trainer.evaluate(val_loader)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {epoch_loss['total'] / len(train_loader):.4f}")
        print(f"Val Emotion Acc: {val_metrics['emotion_acc']:.4f}")
        print(f"Val Sentiment Acc: {val_metrics['sentiment_acc']:.4f}")

        # Save best model
        if val_metrics["emotion_acc"] > best_val_acc:
            best_val_acc = val_metrics["emotion_acc"]
            model_config = {
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "best_val_acc": best_val_acc,
                "epoch": epoch + 1,
            }
            save_model(model, save_dir, model_config)


if __name__ == "__main__":
    main()
