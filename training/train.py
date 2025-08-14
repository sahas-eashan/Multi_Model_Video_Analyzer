import os
from pathlib import Path
import argparse
import torchaudio
import torch
from tqdm import tqdm
import json
import sys
from meld_dataset import prepare_data_loader
from models import MultiModelTrainer, MultimodelSentimentAnalyzer
import imageio_ffmpeg

# Default paths for local development on Windows
DEFAULT_MODEL_DIR = Path("models").absolute()
DEFAULT_DATA_DIR = Path("data").absolute()

# AWS SageMaker or local paths
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", DEFAULT_MODEL_DIR))
SM_CHANNEL_TRAINING = Path(
    os.environ.get("SM_CHANNEL_TRAINING", DEFAULT_DATA_DIR / "training")
)
SM_CHANNEL_VALIDATION = Path(
    os.environ.get("SM_CHANNEL_VALIDATION", DEFAULT_DATA_DIR / "validation")
)
SM_CHANNEL_TEST = Path(os.environ.get("SM_CHANNEL_TEST", DEFAULT_DATA_DIR / "test"))

# CUDA memory allocation settings for Windows
if os.name == "nt":  # Windows
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
else:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    # Canonical names + env defaults (SageMaker-compatible)
    p.add_argument("--model-dir", dest="model_dir", default=str(SM_MODEL_DIR))
    p.add_argument(
        "--train-dir",
        "--train-data",
        dest="train_dir",
        default=str(SM_CHANNEL_TRAINING),
    )
    p.add_argument(
        "--val-dir", "--val-data", dest="val_dir", default=str(SM_CHANNEL_VALIDATION)
    )
    p.add_argument(
        "--test-dir", "--test-data", dest="test_dir", default=str(SM_CHANNEL_TEST)
    )
    return p.parse_args()


def main():
    # Make ffmpeg available to anything that shells out
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
    os.environ["PATH"] = (
        os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")
    )

    print("Available audio backends:")
    print(torchaudio.list_audio_backends())

    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(
            f"Initial GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB"
        )

    train_loader, val_loader, test_loader = prepare_data_loader(
        train_csv=os.path.join(args.train_dir, "train_sent_emo.csv"),
        train_vid_dir=os.path.join(args.train_dir, "train_splits"),
        dev_csv=os.path.join(args.val_dir, "dev_sent_emo.csv"),
        dev_vid_dir=os.path.join(args.val_dir, "dev_splits_complete"),
        test_csv=os.path.join(args.test_dir, "test_sent_emo.csv"),
        test_vid_dir=os.path.join(
            args.test_dir, "output_repeated_splits_test"
        ),  # change if your folder is different
        batch_size=args.batch_size,
    )

    print(f"Training CSV path: {os.path.join(args.train_dir, 'train_sent_emo.csv')}")
    print(f"Training video directory: {os.path.join(args.train_dir, 'train_splits')}")

    model = MultimodelSentimentAnalyzer().to(device)
    trainer = MultiModelTrainer(model, train_loader, val_loader)
    best_val_loss = float("inf")

    metrics_data = {"train_losses": [], "val_losses": [], "epochs": []}

    from tqdm import tqdm

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_loss = trainer.train_epoch()  # <-- call it
        val_loss, val_metrics = trainer.evaluate(val_loader)  # <-- once is enough

        metrics_data["train_losses"].append(train_loss["total"])
        metrics_data["val_losses"].append(val_loss["total"])
        metrics_data["epochs"].append(epoch)

        print(
            json.dumps(
                {
                    "metrics": [
                        {"name": "train_loss", "value": train_loss["total"]},
                        {"name": "val_loss", "value": val_loss["total"]},
                        {
                            "name": "validation:emotion_acc",
                            "value": val_metrics["emotion_accuracy"],
                        },
                        {
                            "name": "validation:sentiment_acc",
                            "value": val_metrics["sentiment_accuracy"],
                        },
                        {
                            "name": "validation:emotion_prec",
                            "value": val_metrics["emotion_precision"],
                        },
                        {
                            "name": "validation:sentiment_prec",
                            "value": val_metrics["sentiment_precision"],
                        },
                    ]
                }
            )
        )

        if torch.cuda.is_available():
            print(
                f"GPU memory allocated after epoch {epoch}: {torch.cuda.memory_allocated()/1024**3:.2f} GB"
            )

        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
            print(f"Best model saved at {os.path.join(args.model_dir, 'model.pth')}")

    # After training is complete then evl on test set
    print("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
    metrics_data["test_losses"] = test_loss["total"]

    print(
        json.dumps(
            {
                "metrics": [
                    {"name": "test_loss", "value": test_loss["total"]},
                    {
                        "name": "test_emotion_acc",
                        "value": test_metrics["emotion_accuracy"],
                    },
                    {
                        "name": "test_sentiment_acc",
                        "value": test_metrics["sentiment_accuracy"],
                    },
                    {
                        "name": "test_emotion_prec",
                        "value": test_metrics["emotion_precision"],
                    },
                    {
                        "name": "test_sentiment_prec",
                        "value": test_metrics["sentiment_precision"],
                    },
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
