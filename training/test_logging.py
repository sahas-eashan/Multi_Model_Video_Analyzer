import torch
from torch.utils.data import DataLoader, Dataset
from models import MultiModelTrainer, MultimodelSentimentAnalyzer
import time


class MockDataset(Dataset):
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "text_inputs": {
                "input_ids": torch.randint(0, 1000, (128,)),
                "attention_mask": torch.ones(128),
            },
            "video_frames": torch.randn(30, 3, 224, 224),
            "audio_features": torch.randn(1, 64, 300),
            "emotion": torch.tensor(4),
            "sentiment": torch.tensor(1),
        }


def test_logging():
    model = MultimodelSentimentAnalyzer()

    # Create proper DataLoader objects
    mock_train_dataset = MockDataset()
    mock_val_dataset = MockDataset()

    train_loader = DataLoader(mock_train_dataset, batch_size=2)
    val_loader = DataLoader(mock_val_dataset, batch_size=2)

    # Initialize trainer with DataLoader objects
    trainer = MultiModelTrainer(model, train_loader, val_loader)

    # Simulate some training metrics
    for i in range(100):
        fake_losses = {
            "total": torch.tensor(1.0 - 0.008 * i),
            "emotion": torch.tensor(0.6 - 0.005 * i),
            "sentiment": torch.tensor(0.4 - 0.003 * i),
        }

        fake_metrics = {
            "emotion_acc": 0.5 + 0.004 * i,
            "sentiment_acc": 0.6 + 0.003 * i,
            "emotion_prec": 0.55 + 0.003 * i,
            "sentiment_prec": 0.58 + 0.004 * i,
        }

        trainer.log_metrics(fake_losses, fake_metrics, phase="train")
        trainer.global_step += 1
        time.sleep(0.01)  # Small delay to simulate training time

    trainer.writer.flush()
    trainer.writer.close()

    # Assertions
    assert hasattr(trainer, "train_dataloader")
    assert hasattr(trainer, "val_dataloader")


if __name__ == "__main__":
    test_logging()
