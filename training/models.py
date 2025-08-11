import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
from meld_dataset import MeldDataset
from sklearn.metrics import accuracy_score, precision_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # use CLS token representation
        pooler_output = outputs.pooler_output
        return self.projection(pooler_output)


class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128), nn.ReLU(), nn.Dropout(0.2)
        )

    def forward(self, x):
        # x shape: [batch_size, frames, channels, height, width] --> [batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)

        x = self.backbone(x)

        return x


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # lower level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        x = x.squeeze(1)
        # x shape: [batch_size, 128, 1]
        features = self.conv_layers(x)
        features = self.projection(features.squeeze(-1))
        return features


class MultimodelSentimentAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion Layer
        self.FusionLayer = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Classifiers
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 7)
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3)
        )

        self.current_train_losses = None

    def forward(self, text_inputs, video_frames, audio_features):
        text_features = self.text_encoder(
            text_inputs["input_ids"], text_inputs["attention_mask"]
        )

        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Concatenate features from all modalities
        combined = torch.cat((text_features, video_features, audio_features), dim=1)

        fused_features = self.FusionLayer(combined)

        return {
            "emotion": self.emotion_classifier(fused_features),
            "sentiment": self.sentiment_classifier(fused_features),
        }


class MultiModelTrainer:
    def __init__(self, model, train_dataloader, val_dataloader):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Log dataset Sizes
        train_size = len(train_dataloader.dataset)
        val_size = len(val_dataloader.dataset)
        print(f"Training dataset size: {train_size}")
        print(f"Validation dataset size: {val_size}")
        print(f"Batches per epoch: {len(train_dataloader):,}")

        # Use Windows-compatible timestamp format (replace : with -)
        timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
        base_dir = (
            "runs" if "SM_MODEL_DIR" not in os.environ else "/opt/ml/output/tensorboard"
        )
        log_dir = os.path.join(base_dir, f"run_{timestamp}")
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

        # very high =1, high = 0.1-0.01, medium = 1e-1, low = 1e-4 , very low = 1e-5
        self.optimizer = torch.optim.Adam(
            [
                {"params": model.text_encoder.parameters(), "lr": 8e-6},
                {"params": model.video_encoder.parameters(), "lr": 8e-5},
                {"params": model.audio_encoder.parameters(), "lr": 8e-5},
                {"params": model.FusionLayer.parameters(), "lr": 5e-4},
                {"params": model.emotion_classifier.parameters(), "lr": 5e-4},
                {"params": model.sentiment_classifier.parameters(), "lr": 5e-4},
            ],
            weight_decay=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=2
        )

        self.emotion_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.sentiment_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    def log_metrics(self, losses, metrics=None, phase="train"):
        """
        Log training/validation metrics to TensorBoard
        """
        # Log losses
        self.writer.add_scalar(f"{phase}/loss/total", losses["total"], self.global_step)
        self.writer.add_scalar(
            f"{phase}/loss/emotion", losses["emotion"], self.global_step
        )
        self.writer.add_scalar(
            f"{phase}/loss/sentiment", losses["sentiment"], self.global_step
        )

        # Log metrics if provided
        if metrics:
            self.writer.add_scalar(
                f"{phase}/accuracy/emotion", metrics["emotion_acc"], self.global_step
            )
            self.writer.add_scalar(
                f"{phase}/accuracy/sentiment",
                metrics["sentiment_acc"],
                self.global_step,
            )
            self.writer.add_scalar(
                f"{phase}/precision/emotion", metrics["emotion_prec"], self.global_step
            )
            self.writer.add_scalar(
                f"{phase}/precision/sentiment",
                metrics["sentiment_prec"],
                self.global_step,
            )

        # Log learning rates
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(
                f"{phase}/learning_rate/group_{i}", param_group["lr"], self.global_step
            )

    def train_epoch(self):
        self.model.train()
        running_loss = {"total": 0, "emotion": 0, "sentiment": 0}
        for batch in self.train_dataloader:
            torch.device = next(self.model.parameters()).device

            text_inputs = {
                "input_ids": batch["text_inputs"]["input_ids"].todevice,
                "attention_mask": batch["text_inputs"]["attention_mask"].todevice,
            }
            video_frames = batch["video_frames"].todevice
            audio_features = batch["audio_features"].todevice
            emotion_labels = batch["emotion_labels"].todevice
            sentiment_labels = batch["sentiment_labels"].todevice

            # zero the gradients
            self.optimizer.zero_grad()

            # forward pass
            outputs = self.model(text_inputs, video_frames, audio_features)

            # calculate the losses
            emotion_loss = self.emotion_criterion(outputs["emotion"], emotion_labels)
            sentiment_loss = self.sentiment_criterion(
                outputs["sentiment"], sentiment_labels
            )

            total_loss = emotion_loss + sentiment_loss
            # backward pass
            total_loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # track losses
            running_loss["total"] += total_loss.item()
            running_loss["emotion"] += emotion_loss.item()
            running_loss["sentiment"] += sentiment_loss.item()

            self.log_metrics(
                {
                    "total": running_loss["total"].item(),
                    "emotion": running_loss["emotion"].item(),
                    "sentiment": running_loss["sentiment"].item(),
                }
            )

            self.global_step += 1

        return {k: v / len(self.train_dataloader) for k, v in running_loss.items()}

    def evaluate(self, data_loader, phase="val"):
        self.model.eval()
        losses = {"total": 0, "emotion": 0, "sentiment": 0}
        all_emotion_preds = []
        all_emotion_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []

        with torch.inference_mode():
            for batch in data_loader:
                device = next(self.model.parameters()).device
                text_inputs = {
                    "input_ids": batch["text_inputs"]["input_ids"].to(device),
                    "attention_mask": batch["text_inputs"]["attention_mask"].to(device),
                }
                video_frames = batch["video_frames"].to(device)
                audio_features = batch["audio_features"].to(device)
                sentiment_labels = batch["sentiment_labels"].to(device)
                emotion_labels = batch["emotion_labels"].to(device)

                outputs = self.model(text_inputs, video_frames, audio_features)

                emotion_loss = self.emotion_criterion(
                    outputs["emotion"], emotion_labels
                )
                sentiment_loss = self.sentiment_criterion(
                    outputs["sentiment"], sentiment_labels
                )

                total_loss = emotion_loss + sentiment_loss

                all_emotion_preds.extend(outputs["emotion"].argmax(dim=1).cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_sentiment_preds.extend(
                    outputs["sentiment"].argmax(dim=1).cpu().numpy()
                )
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

                # track losses
                losses["total"] += total_loss.item()
                losses["emotion"] += emotion_loss.item()
                losses["sentiment"] += sentiment_loss.item()

        avg_loss = {k: v / len(data_loader) for k, v in losses.items()}

        # calculate precision and accuracy
        emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
        sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)
        emotion_precision = precision_score(
            all_emotion_labels, all_emotion_preds, average="weighted"
        )
        sentiment_precision = precision_score(
            all_sentiment_labels, all_sentiment_preds, average="weighted"
        )

        if phase == "val":
            self.scheduler.step(avg_loss["total"])

        self.log_metrics(
            avg_loss,
            {
                "val_emotion_accuracy": emotion_accuracy,
                "val_sentiment_accuracy": sentiment_accuracy,
                "val_emotion_precision": emotion_precision,
                "val_sentiment_precision": sentiment_precision,
            },
            phase=phase,
        )

        return avg_loss, {
            "emotion_accuracy": emotion_accuracy,
            "sentiment_accuracy": sentiment_accuracy,
            "emotion_precision": emotion_precision,
            "sentiment_precision": sentiment_precision,
        }


if __name__ == "__main__":
    dataset = MeldDataset(
        "../dataset/train/train_sent_emo.csv",
        "../dataset/train/train_splits",
    )
    sample = dataset[0]

    model = MultimodelSentimentAnalyzer()
    model.eval()

    text_inputs = {
        "input_ids": sample["text_inputs"]["input_ids"].unsqueeze(0),
        "attention_mask": sample["text_inputs"]["attention_mask"].unsqueeze(0),
    }

    video_frames = sample["video_frames"].unsqueeze(0)
    audio_features = sample["audio_features"].unsqueeze(0)

    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)
        emotion_probs = torch.softmax(outputs["emotion"], dim=1)[0]
        sentiment_probs = torch.softmax(outputs["sentiment"], dim=1)[0]

    print(emotion_probs)
    print(sentiment_probs)
    emotion_map = {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "joy",
        4: "neutral",
        5: "sadness",
        6: "surprise",
    }
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}

    for i, prob in enumerate(sentiment_probs):
        print(f"Sentiment {sentiment_map[i]}: {prob:.2f}")
    for i, prob in enumerate(emotion_probs):
        print(f"Emotion {emotion_map[i]}: {prob:.2f}")

    print("Model inference completed successfully.")
