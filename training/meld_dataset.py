from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import torchaudio
import imageio_ffmpeg
import tempfile

_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["IMAGEIO_FFMPEG_EXE"] = _ffmpeg
os.environ["PATH"] = os.path.dirname(_ffmpeg) + os.pathsep + os.environ.get("PATH", "")


class MeldDataset(Dataset):
    def __init__(self, csv_file, Video_dir):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.Video_dir = Video_dir
        self.emotion_map = {
            0: "anger",
            1: "disgust",
            2: "fear",
            3: "joy",
            4: "neutral",
            5: "sadness",
            6: "surprise",
        }
        self.sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        self.emotion_str2id = {v: k for k, v in self.emotion_map.items()}
        self.sentiment_str2id = {v: k for k, v in self.sentiment_map.items()}

    def __len__(self):
        return len(self.data)

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise ValueError("Error opening video file")
            frames = []
            while len(frames) < 30 and cap.isOpened():  # Load only the first 30 frames
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))  # Resize frame to 224x224
                frame = frame / 255.0  # Normalize pixel values to [0, 1]
                frames.append(frame)
        except Exception as e:
            raise ValueError(f"Error loading video frames: {e}")
        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError("No frames extracted from video")

        # pad or truncate frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]
        return torch.FloatTensor(np.array(frames)).permute(
            0, 3, 1, 2
        )  # from [frames,height,width,channels] to [frames,channels,height,width]

    def _silent_mel(self):
        # fallback to a zero feature if extraction fails
        return torch.zeros(1, 64, 300, dtype=torch.float32)

    def extract_audio_features(self, video_path):
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()  # <- absolute path

        # temp WAV (safer than .replace(".mp4", ".wav"))
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_path = tmp.name
        tmp.close()

        try:
            cmd = [
                ffmpeg,  # <- use resolved binary
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",  # quieter logs
                "-y",  # overwrite temp file
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                audio_path,
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if (
                proc.returncode != 0
                or not os.path.exists(audio_path)
                or os.path.getsize(audio_path) == 0
            ):
                # No audio stream or extraction failed
                return self._silent_mel()

            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64
            )
            mel_spec = mel_spectrogram(waveform)

            # normalize (avoid div-by-zero)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

            # pad/crop to 300 frames
            T = mel_spec.size(-1)
            if T < 300:
                mel_spec = torch.nn.functional.pad(mel_spec, (0, 300 - T))
            else:
                mel_spec = mel_spec[..., :300]

            return mel_spec

        except Exception:
            # Any error -> safe fallback
            return self._silent_mel()
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

    def __getitem__(self, idx):

        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        item = self.data.iloc[idx]
        try:
            video_file_name = (
                f"""dia{item['Dialogue_ID']}_utt{item['Utterance_ID']}.mp4"""
            )
            path = os.path.join(self.Video_dir, video_file_name)
            video_path_exists = os.path.exists(path)
            if not video_path_exists:
                raise FileNotFoundError(
                    f"Video file {video_file_name} not found in {self.Video_dir}"
                )
                return None

            test_inputs = self.tokenizer(
                item["Utterance"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128,
            )

            video_frames = self.load_video_frames(path)
            audio_features = self.extract_audio_features(path)

            # map sentiments and emotion labels
            emotion_label = self.emotion_str2id[item["Emotion"].lower()]
            sentiment_label = self.sentiment_str2id[item["Sentiment"].lower()]

            return {
                "text_inputs": {
                    "input_ids": test_inputs["input_ids"].squeeze(),
                    "attention_mask": test_inputs["attention_mask"].squeeze(),
                },
                "video_frames": video_frames,
                "audio_features": audio_features,
                "emotion": torch.tensor(emotion_label),
                "sentiment": torch.tensor(sentiment_label),
            }
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            return None


def collate_fn(batch):
    # filter out none samples
    batch = list(filter(None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def prepare_data_loader(
    train_csv,
    train_vid_dir,
    dev_csv,
    dev_vid_dir,
    test_csv,
    test_vid_dir,
    batch_size=32,
):
    train_dataset = MeldDataset(train_csv, train_vid_dir)
    dev_dataset = MeldDataset(dev_csv, dev_vid_dir)
    test_dataset = MeldDataset(test_csv, test_vid_dir)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    train_loader, dev_loader, test_loader = prepare_data_loader(
        "../dataset/train/train_sent_emo.csv",
        "../dataset/train/train_splits",
        "../dataset/dev/dev_sent_emo.csv",
        "../dataset/dev/dev_splits_complete",
        "../dataset/test/test_sent_emo.csv",
        "../dataset/test/output_repeated_splits_test",
        batch_size=32,
    )

    for batch in train_loader:
        print(batch["text_inputs"])
        print(batch["video_frames"].shape)
        print(batch["audio_features"].shape)
        print(batch["emotion"])
        print(batch["sentiment"])
