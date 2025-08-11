from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess   

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
        return torch.FloatTensor(
            np.array(frames).permute(0, 3, 1, 2)
        )  # from [frames,height,width,channels] to [frames,channels,height,width]

    def extract_audio_frames(self, video_path):
        audio_path = video_path.replace(".mp4", ".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
                check=True,
            )   

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        video_file_name = f"""dia{item['Dialog_ID']}_utt{item['Utterance_ID']}.mp4"""
        path = os.path.join(self.Video_dir, video_file_name)
        video_path_exists = os.path.exists(path)
        if not video_path_exists:
            raise FileNotFoundError(
                f"Video file {video_file_name} not found in {self.Video_dir}"
            )
            return None

        test_inputs = self.tokenizer(
            item["utterance"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )

        video_frames = self.extract_video_frames(path)
        return {
            "input_ids": test_inputs["input_ids"].squeeze(),
            "attention_mask": test_inputs["attention_mask"].squeeze(),
            "video_path": video_path,
            "emotion": self.emotion_map[item["emotion"]],
            "sentiment": self.sentiment_map[item["sentiment"]],
        }


if __name__ == "__main__":
    # Example usage
    dataset = MeldDataset("path/to/csv_file.csv", "path/to/video_dir")
    print(len(dataset.data))  # Print the number of entries in the dataset
    print(dataset.data.head())  # Print the first few rows of the dataset
