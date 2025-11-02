import torch
import cv2
import librosa
import numpy as np
from PIL import Image
from transformers import CLIPProcessor


class MultimodalPreprocessor:
    def __init__(self, device='cpu'):
        # CLIP processor for resizing/normalizing frames
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.sample_rate = 16000
        self.n_mels = 128
        self.device = device

    def process_video(self, video_path, fps_target=2):
        """Extract frames at ~2 FPS and return [T, 3, H, W] tensor for temporal processing"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ Could not open {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frames = []
        frame_idx = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Take approx 2 frames per second for temporal context
            if int(frame_idx % max(1, round(fps / fps_target))) == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frames.append(frame_pil)

            frame_idx += 1

        cap.release()

        if len(frames) == 0:
            print(f"No frames extracted from {video_path}")
            return None

        # Limit to 32 frames to save memory
        if len(frames) > 32:
            indices = np.linspace(0, len(frames)-1, 32, dtype=int)
            frames = [frames[i] for i in indices]

        processed = self.clip_processor(images=frames, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(self.device)
        return pixel_values  # shape: [T, 3, 224, 224]

    def process_audio(self, audio_path):
        """Extract improved mel-spectrogram with augmentation"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels,
            n_fft=2048, hop_length=512
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize to [0, 1]
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        
        mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)  # [1, n_mels, time]
        return mel_tensor.to(self.device)
