import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path


class MultimodalEmotionDataset(Dataset):
    def __init__(self, csv_file, data_dir, preprocessor, split='train'):
        """
        Dataset for multimodal emotion recognition using MELD or similar datasets.
        """
        self.df = pd.read_csv(csv_file)
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.split = split

        # Emotion label mapping
        self.emotion_map = {
            'neutral': 0, 'joy': 1, 'surprise': 2,
            'anger': 3, 'sadness': 4, 'disgust': 5, 'fear': 6
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # MELD fields (Dialogue_ID, Utterance_ID, Utterance, Emotion)
        dialogue_id = str(row.get('Dialogue_ID', ''))
        utterance_id = str(row.get('Utterance_ID', ''))
        text = row.get('Utterance', '')
        emotion = row.get('Emotion', 'neutral')

        # Build paths (if local video/audio exist)
        video_path = self.data_dir / f"{self.split}_videos" / f"{dialogue_id}_{utterance_id}.mp4"
        audio_path = self.data_dir / f"{self.split}_audio" / f"{dialogue_id}_{utterance_id}.wav"

        # Process each modality
        visual = None
        audio = None

        try:
            if video_path.exists():
                visual = self.preprocessor.process_video(str(video_path))
        except Exception as e:
            print(f"Video processing failed for {video_path}: {e}")

        try:
            if audio_path.exists():
                audio = self.preprocessor.process_audio(str(audio_path))
        except Exception as e:
            print(f"⚠️ Audio processing failed for {audio_path}: {e}")

        # Map emotion label to int
        label = torch.tensor(self.emotion_map.get(emotion, 0), dtype=torch.long)

        return {
            'visual': visual,       # tensor [T, 3, H, W] or None
            'audio': audio,         # tensor [1, n_mels, time] or None
            'text': text,           # raw text string
            'label': label
        }


# Collate function for batching
def collate_fn(batch):
    """
    Collates multimodal samples with different lengths.
    """
    visuals = [item['visual'] for item in batch if item['visual'] is not None]
    audios = [item['audio'] for item in batch if item['audio'] is not None]
    texts = [item['text'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])

    batch_dict = {
        'visual': torch.stack(visuals) if visuals else None,
        'audio': torch.nn.utils.rnn.pad_sequence(audios, batch_first=True) if audios else None,
        'text': texts,
        'label': labels
    }

    return batch_dict
