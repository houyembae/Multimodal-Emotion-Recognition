import torch
import torch.nn as nn


class SpectralAudioEncoder(nn.Module):
    """
    Audio encoder using mel-spectrogram features.
    Designed for emotion recognition with attention to prosody.
    """
    def __init__(self, embedding_dim=256, n_mels=128, device='cpu'):
        super().__init__()
        self.device = device
        self.n_mels = n_mels
        
        # CNN layers for spectrogram processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        
        # Attention mechanism for important frequency bands
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Softmax(dim=-1)
        )
        
        # Projection to embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )
    
    def forward(self, mel_spectrogram):
        """
        Args:
            mel_spectrogram: [B, 1, n_mels, time] - mel-spectrogram
        Returns:
            embeddings: [B, embedding_dim]
        """
        if mel_spectrogram is None:
            return None
        
        # Ensure shape is correct
        if mel_spectrogram.dim() == 3:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)
        
        # Pad time dimension if too short
        if mel_spectrogram.shape[-1] < 64:
            pad_size = 64 - mel_spectrogram.shape[-1]
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, pad_size))
        
        # CNN feature extraction
        features = self.conv_layers(mel_spectrogram)  # [B, 128, h, w]
        
        # Global average pooling
        pooled = features.mean(dim=(2, 3))  # [B, 128]
        
        # Attention weighting
        attention_weights = self.attention(pooled)  # [B, 128]
        pooled = pooled * attention_weights  # Element-wise multiplication
        
        # Project to embedding_dim
        embeddings = self.projection(pooled)  # [B, embedding_dim]
        
        return embeddings
