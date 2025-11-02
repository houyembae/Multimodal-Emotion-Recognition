import torch
import torch.nn as nn


class EnsembleMultimodalModel(nn.Module):
    """
    Ensemble combining early fusion and late fusion strategies.
    Improves robustness and generalisation.
    """
    def __init__(self, embedding_dim=256, num_emotions=7, device='cpu'):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_emotions = num_emotions
        
        # Early fusion pathway
        self.early_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions)
        )
        
        # Late fusion pathway - separate classifiers
        self.text_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions)
        )
        
        self.audio_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions)
        )
        
        self.visual_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions)
        )
        
        # Learnable ensemble weights
        self.early_weight = nn.Parameter(torch.tensor(0.5))
        self.late_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, text_emb, audio_emb, visual_emb):
        """
        Args:
            text_emb: [B, embedding_dim]
            audio_emb: [B, embedding_dim]
            visual_emb: [B, embedding_dim]
        Returns:
            logits: [B, num_emotions]
        """
        batch_size = text_emb.shape[0] if text_emb is not None else (
            audio_emb.shape[0] if audio_emb is not None else visual_emb.shape[0]
        )
        
        # Handle None values
        if text_emb is None:
            text_emb = torch.zeros(batch_size, self.embedding_dim, device=self.device)
        if audio_emb is None:
            audio_emb = torch.zeros(batch_size, self.embedding_dim, device=self.device)
        if visual_emb is None:
            visual_emb = torch.zeros(batch_size, self.embedding_dim, device=self.device)
        
        # Early fusion pathway
        concatenated = torch.cat([text_emb, audio_emb, visual_emb], dim=1)
        early_logits = self.early_fusion(concatenated)
        
        # Late fusion pathway
        text_logits = self.text_classifier(text_emb)
        audio_logits = self.audio_classifier(audio_emb)
        visual_logits = self.visual_classifier(visual_emb)
        
        late_logits = (text_logits + audio_logits + visual_logits) / 3
        
        # Ensemble with learned weights
        early_w = torch.sigmoid(self.early_weight)
        late_w = torch.sigmoid(self.late_weight)
        total_weight = early_w + late_w
        
        logits = (early_logits * early_w + late_logits * late_w) / total_weight
        
        return logits
