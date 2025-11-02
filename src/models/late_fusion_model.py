import torch
import torch.nn as nn


class LateFusionModel(nn.Module):
    """
    Late fusion architecture for multimodal emotion recognition.
    Each modality has its own classifier, predictions are averaged.
    """
    def __init__(self, embedding_dim=256, num_emotions=7, device='cpu'):
        super().__init__()
        self.device = device
        
        # Separate classifiers for each modality
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
        
        # Learnable weights for modality fusion
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, text_emb, audio_emb, visual_emb):
        """
        Args:
            text_emb: [B, embedding_dim] or None
            audio_emb: [B, embedding_dim] or None
            visual_emb: [B, embedding_dim] or None
        Returns:
            logits: [B, num_emotions]
            modality_scores: dict with individual modality logits
        """
        modality_scores = {}
        valid_modalities = []
        
        # Get predictions from each modality
        if text_emb is not None:
            text_logits = self.text_classifier(text_emb)
            modality_scores['text'] = text_logits
            valid_modalities.append(text_logits)
        
        if audio_emb is not None:
            audio_logits = self.audio_classifier(audio_emb)
            modality_scores['audio'] = audio_logits
            valid_modalities.append(audio_logits)
        
        if visual_emb is not None:
            visual_logits = self.visual_classifier(visual_emb)
            modality_scores['visual'] = visual_logits
            valid_modalities.append(visual_logits)
        
        # Fuse with learned weights
        if valid_modalities:
            # Normalize weights based on available modalities
            weights = torch.softmax(self.fusion_weights[:len(valid_modalities)], dim=0)
            logits = torch.stack(valid_modalities) * weights.view(-1, 1, 1)
            logits = logits.sum(dim=0)
        else:
            raise ValueError("No valid modalities available")
        
        return logits, modality_scores
