import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class CLIPVisualEncoder(nn.Module):
    """
    CLIP-based visual encoder for multimodal emotion recognition.
    Uses pretrained CLIP ViT model for advanced visual understanding.
    """
    def __init__(self, embedding_dim=256, device='cpu'):
        super().__init__()
        self.device = device
        
        # Load pretrained CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze CLIP weights (transfer learning)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # CLIP outputs 512-dim embeddings, project to embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )
    
    def forward(self, frames_batch):
        """
        Args:
            frames_batch: [B, T, 3, 224, 224] - batch of video frames
        Returns:
            embeddings: [B, embedding_dim] - aggregated visual embeddings
        """
        if frames_batch is None:
            return None
        
        batch_size, num_frames = frames_batch.shape[:2]
        
        # Process frames through CLIP
        with torch.no_grad():
            # Reshape to [B*T, 3, 224, 224]
            frames_flat = frames_batch.reshape(batch_size * num_frames, *frames_batch.shape[2:])
            
            # Get CLIP vision embeddings
            image_features = self.clip_model.get_image_features(pixel_values=frames_flat)
            
            # Reshape back to [B, T, 512]
            image_features = image_features.reshape(batch_size, num_frames, -1)
        
        # Temporal pooling (mean over frames)
        visual_embedding = image_features.mean(dim=1)  # [B, 512]
        
        # Project to embedding_dim
        visual_embedding = self.projection(visual_embedding)  # [B, embedding_dim]
        
        return visual_embedding
