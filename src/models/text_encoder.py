from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch

class TextEncoder(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', embedding_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        
        self.projection = nn.Sequential(
            nn.Linear(768, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )
    
    def forward(self, text_list, device='cpu'):
        """
        Encode text to embeddings.
        
        Args:
            text_list: list of text strings
            device: torch device
        
        Returns:
            embeddings: [B, embedding_dim]
        """
        tokens = self.tokenizer(
            text_list, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        )
        
        # Move tokens to device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        out = self.bert(**tokens)
        embeddings = self.projection(out.last_hidden_state[:, 0, :])  # [B, embedding_dim]
        
        return embeddings
