import torch
import numpy as np
import json
from pathlib import Path


class ComparativeAnalysis:
    """
    Analyzes the contribution of each modality to emotion recognition.
    Useful for understanding what the model learned.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {
            'text_only': {'accuracy': 0, 'f1': 0},
            'audio_only': {'accuracy': 0, 'f1': 0},
            'visual_only': {'accuracy': 0, 'f1': 0},
            'all_modalities': {'accuracy': 0, 'f1': 0},
            'modality_importance': {}
        }
    
    def evaluate_single_modality(self, model, dataloader, modality, device):
        """
        Evaluates model using only one modality at a time.
        """
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                text_emb = batch.get('text_emb')
                audio_emb = batch.get('audio_emb')
                visual_emb = batch.get('visual_emb')
                labels = batch['label'].to(device)
                
                # Zero out other modalities
                if modality != 'text':
                    text_emb = None
                if modality != 'audio':
                    audio_emb = None
                if modality != 'visual':
                    visual_emb = None
                
                outputs = model(text_emb, audio_emb, visual_emb)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels)
    
    def compute_modality_importance(self, accuracies):
        """
        Compute importance scores based on single-modality performance.
        """
        all_acc = accuracies.get('all_modalities', 0)
        
        importance = {
            'text': max(0, all_acc - accuracies.get('text_only', 0)) / (all_acc + 1e-8),
            'audio': max(0, all_acc - accuracies.get('audio_only', 0)) / (all_acc + 1e-8),
            'visual': max(0, all_acc - accuracies.get('visual_only', 0)) / (all_acc + 1e-8),
        }
        
        return importance
    
    def save_analysis(self, output_dir, analysis_dict):
        """Save comparative analysis results."""
        output_path = Path(output_dir) / 'modality_analysis.json'
        with open(output_path, 'w') as f:
            json.dump(analysis_dict, f, indent=2)
        print(f"Analysis saved to {output_path}")
