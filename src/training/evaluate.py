import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(model, text_encoder, audio_encoder, visual_encoder, dataloader, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model: FusionModel
        text_encoder: TextEncoder
        audio_encoder: AudioEncoder
        visual_encoder: VisualEncoder
        dataloader: DataLoader with collate_fn
        device: torch device
    
    Returns:
        accuracy: accuracy score
        precision, recall, f1: weighted metrics
    """
    model.eval()
    text_encoder.eval()
    audio_encoder.eval()
    visual_encoder.eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            # Extract batch components
            text_list = batch['text']
            audio = batch['audio'].to(device) if batch['audio'] is not None else None
            visual = batch['visual'].to(device) if batch['visual'] is not None else None
            labels = batch['label'].to(device)
            
            text_emb = text_encoder(text_list, device=device)
            audio_emb = audio_encoder(audio) if audio is not None else torch.zeros(len(text_list), 256, device=device)
            visual_emb = visual_encoder(visual) if visual is not None else torch.zeros(len(text_list), 256, device=device)
            
            text_emb = text_emb.to(device)
            audio_emb = audio_emb.to(device)
            visual_emb = visual_emb.to(device)
            
            # Forward pass
            preds = model(text_emb, audio_emb, visual_emb)
            all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    return accuracy, precision, recall, f1
