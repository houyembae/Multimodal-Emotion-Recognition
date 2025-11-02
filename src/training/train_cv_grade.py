import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from pathlib import Path


def train_epoch_cv(
    model, train_loader, text_encoder, audio_encoder, visual_encoder,
    optimizer, criterion, device, epoch
):
    """
    Training loop for CV-grade model with multiple encoders.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(train_loader):
        texts = batch['text']
        labels = batch['label'].to(device)
        
        # Encode modalities
        with torch.no_grad():
            text_emb = text_encoder(texts, device=device) if texts else None
            
            audio = batch.get('audio')
            audio_emb = audio_encoder(audio.to(device)) if audio is not None else None
            
            visual = batch.get('visual')
            visual_emb = visual_encoder(visual.to(device)) if visual is not None else None
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(text_emb, audio_emb, visual_emb)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item()
        total_samples += labels.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy


def evaluate_cv(
    model, eval_loader, text_encoder, audio_encoder, visual_encoder,
    criterion, device
):
    """
    Evaluation loop for CV-grade model.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            texts = batch['text']
            labels = batch['label'].to(device)
            
            # Encode modalities
            text_emb = text_encoder(texts, device=device) if texts else None
            
            audio = batch.get('audio')
            audio_emb = audio_encoder(audio.to(device)) if audio is not None else None
            
            visual = batch.get('visual')
            visual_emb = visual_encoder(visual.to(device)) if visual is not None else None
            
            # Forward pass
            logits = model(text_emb, audio_emb, visual_emb)
            loss = criterion(logits, labels)
            
            # Metrics
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(eval_loader)
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy
