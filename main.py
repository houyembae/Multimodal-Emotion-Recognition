import os
import sys
import torch
import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# Import your modules
from src.data.preprocessing import MultimodalPreprocessor
from src.data.dataset import MultimodalEmotionDataset, collate_fn
from src.models.text_encoder import TextEncoder
from src.models.clip_visual_encoder import CLIPVisualEncoder
from src.models.audio_encoder_spectral import SpectralAudioEncoder
from src.models.ensemble_model import EnsembleMultimodalModel
from src.training.train_cv_grade import train_epoch_cv, evaluate_cv
from src.training.evaluate import evaluate_model
from src.analysis.comparative_analysis import ComparativeAnalysis

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directories
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load dataset
print("Loading dataset...")
preprocessor = MultimodalPreprocessor()
data_dir = Path('.')  # Assuming train/dev/test folders are in the same directory

csv_dir = Path('data/raw')
train_csv = csv_dir / 'meld_train.csv'
val_csv = csv_dir / 'meld_evaluation.csv'
test_csv = csv_dir / 'meld_test.csv'

# Check if CSV files exist, if not print helpful error message
for csv_file in [train_csv, val_csv, test_csv]:
    if not csv_file.exists():
        print(f"Warning: {csv_file} not found in {csv_dir}")

train_dataset = MultimodalEmotionDataset(str(train_csv), data_dir, preprocessor, split='train')
val_dataset = MultimodalEmotionDataset(str(val_csv), data_dir, preprocessor, split='dev')
test_dataset = MultimodalEmotionDataset(str(test_csv), data_dir, preprocessor, split='test')

# Create dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Initialize models
print("\nInitializing CV-Grade Architecture...")
text_encoder = TextEncoder().to(device)
visual_encoder = CLIPVisualEncoder(device=device).to(device)
audio_encoder = SpectralAudioEncoder(device=device).to(device)

# Initialize ensemble model (combines early + late fusion)
model = EnsembleMultimodalModel(
    embedding_dim=256,
    num_emotions=7,
    device=device
).to(device)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 15
best_val_acc = 0
training_history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'val_f1': []
}

print("\nStarting training...")
for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train_epoch_cv(
        model, train_loader, text_encoder, audio_encoder, visual_encoder,
        optimizer, loss_fn, device, epoch
    )
    
    # Validate
    val_loss, val_acc = evaluate_cv(
        model, val_loader, text_encoder, audio_encoder, visual_encoder,
        loss_fn, device
    )
    
    from sklearn.metrics import f1_score
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            texts = batch['text']
            labels = batch['label'].to(device)
            text_emb = text_encoder(texts, device=device) if texts else None
            audio = batch.get('audio')
            audio_emb = audio_encoder(audio.to(device)) if audio is not None else None
            visual = batch.get('visual')
            visual_emb = visual_encoder(visual.to(device)) if visual is not None else None
            logits = model(text_emb, audio_emb, visual_emb)
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
    
    training_history['train_loss'].append(train_loss)
    training_history['train_acc'].append(train_acc)
    training_history['val_loss'].append(val_loss)
    training_history['val_acc'].append(val_acc)
    training_history['val_f1'].append(val_f1)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
    
    # Scheduler step
    scheduler.step(val_acc)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'checkpoints/best_model_cv_grade.pt')
        print(f"  Best model saved!")

# Load best model
print("\nLoading best model for evaluation...")
model.load_state_dict(torch.load('checkpoints/best_model_cv_grade.pt'))

print("\nEvaluating on test set...")
test_loss, test_acc = evaluate_cv(
    model, test_loader, text_encoder, audio_encoder, visual_encoder,
    loss_fn, device
)

test_preds = []
test_labels = []
with torch.no_grad():
    for batch in test_loader:
        texts = batch['text']
        labels = batch['label'].to(device)
        text_emb = text_encoder(texts, device=device) if texts else None
        audio = batch.get('audio')
        audio_emb = audio_encoder(audio.to(device)) if audio is not None else None
        visual = batch.get('visual')
        visual_emb = visual_encoder(visual.to(device)) if visual is not None else None
        logits = model(text_emb, audio_emb, visual_emb)
        preds = torch.argmax(logits, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)

print(f"\nTest Results:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_acc:.4f}")
print(f"  Test F1: {test_f1:.4f}")

# Comparative analysis
print("\nRunning comparative modality analysis...")
analyzer = ComparativeAnalysis(device=device)

# Save results
results = {
    'test_accuracy': float(test_acc),
    'test_f1': float(test_f1),
    'test_loss': float(test_loss),
    'best_val_accuracy': float(best_val_acc),
    'training_history': training_history
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_path = f'results/results_cv_grade_{timestamp}.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {results_path}")
print("\nCV-Grade Architecture Training Complete!")
