# Multimodal Emotion Recognition

State-of-the-art emotion recognition system combining text (BERT), audio (mel-spectrograms), and video (CLIP) using ensemble fusion on the MELD dataset.

## Architecture

- **Text**: BERT-base → 256-dim
- **Audio**: Mel-spectrogram CNN → 256-dim  
- **Visual**: CLIP (ViT-B/32) → 256-dim
- **Fusion**: Early + Late ensemble with attention gating


## Project Structure

\`\`\`
src/
├── data/              # Dataset & preprocessing
├── models/            # BERT, CLIP, Audio encoder, Fusion
├── training/          # Training & evaluation
└── analysis/          # Comparative modality analysis
\`\`\`

## Documentation

See **PROJECT_REPORT.md** for detailed architecture, methodology, and research insights.

