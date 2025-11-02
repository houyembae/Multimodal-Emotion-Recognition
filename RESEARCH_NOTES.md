# Research Notes & Implementation Details

## Architecture Decisions

### Why CLIP for Visual Encoding?
- CLIP provides multimodal alignment (image-text understanding)
- Better than standard CNNs for emotion recognition
- Pretrained on 400M image-text pairs
- Transfer learning provides strong baseline

### Why Mel-Spectrograms?
- Better frequency resolution than raw waveforms
- Captures emotional prosody effectively
- Biological basis: human auditory system uses spectral analysis
- Well-studied in speech emotion recognition

### Why Ensemble Fusion?
- Combines strengths of early and late fusion
- Early fusion captures inter-modality interactions
- Late fusion allows modality-specific specialization
- Ensemble voting reduces individual modality biases

## Hyperparameter Justification

- **Batch Size 4**: GPU memory constraint (Colab T4)
- **LR 2e-5**: Standard for fine-tuning transformers
- **15 Epochs**: Validation loss plateaus; no benefit from more
- **256-dim embeddings**: Balance between expressiveness and efficiency

## Known Limitations

1. Limited dataset size (9,989 train samples)
2. Class imbalance (some emotions underrepresented)
3. GPU memory constraints limit batch size
4. Audio quality varies across videos
5. Some videos have background noise

## Performance Analysis

### Confusion Matrix Insights
- Model struggles most with: [check actual confusion matrix]
- Easiest to distinguish: [check actual confusion matrix]

### Modality Ablation
- Text only: ~XX%
- Audio only: ~XX%
- Visual only: ~XX%
- All modalities (Ensemble): 62.4%

## Recommendations for Improvement

1. **Increase batch size** when GPU memory available
2. **Implement class weighting** for underrepresented emotions
3. **Add data augmentation** (audio, visual, text)
4. **Fine-tune encoders** with careful regularization
5. **Use focal loss** for harder examples
