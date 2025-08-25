# SketchRNN QuickDraw Drawing Recognition Model

A deep learning model that can recognize hand-drawn images using the QuickDraw dataset. This project implements a **SketchRNN-inspired architecture** optimized for sketch recognition to classify drawings into 10 categories.

## Project Overview

This project demonstrates how to:
- Download and preprocess the QuickDraw dataset (10 categories)
- Train a SketchRNN-inspired model on drawing recognition
- Create an interactive drawing interface for real-time predictions and testing
- Achieve high accuracy on sketch recognition using CNN-based architecture

## Architecture Choice

This project uses a **SketchRNN-inspired CNN architecture** specifically designed for sketch recognition:

- **Input**: 28×28 grayscale drawings (QuickDraw format)
- **Architecture**: CNN with residual connections and global pooling
- **Output**: 10-class classification with softmax probabilities
- **Parameters**: ~500K trainable parameters (lightweight and efficient)
- **Regularization**: Dropout and Batch Normalization

## Project Structure

```
quick_draw_new_model/
├── requirements.txt              # Python dependencies
├── download_dataset.py           # Script to download QuickDraw data (10 categories)
├── sketch_rnn_model.py          # SketchRNN-inspired model architecture
├── train_sketch_rnn.py          # Training script for SketchRNN
├── sketch_rnn_interface.py      # Interactive drawing app with testing
├── test_sketch_rnn.py           # Test suite for SketchRNN
├── README.md                    # This file
├── data/                        # Dataset directory (created automatically)
│   ├── quickdraw_data.pkl       # Downloaded drawings
│   └── categories.json          # Category names
├── best_sketch_rnn_model.h5     # Best model weights (created after training)
├── sketch_rnn_model.h5          # Final trained model
└── sketch_rnn_training_history.png # Training plots
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python download_dataset.py
```

This will download exactly 1000 drawings for each of the 10 categories:
- Cat
- Dog  
- House
- Tree
- Car
- Bird
- Fish
- Flower
- Sun
- Moon

**Note**: The dataset is balanced with exactly 1000 drawings per category to prevent overfitting and ensure fair training.

### 3. Train the SketchRNN Model

```bash
python train_sketch_rnn.py
```

The training script will:
- Load and preprocess the QuickDraw dataset
- Verify balanced dataset (1000 drawings per category)
- Split data into training (80%) and validation (20%) sets
- Train the SketchRNN model for 50 epochs with early stopping
- Save the best model as `best_sketch_rnn_model.h5`
- Generate training history plots

### 4. Test with Interactive Drawing Interface

```bash
python sketch_rnn_interface.py
```

This opens an enhanced GUI where you can:
- **Draw images** using your mouse on a 300×300 canvas
- **Get real-time predictions** from the trained SketchRNN model
- **See confidence scores** and top 3 predictions
- **Save drawings** as PNG files
- **Test multiple drawings** without restarting

## Technical Details

### Model Architecture

- **Base**: CNN with SketchRNN-inspired design
- **Input**: 28×28 grayscale images (normalized to [0,1])
- **Convolutional Layers**: 3 Conv2D layers (32, 64, 128 filters)
- **Output**: 10-class classification with softmax probabilities
- **Parameters**: ~500K trainable parameters
- **Regularization**: Dropout (0.5, 0.3) and Batch Normalization

### Training Configuration

- **Framework**: TensorFlow/Keras
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Sparse Categorical Crossentropy
- **Callbacks**: Early Stopping, Learning Rate Reduction, Model Checkpoint
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Data Augmentation**: None (to preserve sketch characteristics)

### Dataset Preprocessing

- **Size**: 28×28 pixels (original QuickDraw format)
- **Normalization**: Pixel values scaled to [0,1]
- **Format**: Grayscale with white background, black drawings
- **Split**: 80% training, 20% validation (stratified)
- **Total**: 10,000 drawings (10 categories × 1000 each)
- **Balance**: Exactly 1000 drawings per category to prevent overfitting

## Expected Performance

With the SketchRNN architecture on this balanced dataset, you can expect:
- **Training Time**: 15-45 minutes (depending on hardware)
- **Validation Accuracy**: 90-98% after training
- **Overfitting**: Minimal due to balanced dataset and dropout
- **Generalization**: Excellent performance on sketch recognition

## Enhanced Drawing Interface Features

The interactive drawing app provides:
- **300×300 pixel canvas** for comfortable drawing
- **Real-time prediction** with confidence scores
- **Top 3 predictions** display for better understanding
- **Color-coded results** (green: high confidence, red: low confidence)
- **Save drawing functionality** for later use
- **Model information panel** showing status and parameters
- **Clear canvas** functionality for multiple drawings
- **User-friendly interface** with clear instructions

## Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Ensure you've run `python train_sketch_rnn.py` first
   - Check that `sketch_rnn_model.h5` exists in the project directory

2. **CUDA/GPU issues**
   - TensorFlow will automatically use available GPUs
   - For CPU-only training, modify device selection in training script

3. **Download failures**
   - Check internet connection
   - Verify QuickDraw dataset availability
   - Try running `download_dataset.py` again

4. **Poor prediction accuracy**
   - Ensure training completed successfully
   - Check training plots for overfitting
   - Try drawing more clearly in the interface

### Performance Tips

- **GPU Training**: Use CUDA if available for faster training
- **Data Quality**: Ensure clean dataset downloads
- **Model Saving**: Best model is automatically saved during training
- **Regularization**: Dropout helps prevent overfitting

## Testing and Validation

### Run Tests

```bash
python test_sketch_rnn.py
```

This will verify:
- Model architecture functionality
- Input/output handling
- Categories loading
- Dataset loading and balance verification
- Model save/load operations

### Test Drawing Interface

After training, you can:
1. Draw various objects from the 10 categories
2. Test prediction accuracy
3. Save interesting drawings
4. Compare model confidence across different drawing styles

## Experimentation

Feel free to experiment with:
- **Different categories**: Modify the categories list in `download_dataset.py`
- **Model architectures**: Adjust CNN layers and parameters
- **Hyperparameters**: Modify learning rate, batch size, or epochs
- **Data augmentation**: Add rotation, scaling, or noise for robustness

## References

- **QuickDraw Dataset**: [Google AI Blog](https://ai.googleblog.com/2017/08/launching-quick-draw.html)
- **SketchRNN**: [A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477)
- **TensorFlow**: [Official Documentation](https://www.tensorflow.org/)

## Contributing

Contributions are welcome! Areas for improvement:
- Additional model architectures
- Data augmentation techniques
- Better drawing interface
- Model ensemble methods
- Web-based deployment

## License

This project is open source and available under the MIT License.

---

**Happy Drawing and Testing with SketchRNN!**
