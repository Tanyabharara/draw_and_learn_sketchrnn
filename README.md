# SketchRNN Drawing Recognition Game

An AI-powered drawing recognition system using SketchRNN with 28 categories and 86.71% accuracy.

## 🎯 Overview

This project implements a deep learning model that can recognize hand-drawn sketches across 28 different categories. The system includes both a training pipeline and a web-based interactive drawing game.

## 📊 Dataset

- **28 Categories**: cat, dog, house, tree, car, bird, fish, flower, sun, moon, apple, banana, grapes, strawberry, pizza, hamburger, hot dog, ice cream, cake, airplane, train, bicycle, helicopter, elephant, giraffe, lion, tiger, bear
- **1,000 drawings per category** (28,000 total)
- **Source**: QuickDraw dataset from Google
- **Image size**: 28x28 pixels (grayscale)

## 🏗️ Architecture

- **Model Type**: Convolutional Neural Network (CNN)
- **Parameters**: 5.9 million
- **Input**: 28x28x1 grayscale images
- **Output**: 28-class classification (softmax)
- **Accuracy**: 86.71% on test set

### Model Structure
```
Data Augmentation → 4 Conv Blocks → Global Pooling → Dense Layers → 28 Classes
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Download Dataset
```bash
python download_dataset.py
```

### Train Model
```bash
python train_smart_sketch_rnn.py
```

### Start Web Game
```bash
python start_web_game.py
```

Then open: http://localhost:5000

## 🎮 Web Interface

The web game features:
- **Interactive drawing canvas**
- **Real-time AI predictions**
- **Top 5 predictions with confidence scores**
- **28 category challenges**
- **Score tracking system**

## 📁 Project Structure

```
├── train_smart_sketch_rnn.py    # Main training script
├── web_drawing_game.py          # Flask web server
├── start_web_game.py            # Launcher script
├── download_dataset.py          # Dataset downloader
├── requirements.txt             # Dependencies
├── best_smart_sketch_rnn_model.h5  # Trained model (68MB)
├── data/
│   ├── categories.json          # 28 categories list
│   └── filtered_quickdraw_data.pkl  # Dataset (84MB)
└── templates/
    └── index.html               # Web interface
```

## 🎯 Performance

### Per-Category Accuracy (Top Performers)
- **apple**: 98.67%
- **bicycle**: 98.00%
- **ice cream**: 96.67%
- **house**: 96.00%
- **tree**: 95.33%

### Overall Performance
- **Test Accuracy**: 86.71%
- **Model Size**: 68MB
- **Inference Speed**: Real-time

## 🔧 Technical Details

### Training Configuration
- **Epochs**: 110 (with early stopping)
- **Batch Size**: 64
- **Learning Rate**: 0.0005 (with scheduling)
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy

### Data Augmentation
- Random rotation (±10%)
- Random zoom (±10%)
- Random translation (±10%)

## 🌟 Features

- ✅ **28-category recognition**
- ✅ **Web-based drawing interface**
- ✅ **Real-time predictions**
- ✅ **High accuracy (86.71%)**
- ✅ **Professional UI/UX**
- ✅ **Complete training pipeline**

## 📈 Usage Examples

1. **Training**: Train on custom categories
2. **Web Game**: Interactive drawing challenges
3. **API**: Use model for predictions
4. **Research**: Extend for new categories

## 🤝 Contributing

Feel free to contribute by:
- Adding new categories
- Improving model architecture
- Enhancing web interface
- Reporting issues

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Tanya Bharara**
- GitHub: [@Tanyabharara](https://github.com/Tanyabharara)
- Project: SketchRNN Drawing Recognition Game

---

**Ready to draw? Start the web game and challenge the AI!** 🎨✨
