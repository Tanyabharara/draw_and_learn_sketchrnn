#!/usr/bin/env python3
"""
SketchRNN Web Drawing Game
Interactive web interface for drawing recognition
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import io
from PIL import Image
import json
from train_smart_sketch_rnn import create_sketch_rnn_model

app = Flask(__name__)

# Global variables
model = None
categories = []
num_classes = 28

def load_model():
    """Load the trained SketchRNN model"""
    global model, categories
    
    try:
        print("Loading SketchRNN model for 28 categories...")
        
        # Create model architecture
        model = create_sketch_rnn_model(num_classes=num_classes)
        
        # Load trained weights
        model.load_weights('best_sketch_rnn_model.h5')
        print("Model weights loaded successfully!")
        
        # Load categories
        with open('data/categories.json', 'r') as f:
            categories = json.load(f)
        
        print(f"Model info: {len(categories)} total categories")
        print(f"Available categories: {categories}")
        
        # Test model
        test_input = np.random.random((1, 28, 28, 1))
        test_output = model.predict(test_input, verbose=0)
        print(f"Model test successful - output shape: {test_output.shape}")
        
        print("Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_image(image_data):
    """Preprocess the drawn image for model input"""
    try:
        # Remove data URL prefix
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale and resize to 28x28
        image = image.convert('L')
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize
        image_array = np.array(image)
        image_array = image_array.astype('float32') / 255.0
        
        # Invert colors (white background to black)
        image_array = 1.0 - image_array
        
        # Add batch and channel dimensions
        image_array = np.expand_dims(image_array, axis=0)
        image_array = np.expand_dims(image_array, axis=-1)
        
        return image_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict the drawn image"""
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data received'}), 400
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get top predictions
        top_indices = np.argsort(predictions[0])[::-1][:5]
        top_predictions = []
        
        for idx in top_indices:
            category = categories[idx]
            confidence = float(predictions[0][idx])
            top_predictions.append({
                'category': category,
                'confidence': confidence
            })
        
        # Get predicted class
        predicted_class = categories[top_indices[0]]
        confidence = float(predictions[0][top_indices[0]])
        
        # Get all predictions for detailed analysis
        all_predictions = []
        for i, (category, conf) in enumerate(zip(categories, predictions[0])):
            all_predictions.append({
                'category': category,
                'confidence': float(conf),
                'rank': i + 1
            })
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'all_predictions': all_predictions[:10]  # Top 10
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Error processing prediction'}), 500

@app.route('/categories')
def get_categories():
    """Get available categories"""
    return jsonify({
        'categories': categories,
        'total': len(categories)
    })

if __name__ == '__main__':
    # Load model before starting server
    if load_model():
        print(f"Total categories available: {len(categories)}")
        print("Starting web server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Exiting.")
