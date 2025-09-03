#!/usr/bin/env python3
"""
Enhanced SketchRNN Training Script for 51 Categories
Trains a SketchRNN model on 51 drawing categories with 8,000 drawings each
"""

import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import tensorflow as tf

def load_dataset():
    """Load the complete dataset for 51 categories"""
    print("Loading enhanced dataset for 51 categories...")
    
    # Load categories
    with open('data/categories_51.json', 'r') as f:
        categories = json.load(f)
    
    # Try to load the enhanced dataset first
    if os.path.exists('data/enhanced_quickdraw_data.pkl'):
        with open('data/enhanced_quickdraw_data.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded enhanced dataset with {len(categories)} categories")
    else:
        # Fallback to original dataset
        with open('data/quickdraw_data.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded original dataset with {len(categories)} categories")
    
    print(f"Categories: {categories}")
    
    return categories, data

def prepare_training_data(categories, data):
    """Prepare data for training all 51 categories with 8,000 drawings each"""
    print("\nPreparing training data for all 51 categories...")
    
    print(f"Total categories to train: {len(categories)}")
    
    # Prepare data for all categories
    X_all = []
    y_all = []
    
    for i, category in enumerate(categories):
        if category in data:
            category_data = data[category]
            # Ensure we have up to 8,000 drawings per category
            if len(category_data) > 8000:
                category_data = category_data[:8000]
            elif len(category_data) < 8000:
                print(f"Warning: {category} only has {len(category_data)} drawings (target: 8,000)")
            
            X_all.extend(category_data)
            y_all.extend([i] * len(category_data))
            print(f"  {category}: {len(category_data)} drawings")
    
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    # Reshape for CNN (add channel dimension)
    X_all = X_all.reshape(-1, 28, 28, 1)
    
    print(f"\nTraining data shape: {X_all.shape}")
    print(f"Training labels shape: {y_all.shape}")
    print(f"Total categories to train: {len(categories)}")
    
    return X_all, y_all, categories

def create_enhanced_sketch_rnn_model(num_classes=51):
    """Create an enhanced SketchRNN model for 51 categories"""
    print(f"\nCreating enhanced SketchRNN model for {num_classes} total classes...")
    
    # Enhanced model with better architecture for 51 categories and larger dataset
    model = keras.Sequential([
        # Enhanced data augmentation layers
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.15, 0.15),
        layers.RandomBrightness(0.2),
        
        # First Conv block - 28x28 -> 14x14
        layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Second Conv block - 14x14 -> 7x7
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Third Conv block - 7x7 -> 4x4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Fourth Conv block for better feature extraction
        layers.Conv2D(1024, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(1024, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Fifth Conv block for enhanced feature learning
        layers.Conv2D(1024, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(1024, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        
        # Enhanced dense layers for 51 categories
        layers.Dense(2048, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.6),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.6),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Output layer for all 51 categories
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model with adjusted learning rate for larger dataset
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Build the model with a sample input to get parameter count
    sample_input = np.random.random((1, 28, 28, 1))
    model.build(sample_input.shape)
    
    print(f"Enhanced model created with {model.count_params():,} parameters")
    return model

def train_enhanced_model(model, X_train, y_train, X_val, y_val, all_categories):
    """Train the enhanced model on all 51 categories"""
    print(f"\nStarting enhanced training on {len(all_categories)} categories...")
    
    # Enhanced callbacks for better accuracy with larger dataset
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=35,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=20,
            min_lr=1e-8,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'best_enhanced_sketch_rnn_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Enhanced learning rate scheduling for larger dataset
        callbacks.LearningRateScheduler(
            lambda epoch: 0.0003 * (0.85 ** (epoch // 25))
        ),
        # Add TensorBoard logging
        callbacks.TensorBoard(
            log_dir='logs/enhanced_train',
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    ]
    
    # Training with more epochs and adjusted batch size for larger dataset
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=128,
        callbacks=callbacks_list,
        verbose=1,
        shuffle=True
    )
    
    return history

def evaluate_enhanced_model(model, X_test, y_test, all_categories):
    """Evaluate the enhanced model on test data"""
    print(f"\nEvaluating enhanced model on {len(all_categories)} categories...")
    
    # Predictions
    y_pred = model.predict(X_test, batch_size=128)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get unique classes from test data
    unique_classes = np.unique(y_test)
    print(f"Found {len(unique_classes)} unique classes in test data")
    print(f"Classes range: {unique_classes.min()} to {unique_classes.max()}")
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report - use actual labels from test data
    print("\nClassification Report:")
    actual_categories = [all_categories[i] for i in unique_classes if i < len(all_categories)]
    print(classification_report(y_test, y_pred_classes, labels=unique_classes, target_names=actual_categories))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes, labels=unique_classes)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=actual_categories, yticklabels=actual_categories)
    plt.title('Enhanced Model Confusion Matrix - All 51 Categories')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('enhanced_sketch_rnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, y_pred_classes

def plot_enhanced_training_history(history):
    """Plot enhanced training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Enhanced Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Enhanced Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('enhanced_sketch_rnn_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main enhanced training function"""
    print("Enhanced SketchRNN Training - All 51 Categories")
    print("=" * 70)
    
    # Load dataset
    categories, data = load_dataset()
    
    # Prepare training data for all 51 categories
    X_all, y_all, all_categories = prepare_training_data(categories, data)
    
    # Split data with adjusted ratios for larger dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.25, random_state=42, stratify=y_all
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.4, random_state=42, stratify=y_temp
    )
    
    print(f"\nEnhanced data split:")
    print(f"  Training: {X_train.shape[0]:,} samples")
    print(f"  Validation: {X_val.shape[0]:,} samples")
    print(f"  Testing: {X_test.shape[0]:,} samples")
    
    # Create and train enhanced model for all 51 categories
    model = create_enhanced_sketch_rnn_model(num_classes=51)
    
    # Train enhanced model
    history = train_enhanced_model(model, X_train, y_train, X_val, y_val, all_categories)
    
    # Evaluate enhanced model
    accuracy, predictions = evaluate_enhanced_model(model, X_test, y_test, all_categories)
    
    # Plot enhanced training history
    plot_enhanced_training_history(history)
    
    print(f"\nEnhanced training complete!")
    print(f"All categories trained: {all_categories}")
    print(f"Final test accuracy: {accuracy*100:.2f}%")
    print(f"Enhanced model saved as: best_enhanced_sketch_rnn_model.h5")

if __name__ == "__main__":
    main()
