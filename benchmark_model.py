#!/usr/bin/env python3
"""
Simple SketchRNN Model Benchmark
Tests inference time and accuracy
"""

import time
import numpy as np
import pickle
import json
import os
from sklearn.metrics import accuracy_score
from train_smart_sketch_rnn import create_sketch_rnn_model

def load_model_and_data():
    """Load the trained model and test data"""
    print("Loading SketchRNN model...")
    
    # Create and load model
    model = create_sketch_rnn_model(num_classes=28)
    model.load_weights('best_sketch_rnn_model.h5')
    print(f"Model loaded successfully. Parameters: {model.count_params():,}")
    
    # Load categories
    with open('data/categories.json', 'r') as f:
        categories = json.load(f)
    print(f"Categories loaded: {len(categories)} categories")
    
    # Load test data
    with open('data/quickdraw_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Prepare test data
    X_test = []
    y_test = []
    
    for i, category in enumerate(categories):
        if category in data:
            category_data = data[category]
            # Take last 150 samples as test data
            test_samples = category_data[-150:]
            X_test.extend(test_samples)
            y_test.extend([i] * len(test_samples))
    
    X_test = np.array(X_test).reshape(-1, 28, 28, 1)
    y_test = np.array(y_test)
    
    print(f"Test data loaded: {len(X_test)} samples")
    return model, categories, X_test, y_test

def test_inference_time(model, test_data, num_samples=100):
    """Test how fast the model responds"""
    print(f"\nTESTING INFERENCE TIME")
    print("=" * 40)
    
    # Warm up the model
    print("Warming up model...")
    _ = model.predict(test_data[:10], verbose=0)
    
    # Test inference time
    print(f"Testing inference time on {num_samples} samples...")
    inference_times = []
    
    for i in range(num_samples):
        # Get a random sample
        sample_idx = np.random.randint(0, len(test_data))
        sample = test_data[sample_idx:sample_idx+1]
        
        # Measure time
        start_time = time.time()
        prediction = model.predict(sample, verbose=0)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        inference_times.append(inference_time)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples")
    
    # Calculate statistics
    avg_time = np.mean(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    print(f"\nINFERENCE TIME RESULTS:")
    print(f"  Average response time: {avg_time:.2f} milliseconds")
    print(f"  Fastest response:      {min_time:.2f} milliseconds")
    print(f"  Slowest response:      {max_time:.2f} milliseconds")
    print(f"  Throughput:            {1000/avg_time:.1f} images per second")
    
    return avg_time

def test_accuracy(model, categories, X_test, y_test):
    """Test how accurate the model is"""
    print(f"\nTESTING ACCURACY")
    print("=" * 40)
    
    print("Getting predictions for all test data...")
    predictions = model.predict(X_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Calculate overall accuracy
    correct_predictions = np.sum(predicted_labels == y_test)
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions
    
    print(f"\nACCURACY RESULTS:")
    print(f"  Correct predictions: {correct_predictions:,} out of {total_predictions:,}")
    print(f"  Test accuracy:       {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Wrong predictions:   {total_predictions - correct_predictions:,}")
    
    # Per-category accuracy
    print(f"\nPER-CATEGORY ACCURACY:")
    print("-" * 40)
    
    category_accuracy = {}
    for i, category in enumerate(categories):
        # Get samples for this category
        cat_mask = (y_test == i)
        if cat_mask.sum() > 0:
            cat_correct = np.sum(predicted_labels[cat_mask] == y_test[cat_mask])
            cat_total = cat_mask.sum()
            cat_accuracy = cat_correct / cat_total
            category_accuracy[category] = cat_accuracy
            
            print(f"  {category:15}: {cat_correct:3}/{cat_total:3} correct ({cat_accuracy*100:6.2f}%)")
    
    # Top and bottom performers
    sorted_categories = sorted(category_accuracy.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTOP 5 PERFORMERS:")
    for i, (category, acc) in enumerate(sorted_categories[:5]):
        correct = int(acc * 150)  # 150 samples per category
        print(f"  {i+1}. {category:15}: {correct:3}/150 correct ({acc*100:6.2f}%)")
    
    print(f"\nBOTTOM 5 PERFORMERS:")
    for i, (category, acc) in enumerate(sorted_categories[-5:]):
        correct = int(acc * 150)  # 150 samples per category
        print(f"  {i+1}. {category:15}: {correct:3}/150 correct ({acc*100:6.2f}%)")
    
    return accuracy, category_accuracy

def main():
    """Main benchmark function"""
    print("SKETCHRNN MODEL BENCHMARK")
    print("=" * 50)
    
    try:
        # Load model and data
        model, categories, X_test, y_test = load_model_and_data()
        
        # Test inference time
        avg_inference_time = test_inference_time(model, X_test, num_samples=100)
        
        # Test accuracy
        overall_accuracy, category_accuracy = test_accuracy(model, categories, X_test, y_test)
        
        # Summary
        print(f"\nFINAL SUMMARY:")
        print("=" * 30)
        print(f"  Model Response Time: {avg_inference_time:.2f} milliseconds")
        print(f"  Test Accuracy:       {overall_accuracy*100:.2f}%")
        print(f"  Total Test Samples:  {len(X_test):,}")
        print(f"  Model Parameters:    {model.count_params():,}")
        
        # Save results
        results = {
            'inference_time_ms': avg_inference_time,
            'test_accuracy': overall_accuracy,
            'correct_predictions': int(overall_accuracy * len(X_test)),
            'total_predictions': len(X_test),
            'category_accuracy': category_accuracy
        }
        
        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: benchmark_results.json")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()
