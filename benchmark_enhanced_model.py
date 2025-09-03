#!/usr/bin/env python3
"""
Enhanced SketchRNN Model Benchmark for 51 Categories
Tests inference time, accuracy, and performance metrics
"""

import time
import numpy as np
import pickle
import json
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from train_sketch_rnn import create_enhanced_sketch_rnn_model
import matplotlib.pyplot as plt
import seaborn as sns

def load_enhanced_model_and_data():
    """Load the enhanced trained model and test data"""
    print("Loading Enhanced SketchRNN model for 51 categories...")
    
    # Create and load enhanced model
    model = create_enhanced_sketch_rnn_model(num_classes=51)
    model.load_weights('best_enhanced_sketch_rnn_model.h5')
    print(f"Enhanced model loaded successfully. Parameters: {model.count_params():,}")
    
    # Load enhanced categories
    with open('data/categories_51.json', 'r') as f:
        categories = json.load(f)
    print(f"Enhanced categories loaded: {len(categories)} categories")
    
    # Load enhanced test data
    with open('data/enhanced_quickdraw_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Prepare test data for 51 categories
    X_test = []
    y_test = []
    
    for i, category in enumerate(categories):
        if category in data:
            category_data = data[category]
            # Take last 200 samples as test data (increased for better evaluation)
            test_samples = category_data[-200:]
            X_test.extend(test_samples)
            y_test.extend([i] * len(test_samples))
    
    X_test = np.array(X_test).reshape(-1, 28, 28, 1)
    y_test = np.array(y_test)
    
    print(f"Enhanced test data loaded: {len(X_test):,} samples")
    print(f"Test samples per category: 200")
    return model, categories, X_test, y_test

def test_enhanced_inference_time(model, test_data, num_samples=200):
    """Test how fast the enhanced model responds"""
    print(f"\nTESTING ENHANCED INFERENCE TIME")
    print("=" * 50)
    
    # Warm up the model
    print("Warming up enhanced model...")
    _ = model.predict(test_data[:20], verbose=0, batch_size=128)
    
    # Test inference time
    print(f"Testing inference time on {num_samples} samples...")
    inference_times = []
    
    for i in range(num_samples):
        # Get a random sample
        sample_idx = np.random.randint(0, len(test_data))
        sample = test_data[sample_idx:sample_idx+1]
        
        # Measure time
        start_time = time.time()
        prediction = model.predict(sample, verbose=0, batch_size=128)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        inference_times.append(inference_time)
        
        if (i + 1) % 40 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples")
    
    # Calculate statistics
    avg_time = np.mean(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    std_time = np.std(inference_times)
    
    print(f"\nENHANCED INFERENCE TIME RESULTS:")
    print(f"  Average response time: {avg_time:.2f} ± {std_time:.2f} milliseconds")
    print(f"  Fastest response:      {min_time:.2f} milliseconds")
    print(f"  Slowest response:      {max_time:.2f} milliseconds")
    print(f"  Throughput:            {1000/avg_time:.1f} images per second")
    
    return avg_time, std_time

def test_enhanced_accuracy(model, categories, X_test, y_test):
    """Test how accurate the enhanced model is"""
    print(f"\nTESTING ENHANCED ACCURACY")
    print("=" * 50)
    
    print("Getting predictions for all enhanced test data...")
    predictions = model.predict(X_test, verbose=0, batch_size=128)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Calculate overall accuracy
    correct_predictions = np.sum(predicted_labels == y_test)
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions
    
    print(f"\nENHANCED ACCURACY RESULTS:")
    print(f"  Correct predictions: {correct_predictions:,} out of {total_predictions:,}")
    print(f"  Test accuracy:       {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Wrong predictions:   {total_predictions - correct_predictions:,}")
    
    # Per-category accuracy
    print(f"\nPER-CATEGORY ACCURACY (51 categories):")
    print("-" * 60)
    
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
    
    print(f"\nTOP 10 PERFORMERS:")
    for i, (category, acc) in enumerate(sorted_categories[:10]):
        correct = int(acc * 200)  # 200 samples per category
        print(f"  {i+1:2}. {category:15}: {correct:3}/200 correct ({acc*100:6.2f}%)")
    
    print(f"\nBOTTOM 10 PERFORMERS:")
    for i, (category, acc) in enumerate(sorted_categories[-10:]):
        correct = int(acc * 200)  # 200 samples per category
        print(f"  {i+1:2}. {category:15}: {correct:3}/200 correct ({acc*100:6.2f}%)")
    
    return accuracy, category_accuracy

def analyze_model_complexity(model):
    """Analyze the complexity and efficiency of the enhanced model"""
    print(f"\nMODEL COMPLEXITY ANALYSIS")
    print("=" * 50)
    
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    print(f"  Non-trainable params:  {non_trainable_params:,}")
    
    # Calculate model size in MB
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    print(f"  Estimated model size:  {model_size_mb:.1f} MB")
    
    # Layer analysis
    print(f"\n  Layer breakdown:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'output_shape'):
            print(f"    Layer {i+1:2}: {layer.__class__.__name__:20} - {layer.output_shape}")
    
    return total_params, model_size_mb

def generate_performance_report(avg_inference_time, accuracy, category_accuracy, total_params, model_size_mb):
    """Generate a comprehensive performance report"""
    print(f"\nCOMPREHENSIVE PERFORMANCE REPORT")
    print("=" * 60)
    
    # Calculate additional metrics
    total_categories = len(category_accuracy)
    high_accuracy_categories = sum(1 for acc in category_accuracy.values() if acc >= 0.9)
    medium_accuracy_categories = sum(1 for acc in category_accuracy.values() if 0.7 <= acc < 0.9)
    low_accuracy_categories = sum(1 for acc in category_accuracy.values() if acc < 0.7)
    
    print(f"  Model Performance Summary:")
    print(f"    Overall Accuracy:           {accuracy*100:.2f}%")
    print(f"    Inference Time:             {avg_inference_time:.2f} ms")
    print(f"    Throughput:                 {1000/avg_inference_time:.1f} images/sec")
    print(f"    Total Categories:           {total_categories}")
    print(f"    High Accuracy (≥90%):       {high_accuracy_categories}")
    print(f"    Medium Accuracy (70-90%):   {medium_accuracy_categories}")
    print(f"    Low Accuracy (<70%):        {low_accuracy_categories}")
    
    print(f"\n  Model Specifications:")
    print(f"    Total Parameters:           {total_params:,}")
    print(f"    Model Size:                 {model_size_mb:.1f} MB")
    print(f"    Input Shape:                28x28x1 grayscale")
    print(f"    Output Classes:             {total_categories}")
    
    # Performance rating
    if accuracy >= 0.9:
        performance_rating = "Excellent"
    elif accuracy >= 0.8:
        performance_rating = "Good"
    elif accuracy >= 0.7:
        performance_rating = "Fair"
    else:
        performance_rating = "Needs Improvement"
    
    print(f"\n  Performance Rating:          {performance_rating}")
    
    return {
        'overall_accuracy': accuracy,
        'inference_time_ms': avg_inference_time,
        'throughput': 1000/avg_inference_time,
        'total_categories': total_categories,
        'high_accuracy_count': high_accuracy_categories,
        'medium_accuracy_count': medium_accuracy_categories,
        'low_accuracy_count': low_accuracy_categories,
        'total_parameters': total_params,
        'model_size_mb': model_size_mb,
        'performance_rating': performance_rating
    }

def main():
    """Main enhanced benchmark function"""
    print("ENHANCED SKETCHRNN MODEL BENCHMARK - 51 CATEGORIES")
    print("=" * 70)
    
    try:
        # Load enhanced model and data
        model, categories, X_test, y_test = load_enhanced_model_and_data()
        
        # Test enhanced inference time
        avg_inference_time, std_inference_time = test_enhanced_inference_time(model, X_test, num_samples=200)
        
        # Test enhanced accuracy
        overall_accuracy, category_accuracy = test_enhanced_accuracy(model, categories, X_test, y_test)
        
        # Analyze model complexity
        total_params, model_size_mb = analyze_model_complexity(model)
        
        # Generate comprehensive report
        performance_metrics = generate_performance_report(
            avg_inference_time, overall_accuracy, category_accuracy, 
            total_params, model_size_mb
        )
        
        # Save enhanced results
        enhanced_results = {
            'inference_time_ms': avg_inference_time,
            'inference_time_std_ms': std_inference_time,
            'test_accuracy': overall_accuracy,
            'correct_predictions': int(overall_accuracy * len(X_test)),
            'total_predictions': len(X_test),
            'category_accuracy': category_accuracy,
            'performance_metrics': performance_metrics
        }
        
        with open('enhanced_benchmark_results.json', 'w') as f:
            json.dump(enhanced_results, f, indent=2)
        
        print(f"\nEnhanced benchmark results saved to: enhanced_benchmark_results.json")
        
        # Final summary
        print(f"\nFINAL ENHANCED SUMMARY:")
        print("=" * 40)
        print(f"  Enhanced Model Response Time: {avg_inference_time:.2f} ± {std_inference_time:.2f} ms")
        print(f"  Enhanced Test Accuracy:       {overall_accuracy*100:.2f}%")
        print(f"  Enhanced Total Test Samples:  {len(X_test):,}")
        print(f"  Enhanced Model Parameters:    {total_params:,}")
        print(f"  Enhanced Model Size:          {model_size_mb:.1f} MB")
        
    except Exception as e:
        print(f"Enhanced benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()
