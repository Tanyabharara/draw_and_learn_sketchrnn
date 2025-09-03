#!/usr/bin/env python3
"""
Enhanced QuickDraw Dataset Downloader for 51 Categories
Downloads 8,000 drawings per category for enhanced model training
"""

import os
import json
import pickle
import numpy as np
from quickdraw import QuickDrawData

def load_existing_enhanced_data():
    """Load existing enhanced dataset if available"""
    if os.path.exists('data/enhanced_quickdraw_data.pkl'):
        with open('data/enhanced_quickdraw_data.pkl', 'rb') as f:
            return pickle.load(f)
    return {}

def save_enhanced_dataset(data):
    """Save enhanced dataset to file"""
    os.makedirs('data', exist_ok=True)
    with open('data/enhanced_quickdraw_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"Enhanced dataset saved with {len(data)} categories")

def download_enhanced_category(category_name, max_drawings=8000):
    """Download drawings for a specific category with enhanced processing"""
    try:
        print(f"Downloading {category_name}...")
        qd = QuickDrawData()
        drawings = qd.get_drawing(category_name)
        
        # Convert to numpy arrays and normalize
        processed_drawings = []
        count = 0
        
        for drawing in drawings:
            if count >= max_drawings:
                break
            
            # Convert to 28x28 image
            image = drawing.get_image()
            if image is not None:
                # Resize to 28x28 and normalize
                image = image.resize((28, 28))
                image_array = np.array(image)
                image_array = image_array.astype('float32') / 255.0
                processed_drawings.append(image_array)
                count += 1
        
        print(f"  Downloaded {len(processed_drawings)} drawings")
        return processed_drawings
        
    except Exception as e:
        print(f"  Error downloading {category_name}: {e}")
        return []

def main():
    """Main enhanced download function"""
    print("Enhanced QuickDraw Dataset Downloader - 51 Categories")
    print("=" * 60)
    
    # Define enhanced categories list (51 total)
    enhanced_categories = [
        # Original 28 categories
        "cat", "dog", "house", "tree", "car", "bird", "fish", "flower", "sun", "moon",
        "apple", "banana", "grapes", "strawberry", "pizza", "hamburger", "hot dog", 
        "ice cream", "cake", "airplane", "train", "bicycle", "helicopter", 
        "elephant", "giraffe", "lion", "tiger", "bear",
        
        # Additional 23 categories for enhanced model
        "rabbit", "horse", "cow", "pig", "sheep", "duck", "chicken", "butterfly", 
        "bee", "spider", "snake", "frog", "turtle", "shark", "whale", "dolphin", 
        "octopus", "crab", "lobster", "starfish", "mountain", "river", "ocean"
    ]
    
    print(f"Enhanced categories: {len(enhanced_categories)}")
    print(f"Drawings per category: 8,000")
    print(f"Total expected drawings: {len(enhanced_categories) * 8000:,}")
    
    # Load existing enhanced data
    existing_data = load_existing_enhanced_data()
    print(f"\nExisting enhanced categories: {len(existing_data)}")
    
    # Download missing categories
    new_data = {}
    for category in enhanced_categories:
        if category not in existing_data:
            drawings = download_enhanced_category(category, max_drawings=8000)
            if drawings:
                new_data[category] = drawings
        else:
            print(f"Skipping {category} (already exists)")
    
    # Merge with existing data
    all_data = {**existing_data, **new_data}
    
    # Save complete enhanced dataset
    save_enhanced_dataset(all_data)
    
    # Save enhanced categories list
    with open('data/categories_51.json', 'w') as f:
        json.dump(enhanced_categories, f)
    
    print(f"\nEnhanced download complete!")
    print(f"Total categories: {len(all_data)}")
    print(f"Total drawings: {sum(len(drawings) for drawings in all_data.values()):,}")
    
    # Verify dataset integrity
    print(f"\nDataset verification:")
    for category in enhanced_categories:
        if category in all_data:
            count = len(all_data[category])
            print(f"  {category:15}: {count:4,} drawings")
        else:
            print(f"  {category:15}: MISSING")

if __name__ == "__main__":
    main()
