#!/usr/bin/env python3
"""
QuickDraw Dataset Downloader
Downloads and prepares drawing data for SketchRNN training
"""

import os
import json
import pickle
import numpy as np
from quickdraw import QuickDrawData

def load_existing_data():
    """Load existing dataset if available"""
    if os.path.exists('data/quickdraw_data.pkl'):
        with open('data/quickdraw_data.pkl', 'rb') as f:
            return pickle.load(f)
    return {}

def save_dataset(data):
    """Save dataset to file"""
    os.makedirs('data', exist_ok=True)
    with open('data/quickdraw_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"Dataset saved with {len(data)} categories")

def download_category(category_name, max_drawings=1000):
    """Download drawings for a specific category"""
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
    """Main download function"""
    print("QuickDraw Dataset Downloader")
    print("=" * 40)
    
    # Define categories to download
    categories = [
        "cat", "dog", "house", "tree", "car", "bird", "fish", "flower", "sun", "moon",
        "apple", "banana", "grapes", "strawberry", "pizza", "hamburger", "hot dog", 
        "ice cream", "cake", "airplane", "train", "bicycle", "helicopter", 
        "elephant", "giraffe", "lion", "tiger", "bear"
    ]
    
    print(f"Target categories: {len(categories)}")
    print(f"Drawings per category: 1000")
    print(f"Total expected drawings: {len(categories) * 1000:,}")
    
    # Load existing data
    existing_data = load_existing_data()
    print(f"\nExisting categories: {len(existing_data)}")
    
    # Download missing categories
    new_data = {}
    for category in categories:
        if category not in existing_data:
            drawings = download_category(category, max_drawings=1000)
            if drawings:
                new_data[category] = drawings
        else:
            print(f"Skipping {category} (already exists)")
    
    # Merge with existing data
    all_data = {**existing_data, **new_data}
    
    # Save complete dataset
    save_dataset(all_data)
    
    # Save categories list
    with open('data/categories.json', 'w') as f:
        json.dump(categories, f)
    
    print(f"\nDownload complete!")
    print(f"Total categories: {len(all_data)}")
    print(f"Total drawings: {sum(len(drawings) for drawings in all_data.values()):,}")

if __name__ == "__main__":
    main()
