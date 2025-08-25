#!/usr/bin/env python3
"""
SketchRNN Web Game Launcher
Starts the web-based drawing recognition game
"""

import os
import sys

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'best_sketch_rnn_model.h5',
        'data/categories.json',
        'data/filtered_quickdraw_data.pkl',
        'web_drawing_game.py',
        'templates/index.html'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("SketchRNN Web Game Launcher")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        print("\nPlease ensure all required files are present.")
        print("You may need to train the model first using:")
        print("  python train_smart_sketch_rnn.py")
        return
    
    print("All requirements satisfied!")
    print("Starting SketchRNN web game...")
    print("\nThe game will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    # Start the web game
    try:
        from web_drawing_game import app, load_model
        
        if load_model():
            app.run(debug=True, host='0.0.0.0', port=5000)
        else:
            print("Failed to load model. Please check the model file.")
            
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    main()
