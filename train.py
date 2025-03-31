from model import DogBreedClassifier
import os
import torch
import time
from datetime import datetime

def main():
    # Print system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Create output directory for models
    os.makedirs('models', exist_ok=True)
    
    # Initialize the classifier
    classifier = DogBreedClassifier()
    
    # Path to your training data
    train_data_path = 'data/cropped/train'
    
    # Training parameters
    num_epochs = 10
    batch_size = 64
    
    # Start training with timing
    print("\nStarting model training...")
    start_time = time.time()
    
    try:
        # Train the model
        classifier.train(train_data_path, num_epochs=num_epochs)
        
        # Calculate training time
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        
        # Save the final model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/model_{timestamp}.pth'
        classifier.save_model(model_path)
        print(f"Final model saved to: {model_path}")
        
        # Save best model if it exists
        if os.path.exists('best_model.pth'):
            best_model_path = f'models/best_model_{timestamp}.pth'
            os.rename('best_model.pth', best_model_path)
            print(f"Best model saved to: {best_model_path}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 