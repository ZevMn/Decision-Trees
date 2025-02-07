import numpy as np
from read_data import read_dataset, display_barcharts, different_labels
from evaluation import Evaluation

if __name__ == "__main__":
    # Load datasets
    x_full, y_full, classes_full = read_dataset("data/train_full.txt")
    x_sub, y_sub, classes_sub = read_dataset("data/train_sub.txt")
    x_noisy, y_noisy, classes_noisy = read_dataset("data/train_noisy.txt")
    
    
    # Test difference visualization
    print("\nTesting difference visualization...")
    different_labels(x_full, y_full, x_noisy, y_noisy, classes_full)
    
    # Test evaluation plots
    print("\nTesting evaluation plots...")
    evaluation = Evaluation()
    # Create some dummy predictions for testing
    dummy_predictions = np.random.choice(classes_full, size=len(y_full))
    evaluation.evaluate(y_full, dummy_predictions, "Test Plot")