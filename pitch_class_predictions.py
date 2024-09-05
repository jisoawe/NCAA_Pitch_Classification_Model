import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance

def make_predictions(model, new_data):
    """
    Predicting the test outcomes based on the trained model
    Args:
        model: Trained model
        new_data (pd.DataFrame): New data for making predictions
    Returns: pd.Series: The predictions
    """
    return model.predict(new_data)

def load_new_data(file_path):
    """
    Loading in new data to make new predictions
    Args: file_path (str): File path for the new CSV
    Returns: pd.DataFrame: The loaded new data.
    """
    return pd.read_csv(file_path)
