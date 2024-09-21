import pandas as pd
import numpy as np

def make_predictions(model, data):
    """
    Predicting the test outcomes based on the trained model
    Args:
        model: Trained model
        new_data (pd.DataFrame): New data for making predictions
    Returns: pd.Series: The predictions
    """
    return model.predict(data)

def load_new_data(file_path):
    """
    Loading in new data to make new predictions
    Args: file_path (str): File path for the new CSV
    Returns: pd.DataFrame: The loaded new data.
    """
    return pd.read_csv(file_path)
