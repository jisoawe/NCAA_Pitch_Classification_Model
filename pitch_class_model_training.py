#from pitch_classification_preprocessing.py import one_hot_encoded_data, X, y, X_train, X_test, y_train, y_test
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier

def split_data(df, target_column):
    """
    Splitting the data into training and test sets.
    Args: 
        df (pd.DataFrame): The cleaned dataset
        target_column (str): Target variable
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    y = df[target_column]
    X = df.drop(columns = [target_column])

    return train_test_split(X, y, test_size = 0.3, random_state = 22)

def train_model(X_train, y_train):
    """
    Training the model.
    Args: 
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
    Returns: The optimized model
    """
    X_train.dropna(inplace=True)
    y_train = y_train.loc[X_train.index] 
    model = DecisionTreeClassifier(criterion = 'gini',
    max_depth = 8, max_features = None, 
    min_samples_leaf = 1, min_samples_split = 10)
    
    model.fit(X_train, y_train)

    return model

