import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

def evaluate_model(model, X_test, y_test):
    """
    Run statistical evaluations on the trained model after it's run on the test set
    Args:
        model (model): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
    Returns:
        dict: Metrics evaluating the model
    """
    X_test.dropna(inplace=True)
    y_test = y_test.loc[X_test.index] 

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict= True)
    c_matrix = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)

    return {'Classification Report': report,
            'Confusion Matrix': c_matrix,
            'Accuracy Score': acc_score}

