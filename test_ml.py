import os
import sys
import pytest
from ml.model import compute_model_metrics, train_model, inference
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


root_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(root_dir)

def test_metrics():
    """
    Test to ensure the model metrics are populating as expected
    """
  
    y_true, y_preds = [1, 1, 0], [0, 1, 1]
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)
    assert precision is not None
    assert recall is not None
    assert fbeta is not None
    


# TODO: implement the second test. Change the function name and input as needed
def test_pipeline():
    """
    test to ensure pipeline works correctly
    """
    # Your code here
    X = np.random.rand(20, 5)
    y = np.random.randint(2, size=20)
    model = train_model(X, y)
    assert isinstance(model, BaseEstimator) and isinstance(
        model, ClassifierMixin)
    
    


# TODO: implement the third test. Change the function name and input as needed
def test_inference():
    """
    tests the models inference ability
    """
    # Your code here
    X = np.random.rand(20, 5)
    y = np.random.randint(2, size=20)
    model = train_model(X, y)
    y_preds = inference(model, X)
    assert y.shape == y_preds.shape
    
