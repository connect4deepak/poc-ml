import pytest
from train import classifier

def test_model_accuracy():
    assert classifier.score(X_test, y_test) > 0.9