#!/usr/bin/env python3
"""
Script to load the Iris dataset, train a machine learning model and save it
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def main():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Load the Iris dataset
    print("Loading Iris dataset...")
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # Print some information about the dataset
    print(f"Dataset loaded. Shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target classes: {target_names}")


if __name__ == "__main__":
    main()