#!/usr/bin/env python3
"""
CLI script to query the trained Iris classifier model
"""

import joblib
import numpy as np
import os
import sys


def load_model_and_metadata():
    """Load the trained model and metadata"""
    model_path = 'models/iris_model.joblib'
    metadata_path = 'models/iris_metadata.joblib'

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        print("Error: Model or metadata files not found.")
        print("Please run train_model.py first to create the model.")
        sys.exit(1)

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    return model, metadata


def get_user_input(feature_names):
    """Prompt the user for input features"""
    print("Please enter the following measurements in centimeters:")
    features = []

    for feature in feature_names:
        # Make the feature name more user-friendly
        friendly_name = feature.strip()

        while True:
            try:
                value = float(input(f"{friendly_name}: "))
                features.append(value)
                break
            except ValueError:
                print("Please enter a valid number.")

    return np.array([features])


def main():
    # Load the model and metadata
    model, metadata = load_model_and_metadata()
    feature_names = metadata['feature_names']
    target_names = metadata['target_names']
    print(feature_names)
    print("Iris Flower Classifier")
    print("======================")

    while True:
        # Get input from user
        X = get_user_input(feature_names)

        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        # Display results
        print("\nResults:")
        print(f"Predicted species: {target_names[prediction]}")
        print("\nProbabilities:")
        for i, species in enumerate(target_names):
            print(f"{species}: {probabilities[i]:.2f}")

        # Ask if the user wants to make another prediction
        again = input("\nWould you like to make another prediction? (y/n): ")
        if again.lower() != 'y':
            break


if __name__ == "__main__":
    main()