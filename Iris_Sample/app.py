#!/usr/bin/env python3
"""
Web application for Iris classification using Streamlit
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import socket

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)


@st.cache_resource
def load_model_and_metadata():
    """Load the trained model and metadata (cached for performance)"""
    model_path = 'models/iris_model.joblib'
    metadata_path = 'models/iris_metadata.joblib'

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        st.error("Model or metadata files not found. Please run train_model.py first.")
        st.stop()

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    return model, metadata


def main():
    # Load model and metadata
    model, metadata = load_model_and_metadata()
    feature_names = metadata['feature_names']
    target_names = metadata['target_names']

    # Display Hostname, useful to show load balancing
    hostname = socket.gethostname()
    st.title(f"Hostname")
    st.write(f"{hostname}")
    # App title and description
    st.title("Iris Flower Classifier")
    st.write("""
    This application predicts the species of Iris flowers based on their measurements.
    Enter the measurements below to get a prediction.
    """)

    # Create two columns for layout
    # col1 = 3/5 width and col2 2/5 width
    # col1 to display sliders and col2 descriptive information
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Input Features")

        # Create input sliders for each feature
        features = []
        for feature in feature_names:
            # Clean up feature name for display
            display_name = feature.replace('(cm)', '').strip()

            # Create slider with appropriate range
            value = st.slider(
                f"{display_name} (cm)",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1
            )
            features.append(value)

        # Create a button for prediction
        predict_button = st.button("Predict Species")

        if predict_button:
            # Convert inputs to numpy array
            X = np.array([features])

            # Make prediction
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]

            # Display results
            st.subheader("Prediction Result")
            st.success(f"Predicted Species: **{target_names[prediction]}**")

            # Show probabilities as a bar chart
            prob_df = pd.DataFrame({
                'Species': target_names,
                'Probability': probabilities
            })

            fig, ax = plt.subplots(figsize=(8, 3))
            sns.barplot(x='Species', y='Probability', data=prob_df, ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title('Prediction Probabilities')
            st.pyplot(fig)

    with col2:
        st.subheader("About the Iris Dataset")
        st.write("""
        The Iris dataset is a classic dataset in machine learning and statistics.
        It contains measurements for 150 iris flowers from three different species:
        - Setosa
        - Versicolor
        - Virginica

        The features measured are:
        - Sepal length (cm)
        - Sepal width (cm)
        - Petal length (cm)
        - Petal width (cm)
        """)

        # Add iris flower images
        st.image("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png",
                 caption="Iris Species", use_container_width=True)


if __name__ == "__main__":
    main()