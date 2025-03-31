# Iris Flower Classifier

This project demonstrates a complete machine learning workflow using the Iris dataset:
1. Training a model
2. CLI application for predictions
3. Web application for predictions using Streamlit

## Project Structure

```
Iris_Sample/
│
├── train_model.py     # Script to train and save the model
├── predict_cli.py     # CLI application for making predictions
├── app.py             # Web application built with Streamlit
├── requirements.txt   # Project dependencies
├── models/            # Directory where trained models are saved
│   ├── iris_model.joblib
│   └── iris_metadata.joblib
└── README.md          # This file
```

## Setup

1. Clone this repository:
```bash
git clone https://github.com/theurbanpenguin/ml
cd ml/Iris_Sample
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

First, train the model by running:
```bash
python train_model.py
```

This will:
- Load the Iris dataset
- Train a RandomForest classifier
- Save the model and metadata to the `models/` directory

### 2. CLI Application

To make predictions via the command line:
```bash
python predict_cli.py
```

Follow the prompts to enter the four measurements (sepal length, sepal width, petal length, petal width) in centimeters.

### 3. Web Application

To run the Streamlit web application:
```bash
streamlit run app.py
```

This will:
- Start a local web server
- Open the application in your default web browser
- Allow you to adjust the feature values using sliders
- Display the prediction with probabilities

## Features

- **train_model.py**:
  - Uses scikit-learn to train a RandomForest classifier
  - Evaluates model performance
  - Saves the model and metadata for later use

- **predict_cli.py**:
  - Provides a command-line interface for predictions
  - Handles user input validation
  - Displays prediction results with probabilities

- **app.py**:
  - Interactive web interface using Streamlit
  - Visualization of prediction probabilities
  - User-friendly sliders for feature input

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

This project is open-source and available under the Apache License.