import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder


# To load from an actual CSV file, you would do:
df = pd.DataFrame(pd.read_csv('parkrun.csv'))
print(df.head())

# Function to convert various time formats to seconds
def time_to_seconds(time_str):
    # Check if the time string is valid
    if pd.isna(time_str) or time_str == '':
        return None

    # Split the time string by colons
    parts = time_str.split(':')

    if len(parts) == 2:  # MM:SS format
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:  # HH:MM:SS format
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    else:
        # Try to handle other potential formats or return None
        try:
            # If it's just seconds or another number format
            return float(time_str)
        except:
            print(f"Warning: Could not parse time format: {time_str}")
            return None


# Apply the conversion
df['finish_time_seconds'] = df['finish_time'].apply(time_to_seconds)

# Check the result
print("\nData with converted times:")
print(df.head())

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[['age_group', 'gender']])
feature_names = encoder.get_feature_names_out(['age_group', 'gender'])

# Create a DataFrame with encoded features
X = pd.DataFrame(encoded_features, columns=feature_names)
y = df['finish_time_seconds']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Calculate metrics
train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
#R2 is the coefficient of determination 1.- Measures how well the model's predictions approximate the actual data.

r2 = r2_score(y_test, test_predictions)

print(f"Training Mean Absolute Error: {train_mae:.2f} seconds")
print(f"Testing Mean Absolute Error: {test_mae:.2f} seconds")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)


# Make a prediction for a new runner
def predict_time(age_group, gender):
    # Create a DataFrame for the new runner
    new_runner = pd.DataFrame({'age_group': [age_group], 'gender': [gender]})

    # Encode the features
    encoded_runner = encoder.transform(new_runner)
    encoded_runner_df = pd.DataFrame(encoded_runner, columns=feature_names)

    # Make prediction
    predicted_time_seconds = model.predict(encoded_runner_df)[0]

    # Convert to minutes and seconds
    minutes = int(predicted_time_seconds // 60)
    seconds = int(predicted_time_seconds % 60)

    return predicted_time_seconds, f"{minutes}:{seconds:02d}"


# Example prediction
age_group = '60-64'
gender = 'Male'
seconds, time_format = predict_time(age_group, gender)

print(f"\nPredicted time for a {gender} runner in age group {age_group}: {time_format}")

# Visualization: Predicted vs Actual Times
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_predictions, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Time (seconds)')
plt.ylabel('Predicted Time (seconds)')
plt.title('Actual vs Predicted 5K Times')
plt.tight_layout()
plt.savefig('prediction_accuracy.png')
plt.close()

# Visualization: Average times by age group and gender
plt.figure(figsize=(12, 6))
avg_times = df.groupby(['age_group', 'gender'])['finish_time_seconds'].mean().unstack()
avg_times.plot(kind='bar')
plt.xlabel('Age Group')
plt.ylabel('Average Time (seconds)')
plt.title('Average 5K Times by Age Group and Gender')
plt.legend(title='Gender')
plt.tight_layout()
plt.savefig('avg_times_by_group.png')