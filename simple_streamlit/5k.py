import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Sample data - in a real implementation, you'd load this from a CSV file
# Format: age_group, gender, finish_time_seconds
data = {
    'age_group': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+',
                  '18-24', '25-34', '35-44', '45-54', '55-64', '65+',
                  '18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
    'gender': ['M', 'M', 'M', 'M', 'M', 'M',
               'F', 'F', 'F', 'F', 'F', 'F',
               'M', 'F', 'M', 'F', 'M', 'F'],
    'finish_time_seconds': [1140, 1200, 1260, 1320, 1380, 1500,
                            1260, 1320, 1380, 1440, 1560, 1680,
                            1080, 1200, 1290, 1410, 1440, 1620]
}

# Create DataFrame
df = pd.DataFrame(data)

# To load from an actual CSV file, you would do:
# df = pd.DataFrame(pd.read_csv('parkrun_results.csv'))

# Convert finish time to seconds if it's in HH:MM:SS format
# df['finish_time_seconds'] = pd.to_timedelta(df['finish_time']).dt.total_seconds()

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
age_group = '35-44'
gender = 'F'
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