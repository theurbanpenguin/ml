import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the CSV file
file_path = "help_desk_issues.csv"
df = pd.read_csv(file_path)

# Display the first few rows to understand the structure
# print("Data Sample:\n", df.head())

# Check for missing values
# print("\nMissing Values:\n", df.isnull().sum())

# Feature and target selection
X = df["issue_description"]
y = df["resolution"]

# Split data into training and testing sets for both X (fault) and y (resolution)

# test_size = 0.2 or 20% so 20 rows out of 100 = 20 for test 80 for training
# Setting random_state=42 ensures that the split is reproducible every time you run the code.
# Any number could be used but needs to be the same to produce the same result
# 42 The meaning of life the universe and everything
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build a pipeline with TF-IDF and Logistic Regression
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)

# Evaluate the model
# print("\nAccuracy:", accuracy_score(y_test, predictions))
# print("\nClassification Report:\n", classification_report(y_test, predictions, zero_division=0))


# Example prediction for a new issue
# new_issue = ["Keys are overly noisy"]
# predicted_resolution = pipeline.predict(new_issue)
# print("\nPredicted Resolution for New Issue:", predicted_resolution[0])


while True:
    new_issue = input("\nEnter a new issue description or exit to finish: ")
    predicted_resolution = pipeline.predict([new_issue])
    print("\nPredicted Resolution for New Issue:", predicted_resolution[0])
    if new_issue == "exit":
        break
