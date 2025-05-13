import pandas as pd

# Load CSV file
file_path = "help_desk_issues.csv"
df = pd.read_csv(file_path)

# Display first few rows
print("Data Sample:\n", df.head())
