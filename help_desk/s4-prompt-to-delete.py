import pandas as pd

# Load CSV file
file_path = "help_desk_issues.csv"
df = pd.read_csv(file_path)

# Display first few rows
print("Data Sample:\n", df.head())

# Check for missing values and prompt
if df.isnull().values.any():
    print("\nMissing Values Detected:\n", df.isnull().sum())

    # Prompt user only if missing values exist
    choice = (
        input("Rows with missing data are found. Do you want to drop them? (yes/no): ")
        .strip()
        .lower()
    )

    if choice == "yes":
        df.dropna(inplace=True)
        print("\nRows with missing values have been dropped.")
    else:
        print("\nNo rows were dropped.")
else:
    print("\nNo missing values found in the DataFrame.")
