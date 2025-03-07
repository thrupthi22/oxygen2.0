import pandas as pd

# Load the cleaned data
file_path = "C:/backend/archieve/cleaned_air_quality_data.csv"
df = pd.read_csv(file_path)

print("Cleaned Data Loaded Successfully!")
print(df.head())  # Verify data
