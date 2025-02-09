import os
import pandas as pd

# Folder path containing the CSV files
folder_path = "C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm"

# List to store dataframes
dataframes = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)  # Read the CSV file
        dataframes.append(df)  # Append the dataframe to the list

# Concatenate all dataframes
consolidated_df = pd.concat(dataframes, ignore_index=True)

# Save the consolidated dataframe to a new CSV file
consolidated_df.to_csv("full_book_data_file(google_books_api).csv", index=False)

print("Consolidated file created successfully!")
