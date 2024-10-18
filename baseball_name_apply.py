import pandas as pd
import glob
import os

# Path to the folder containing CSV files
folder_path = 'C:\\Mtest\\개인연습'

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Loop through each file
for file_path in csv_files:
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Replace "넥센" with "키움" in the entire DataFrame
    data.replace("넥센", "키움", inplace=True)
    data.replace("SK", "SSG", inplace=True)
    data.replace("울산", "사직", inplace=True)
    data.replace("청주", "대전", inplace=True)
    data.replace("마산", "창원", inplace=True)
    data.replace("포항", "대구", inplace=True)
    data.replace("목동", "고척", inplace=True)
    # Save the updated DataFrame to the same file
    data.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"Updated and saved file: {file_path}")

print("All files processed successfully.")