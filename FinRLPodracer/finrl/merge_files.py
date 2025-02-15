import pandas as pd
import glob
import os

def merge_price_data(directory='.'):
    # Get a list of all PriceData_*.pkl files in the specified directory
    file_pattern = os.path.join(directory, 'PriceData_*.pkl')
    pkl_files = glob.glob(file_pattern)

    if not pkl_files:
        raise FileNotFoundError("No files matching 'PriceData_*.pkl' were found in the directory.")

    # List to hold individual DataFrames
    dataframes = []

    # Load each file and append its DataFrame to the list
    for file in pkl_files:
        try:
            df = pd.read_pickle(file)
            dataframes.append(df)
            print(f"Loaded file: {file}, shape: {df.shape}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    # Merge all DataFrames
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        print(f"Successfully merged {len(dataframes)} files into a single DataFrame of shape: {merged_df.shape}")
    else:
        raise ValueError("No valid DataFrames were loaded from the files.")

    return merged_df

# Example usage:
merged_data = merge_price_data()
# Display first few rows to verify
print(merged_data.head())

# Optionally, save the merged DataFrame to a new file
merged_data.to_pickle('Merged_PriceData.pkl')
