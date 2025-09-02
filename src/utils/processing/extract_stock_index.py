import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="D:/Study/Education/Projects/OmniTS/.env")

data_path = os.getenv('DATA_PATH')

if data_path is None:
    raise ValueError("DATA_PATH not found. Please check your .env file.")

def extract_index(index: str) -> pd.DataFrame:
    """
    Extracts data for a specific index from a CSV file and returns it as a pandas DataFrame.

    This function reads data from a CSV file named 'raw_price_sp100.csv', filters the data for the
    specified index, and returns the result as a pandas DataFrame.

    Parameters:
    index (str): The name of the stock index to extract data for. This should be a valid index
                 name present in the given list of stock names.

    Returns:
    pd.DataFrame: A DataFrame containing the data for the specified index. An overview of the data will contain the following columns:
        - High
        - Low
        - Open
        - Close
        - Date
        - Volume
        relating to the requested stock
        
        If the index is not found, an empty DataFrame is returned.

    Raises:
    FileNotFoundError: If the 'raw_price_sp100.csv' file is not found in the current directory.

    Note:
    - The function assumes that the CSV file has a header row.
    - The function is case-sensitive when matching the index name.
    """
    try:
        df = pd.read_csv(data_path)

        stock = df.filter(regex = f"^{index}")
        
        stock.columns = stock.columns.str.split('_').str[1]
           
        return stock
    
    except FileNotFoundError:
        raise FileNotFoundError("The file 'index_data.csv' was not found in the current directory.")
    
if __name__ == "__main__":
    df = extract_index("ABBV")
    print(df.head())