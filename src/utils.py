import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd


load_dotenv()  # load the .env file variables


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return 


class DataLoadingError(Exception):
    """Custom exception raised when data loading fails."""


def load_data(file_path: str, url: str) -> pd.DataFrame:
    """
    Loads data from a file if it exists, otherwise from a URL.

    Args:
        file_path (str): The path to the file.
        url (str): The URL to load data from if the file doesn't exist.

    Returns:
        pandas.DataFrame: The loaded DataFrame.

    Raise:
        Exception: if no data could be loaded

    Examples:
        Data not saved before in local .csv file:

        >>> from utils import load_data
        >>> file_path = '../data/raw/AB_NYC_2019.csv'
        >>> url = 'https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv'
        >>> df = load_data(file_path=file_path, url=url)

        File not found. Loading data from URL: https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv
        Data saved to file: ../data/raw/AB_NYC_2019.csv

        Data have been saved before in local .csv file:

        >>> from utils import load_data
        >>> file_path = '../data/raw/AB_NYC_2019.csv'
        >>> url = 'https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv'
        >>> df = load_data(file_path=file_path, url=url)

        Loading data from file: ../data/raw/AB_NYC_2019.csv
    """

    # verify if the file exists
    if os.path.exists(file_path):

        print(f"Loading data from file: {file_path}")

        # load the data form the file
        df: pd.DataFrame = pd.read_csv(file_path) # type: ignore

        # return the loaded dataframe form local file
        return df

    else:

        print(f"File not found. Loading data from URL: {url}")

        try:
            # file not found so try to get the data from the URL
            df: pd.DataFrame = pd.read_csv(url) #type: ignore

            # save the DataFrame to the file for future use
            df.to_csv(file_path)
            print(f"Data saved to file: {file_path}")

        except Exception as e:
            print(f"Error loading data from URL: {e}")
            raise DataLoadingError() from e

    return df
