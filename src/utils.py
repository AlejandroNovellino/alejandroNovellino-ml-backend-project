import os
from typing import TypedDict, NotRequired
from dotenv import load_dotenv  # type: ignore
from sqlalchemy import create_engine
import pandas as pd


# load the .env file variables
load_dotenv()  #


def db_connect() -> None:
    """
    Connects to the database using sqlalchemy
    """

    engine = create_engine(os.getenv("DATABASE_URL"))  # type: ignore
    engine.connect()

    return


class DataLoadingError(Exception):
    """Custom exception raised when data loading fails."""


class ReadCsvParams(TypedDict):
    """
    Class for setting the parameters for reading a CSV file.

    Attributes:
        delimiter (str): Delimiter in the CSV file.
        nrows (Optional[int]): Number of rows of file to read.
    """

    delimiter: str
    nrows: NotRequired[int]


class SaveCsvParams(TypedDict):
    """
    Class for setting the parameters for saving a CSV file.

    Attributes:
        sep (str): Delimiter in the output CSV file.
    """

    sep: str


def load_data(
    file_path: str,
    url: str,
    read_csv_params: ReadCsvParams,
    save_csv_params: SaveCsvParams,
) -> pd.DataFrame:
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
        df = pd.read_csv(file_path, **read_csv_params)  # type: ignore

        # return the loaded dataframe form local file
        return df

    else:

        print(f"File not found. Loading data from URL: {url}")

        try:
            # file not found so try to get the data from the URL
            df = pd.read_csv(url, **read_csv_params)  # type: ignore

            # save the DataFrame to the file for future use
            df.to_csv(file_path, index=False, sep=save_csv_params["sep"])

            print(f"Data saved to file: {file_path}")

        except Exception as e:
            print(f"Error loading data from URL: {e}")
            raise DataLoadingError() from e

    return df
