import pandas as pd

from src.config import raw_data_dir, DataValidationConfig

config = DataValidationConfig()


def validate_dataset():
    print("Begin validation of data...")
    file_path = raw_data_dir/config.raw_data_file
    if not file_path.is_file():  # checking if the file exists
        raise FileNotFoundError(f"File was not found: {file_path}.")

    if file_path.stat().st_size == 0:  # checking if the file contains any data
        raise ValueError(f"File {config.raw_data_file} exists, but is completely empty.")

    try:
        df = pd.read_csv(file_path)  # checks if there are any data errors
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError("Dataset is not a valid .csv file.") from error

    if df.shape[1] != config.expected_column_count:  # checks id the column count is correct
        raise ValueError(f"The expected column count is: {config.expected_column_count} "
                         f"Found column count is: {df.shape[1]}")

    if df.shape[0] != config.expected_row_count:  # checks if the row count is correct
        raise ValueError(f"The expected row count is: {config.expected_row_count} "
                         f"Found row count is: {df.shape[0]}")

    if list(df.columns) != list(config.expected_columns):  # checks if column names are correct
        raise ValueError("Unexpected attributes found.\n"
                         f"Expected: {list(config.expected_columns)}\n"
                         f"Found: {list(df.columns)}")

    empty_cells = df.isna().sum().sum()  # checks if there are any empty cells in the file
    if empty_cells > 0:
        raise ValueError("Data is missing values.")

    class_values = set(df["Class"].unique())  # checks if there are any unexpected class labels
    if class_values != {0, 1}:
        raise ValueError("Unexpected class found.\n"
                         "Expected classes: [0, 1]\n"
                         f"Found classes: {class_values}")

    fraudulent_transaction_count = (df["Class"] == 1).sum()
    legitimate_transaction_count = (df["Class"] == 0).sum()

    if fraudulent_transaction_count != config.expected_fraudulent_count:  # checks if the number of fraudulent transactions is correct
        raise ValueError("Unexpected number of fraudulent transactions found.\n"
                         f"Expected: {config.expected_fraudulent_count}.\n"
                         f"Found: {fraudulent_transaction_count}.")

    if legitimate_transaction_count != config.expected_legitimate_count:  # checks if the number of legitimate transactions is correct
        raise ValueError("Unexpected number of legitimate transactions found.\n"
                         f"Expected: {config.expected_legitimate_count}.\n"
                         f"Found: {legitimate_transaction_count}.")

    print("Validation of the dataset is completed successfully.")

