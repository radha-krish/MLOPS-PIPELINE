import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

# for log files handler 
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger=logging.getLogger(log_dir)
logger.setLevel("DEBUG")

# console handler configuration
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
# file handler configuration

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

#formating the logs
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
#adding the handleer to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url:str)-> pd.DataFrame:
    #"loading data set form url"
    try: 
        df= pd.read_csv(data_url)
        logger.debug(f"dataloaded succesfullyfrom {data_url}")
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise
def preprocess(df:pd.DataFrame):
    #preprocessing the data
    try:
        # print(df.info())
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise
data_path = "https://raw.githubusercontent.com/radha-krish/DataSets/refs/heads/main/dataset.csv"
df = load_data(data_url=data_path)
test_size=0.2
final_df = preprocess(df)
train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
save_data(train_data, test_data, data_path='./data')