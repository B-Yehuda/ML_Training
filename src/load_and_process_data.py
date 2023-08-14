import ast
import json
import psycopg2
import pandas as pd
import numpy as np
import os
from google.cloud import storage
from google.auth import compute_engine
from datetime import datetime
import gs_chunked_io as gscio
from scipy.sparse import csr_matrix
import sqlalchemy as sa
from src.settings.redshift_credentials_settings import RedshiftCredentials


# IMPORT DATA #

def load_local_credentials(file_name):
    # navigate to parent dir
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(parent_dir):
        raise ValueError("Parent directory not found")

    # navigate to data dir
    dir_data = os.path.join(parent_dir, "data")
    if not os.path.isdir(dir_data):
        raise ValueError("Data directory not found")

    # navigate to secrets dir
    dir_secrets = os.path.join(dir_data, "secrets")
    if not os.path.isdir(dir_secrets):
        raise ValueError("Secrets directory not found")

    # navigate to file
    fn_connection = os.path.join(dir_secrets, file_name)
    if not os.path.isfile(fn_connection):
        raise ValueError("Json file not found")

    # load file
    with open(fn_connection) as config_file:
        credentials = json.load(config_file)
    assert credentials is not None

    return credentials


def connect_redshift(credentials):
    credentials = RedshiftCredentials(credentials)

    engine = sa.create_engine(sa.engine.url.URL.create(drivername="postgresql+psycopg2",
                                                       username=credentials.USER,
                                                       password=credentials.PASSWORD,
                                                       host=credentials.HOST,
                                                       port=credentials.PORT,
                                                       database=credentials.DBNAME,
                                                       )
                              ).connect()

    conn = engine.connection
    cur = conn.cursor()

    return engine, conn, cur


def load_data(cur, config):
    # retrieve locations from config file
    dataset_location = config["Dataset"].get("location")
    location_for_writing_model = config["Locations"].get("location_for_writing_model")
    filename = config["Dataset"].get("filename")

    if dataset_location == "REDSHIFT":
        print(f"Loading data from REDSHIFT - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")
        # retrieve query from config file
        query = config["Redshift_Data"].get("query")
        # execute query
        cur.execute(query)
        # load the dataset from redshift
        data = cur.fetchall()
        print(f"Loading data from REDSHIFT - finished at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")
        # frame the dataset
        df = pd.DataFrame(data)
        df.columns = [desc[0] for desc in cur.description]
        print(f"\nFraming the data - finished at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")
        # save df in pkl file
        if location_for_writing_model == "LOCAL":
            filename = config["Dataset"].get("filename")
            df.to_pickle(filename)

    elif dataset_location == "GCS" or dataset_location == "GCS_VIA_LOCAL":
        print(f"Loading data from GCS - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")
        # configure paths
        gcs_bucket = config["GCS"].get("training_bucket")
        gcs_folder_path = config["GCS"].get("folder_path_of_model_file")
        gcs_file_path = str(gcs_folder_path) + str(filename)
        # configure environment
        if dataset_location == "GCS":
            wi_credentials = compute_engine.Credentials()
            storage_client = storage.Client(credentials=wi_credentials)
        else:
            try:
                storage_client = storage.Client()
            except Exception as e:
                print("The error is: ", e)
                raise ValueError(
                    "Attempting to run the code in a local development environment - failed,"
                    " you need to run the following command (in CMD): gcloud auth application-default login")
        # access GCS bucket
        bucket = storage_client.bucket(gcs_bucket)
        # check if file exists
        is_file_exists = storage.Blob(bucket=bucket, name=gcs_file_path).exists(storage_client)
        if is_file_exists:
            blob = bucket.blob(gcs_file_path)
            # load the dataset from GCS
            with open(filename, "wb") as f:
                # read GCS object (dataset file) in chunks
                for chunk in gscio.for_each_chunk(blob, 25 * 1024 * 1024):
                    f.write(chunk)
            df = pd.read_pickle(filename)
            print(f"Loading data from GCS - finished at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")
        else:
            raise ValueError("Dataset file was not found in GCS bucket")

    elif dataset_location == "LOCAL":
        print(f"Loading data from LOCAL FILE - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")
        # load the dataset from local dir
        df = pd.read_pickle(filename)
        print(f"Loading data from LOCAL FILE - finished at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")

    else:
        raise ValueError("\033[1m No dataset location was specified in the config file \033[0m")

    return df


# DATA PROCESSING #

def reduce_memory_usage(df, int_cast=True, float_cast=False, obj_to_category=False):
    # check that the df columns are unique
    assert len(df.columns) == len(set(df.columns))

    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    :param df: dataframe to reduce (pd.DataFrame)
    :param int_cast: indicate if columns should be tried to be casted to int (bool)
    :param float_cast: indicate if columns should be tried to be casted to float (bool)
    :param obj_to_category: convert non-datetime related objects to category dtype (bool)
    :return: dataset with the column dtypes adjusted (pd.DataFrame)
    """
    cols = df.columns.tolist()

    for col in cols:
        col_type = df[col].dtypes

        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()

            # test column type
            is_int_column = col_type == np.int64
            is_float_column = col_type == np.float64

            if int_cast and is_int_column:
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype('int8')
                elif c_min >= np.iinfo(np.uint8).min and c_max <= np.iinfo(np.uint8).max:
                    df[col] = df[col].astype('uint8')
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype('int16')
                elif c_min >= np.iinfo(np.uint16).min and c_max <= np.iinfo(np.uint16).max:
                    df[col] = df[col].astype('uint16')
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype('int32')
                elif c_min >= np.iinfo(np.uint32).min and c_max <= np.iinfo(np.uint32).max:
                    df[col] = df[col].astype('uint32')
                elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                    df[col] = df[col].astype('int64')
                elif c_min >= np.iinfo(np.uint64).min and c_max <= np.iinfo(np.uint64).max:
                    df[col] = df[col].astype('uint64')
            elif float_cast and is_float_column:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype('float16')
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype('float32')
                else:
                    df[col] = df[col].astype('float64')
        elif 'datetime' not in col_type.name and obj_to_category:
            df[col] = df[col].astype('category')

    return df


def remove_outliers(df, feature: str, threshold: int, is_above):
    # find index of outliers
    if is_above:
        drop_index = set(df[df[feature] > threshold].index)
    else:
        drop_index = set(df[df[feature] < threshold].index)

    # get index of df without outliers
    new_index = list(set(df.index) - set(drop_index))

    # remove outliers
    df = df.iloc[new_index]

    # reset index
    df.reset_index(drop=True, inplace=True)

    return df


def extract_most_common_values(df, feature, top_x=None):
    # get the top x most common values of a feature
    data = df[feature].value_counts(normalize=True).head(top_x).to_frame().reset_index()
    top_x_most_common_values = list(data.iloc[:, 0])

    return top_x_most_common_values


def bucket_col_based_on_most_common_values(row, feature, most_common_values):
    # bucket feature column based on most_common_values
    if pd.isnull(row[feature]):
        return None
    elif row[feature] in most_common_values:
        return row[feature]
    else:
        return 'Other'


def add_bucked_col_to_df(df, feature, top_x):
    # create top x list that contains most common values of a feature
    most_common_values = extract_most_common_values(df, feature, top_x)
    # create function that will apply the relevant bucket for each row in the data set
    bucket_func = lambda row: bucket_col_based_on_most_common_values(row, feature, most_common_values)
    # create a new column that will contain the bucked rows
    bucket_col_name = f"{feature}_bucket"
    # apply function
    bucket_col_data = df.apply(bucket_func, axis=1)
    # add the new column to the df and collect garbage
    df = df.assign(**{bucket_col_name: bucket_col_data.values})
    del bucket_col_data

    return df


def pre_training_data_processing(df, config):
    # retrieve outliers_to_be_removed from config file
    outliers_to_be_removed = ast.literal_eval(config["Data_Processing"].get("outliers_to_be_removed"))

    # retrieve features_to_bucket from config file
    features_to_bucket = ast.literal_eval(config["Data_Processing"].get("features_to_bucket"))

    # retrieve cols_to_drop from config file
    cols_to_drop = ast.literal_eval(config["Data_Processing"].get("cols_to_drop"))

    # retrieve numeric features to convert to categorical from config file
    numeric_to_category = ast.literal_eval(config["Data_Processing"].get("numeric_to_category"))

    # remove outliers
    if outliers_to_be_removed:
        for feature, value in outliers_to_be_removed.items():
            df = remove_outliers(df, feature, value['threshold'], value['is_above'])

    # create bucked categorical features
    if features_to_bucket:
        for feature, bucket in features_to_bucket.items():
            df = add_bucked_col_to_df(df, feature, bucket)

    # drop columns
    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)

    # convert numeric features to category
    if numeric_to_category:
        for feature in numeric_to_category:
            df[feature] = df[feature].astype(np.uint8)

    return df


# EXECUTION FUNCTIONS #

def make_dummies(df, config):
    # retrieve target column
    target = ast.literal_eval(config["Data_Processing"].get("target"))

    # declare feature vector and target variable
    X = df.drop(target, axis=1)
    y = df[target]

    # create dummy variables
    X = pd.get_dummies(X)

    # # sparse the data
    # X = csr_matrix(X)

    return X, y


def data_processing(config, credentials):
    # connect redshift
    cur = connect_redshift(credentials=credentials)[2]

    # load data from redshift and reduce its memory usage
    df = reduce_memory_usage(df=load_data(cur=cur, config=config),
                             int_cast=True,
                             float_cast=False,
                             obj_to_category=False)

    # data processing
    df = pre_training_data_processing(df, config)

    # create dataset
    X, y = make_dummies(df, config)

    # delete unnecessary objects from memory
    del df

    return X, y
