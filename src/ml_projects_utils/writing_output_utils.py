from datetime import datetime, timezone
from google.cloud import storage
from google.auth import compute_engine
import uuid
import logging
import boto3
from botocore.exceptions import ClientError
from sqlalchemy.engine import Connection
from src.ml_projects_utils.sql_utils import actual_create_table, read_sql_file, table_exists, drop_table_if_exists, \
    create_empty_table
from src.load_and_process_data import load_local_credentials
import pandas as pd


def write_df_to_csv_file(df: pd.DataFrame,
                         object_name):
    object_name = object_name + "_" + str(datetime.now().strftime("%Y%m%d_%H%M"))
    file_name = object_name + ".csv"
    df.to_csv(file_name, index=False)

    return object_name, file_name


def prepare_df_for_json_file(df: pd.DataFrame,
                             database_name: str,
                             collection_name: str) -> pd.DataFrame:
    df["data"] = df.apply(lambda row: row.to_dict(), axis=1)

    df["_metadata"] = df.apply(
        lambda row: {"timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                     "database_name": database_name,
                     "collection_name": collection_name,
                     "op": "i",
                     "doc_id ": str(uuid.uuid4())
                     },
        axis=1
    )

    df = df[["_metadata", "data"]]

    return df


def write_json_to_local_file(df: pd.DataFrame,
                             object_name: str):
    file_name = str(object_name) + "_" + str(datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")) + ".json"
    with open(file_name, 'w') as f:
        f.write(df.to_json(orient='records', lines=True))


def upload_to_gcs(location_for_writing_output: str,
                  gcs_bucket,
                  gcs_file_path: str,
                  file_name,
                  drop_if_exists: bool):
    if location_for_writing_output == "GCS" or location_for_writing_output == "GCS_VIA_LOCAL":
        # configure environment
        if location_for_writing_output == "GCS":
            wi_credentials = compute_engine.Credentials()
            storage_client = storage.Client(credentials=wi_credentials)
        else:
            try:
                storage_client = storage.Client()
            except Exception as e:
                print("The error is: ", e)
                raise ValueError(
                    "Attempting to run the code in a local development environment - failed."
                    " Run the following command (in CMD): gcloud auth application-default login")
        # access GCS bucket
        bucket = storage_client.bucket(gcs_bucket)
        # check if file exists
        is_file_exists = storage.Blob(bucket=bucket, name=gcs_file_path).exists(storage_client)
        if is_file_exists:
            if drop_if_exists:
                # delete file
                blob = bucket.blob(gcs_file_path)
                blob.delete()
                print(f"Dropped existing object in: \033[1m{gcs_bucket}\033[0m bucket")
                # upload compressed file
                blob = bucket.blob(gcs_file_path)
                blob.upload_from_filename(filename=file_name)  # , content_type='application/x-gzip')
                print(f"Uploaded new object to: \033[1m{gcs_bucket}\033[0m bucket")
            else:
                pass
        else:
            # upload file
            blob = bucket.blob(gcs_file_path)
            blob.upload_from_filename(filename=file_name)  # , content_type='application/x-gzip')
            print(f"Uploaded new object to: \033[1m{gcs_bucket}\033[0m bucket")

    else:
        raise ValueError("No location was specified in the config file")


def upload_to_aws(location_for_writing_output: str,
                  aws_bucket,
                  aws_file_path: str,
                  file_name,
                  drop_if_exists: bool):
    if location_for_writing_output == "AWS" or location_for_writing_output == "AWS_VIA_LOCAL":
        # configure environment
        if location_for_writing_output == "AWS":
            # TODO: Extract credentials in AWS, similar to what's done in GCS:
            wi_credentials = compute_engine.Credentials()
            storage_client = storage.Client(credentials=wi_credentials)
        else:
            try:
                aws_credentials = load_local_credentials(file_name="aws_config.json")
                s3_client = boto3.client('s3',
                                         aws_access_key_id=aws_credentials["aws_access_key_id"],
                                         aws_secret_access_key=aws_credentials["aws_secret_access_key"])
            except Exception as e:
                print("The error is: ", e)
                # TODO: Verify this is the actual command to log into AWS ("aws configure")
                raise ValueError(
                    "Attempting to run the code in a local development environment - failed."
                    " Run the following command (in CMD): aws configure")

        # check if file exists
        def check_file_exists(s3, bucket_name, file_key):
            try:
                s3.head_object(Bucket=bucket_name, Key=file_key)
                return True
            except Exception as e:
                if e.response['Error']['Code'] == '404':
                    return False
                else:
                    print(f"An error occurred: {str(e)}")
                    return False

        is_file_exists = check_file_exists(s3=s3_client, bucket_name=aws_bucket, file_key=aws_file_path)

        if is_file_exists:
            if drop_if_exists:
                # delete file
                s3_client.delete_object(Bucket=aws_bucket, Key=aws_file_path)
                print(f"- Dropped existing object in: \033[1m{aws_bucket}\033[0m bucket")
                # upload file
                s3_client.upload_file(Filename=file_name, Bucket=aws_bucket, Key=aws_file_path)
                print(f"- Uploaded new object to: \033[1m{aws_bucket}\033[0m bucket")
            else:
                pass
        else:
            # upload file
            s3_client.upload_file(Filename=file_name, Bucket=aws_bucket, Key=aws_file_path)
            print(f"- Uploaded new object to: \033[1m{aws_bucket}\033[0m bucket")

    else:
        raise ValueError("No location was specified in the config file")


def copy_from_aws_to_redshift_table(conn: Connection,
                                    cur: Connection,
                                    schema: str,
                                    table_name: str,
                                    columns_and_types_dict: dict,
                                    skip_if_exists: bool,
                                    drop_if_exists: bool,
                                    bucket_name: str,
                                    aws_file_path: str):
    print(
        f"\nCopying data from AWS \033[1m{bucket_name}\033[0m bucket to Redshift \033[1m{schema}.{table_name}\033[0m table:")

    aws_credentials = load_local_credentials(file_name="aws_config.json")

    # skip table if exists
    if skip_if_exists and table_exists(conn=conn, table_name=table_name, schema=schema):
        print(f"- Table {schema}.{table_name} already exists, skipping.")
        return

    # drop table if exists
    if drop_if_exists and table_exists(conn=conn, table_name=table_name, schema=schema):
        drop_table_if_exists(cur=cur, schema=schema, table_name=table_name)

    # create table
    create_empty_table(cur=cur,
                       schema=schema,
                       table_name=table_name,
                       columns_and_types_dict=columns_and_types_dict)

    conn.commit()

    # copy the file from the bucket into the table on Redshift
    copy_query = f"""
        COPY {schema}.{table_name}
        FROM 's3://{bucket_name}/{aws_file_path}'
        CREDENTIALS 'aws_access_key_id={aws_credentials["aws_access_key_id"]};aws_secret_access_key={aws_credentials["aws_secret_access_key"]}'
        CSV IGNOREHEADER 1
    """
    cur.execute(copy_query)
    conn.commit()
    print(f"- Copied data from \033[1m{aws_file_path}\033[0m path to \033[1m{schema}.{table_name}\033[0m table")


def configure_paths(bucket: str,
                    file_name,
                    config) -> str:
    if bucket == "streamelements-datalake3":
        gcs_folder_path = config["GCS"].get("folder_path_of_predictions_file") + \
                          "time_bucket=" + \
                          datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + "/"
        gcs_file_path = str(gcs_folder_path) + str(file_name)

        return gcs_file_path

    elif bucket == "se-etl-bucket":
        currentYear = datetime.now().year
        currentMonth = datetime.now().month
        currentDay = datetime.now().day
        currentHour = datetime.now().hour

        gcs_folder_path = "ml/" + config["GCS"].get("folder_path_of_predictions_file") + \
                          str(currentYear) + "/" + \
                          str(currentMonth) + "/" + \
                          str(currentDay) + "/" + \
                          str(currentHour) + "/"
        gcs_file_path = str(gcs_folder_path) + str(file_name)

        return gcs_file_path

    elif bucket == "streamelements-machine-learning":
        gcs_folder_path = config["GCS"].get("folder_path_of_model_file")
        gcs_file_path = str(gcs_folder_path) + str(datetime.now().strftime("%Y%m%d")) + "/" + str(file_name)

        return gcs_file_path

    elif bucket == "temp-yoav":
        aws_folder_path = config["AWS"].get("folder_path_of_predictions_file")
        aws_file_path = str(aws_folder_path) + str(datetime.now().strftime("%Y%m%d")) + "/" + str(file_name)

        return aws_file_path

    else:
        raise ValueError("Wrong bucket was specified in the config file")


def write_json_to_gcs_bucket(location_for_writing_output: str,
                             buckets: list,
                             df: pd.DataFrame,
                             object_name: str,
                             drop_if_exists: bool,
                             config):
    # configure file
    file_name = "0." + str(uuid.uuid4()) + ".json"  # .gz"
    df.to_json(file_name, orient='records', lines=True)  # , compression='gzip')

    for gcs_bucket in buckets:
        # configure paths
        gcs_file_path = configure_paths(bucket=gcs_bucket,
                                        file_name=file_name,
                                        config=config)
        # upload file to GCS
        upload_to_gcs(location_for_writing_output=location_for_writing_output,
                      gcs_bucket=gcs_bucket,
                      gcs_file_path=gcs_file_path,
                      file_name=file_name,
                      drop_if_exists=drop_if_exists)


def write_output(location_for_writing_output,
                 engine,
                 conn: Connection,
                 cur: Connection,
                 schema_name: str,
                 df: pd.DataFrame,
                 object_name: str,
                 database_name: str or None,
                 skip_if_exists: bool,
                 drop_if_exists: bool,
                 create_table_from_sql_or_df: str or None,
                 buckets: list,
                 config
                 ):
    print(f"\nWriting predictions to \033[1m{location_for_writing_output}\033[0m")

    if location_for_writing_output == "LOCAL_CSV":
        write_df_to_csv_file(df=df, object_name=object_name)

    elif location_for_writing_output == "LOCAL_JSON":
        df = prepare_df_for_json_file(df=df,
                                      database_name=database_name,
                                      collection_name=object_name)
        write_json_to_local_file(df=df,
                                 object_name=object_name)

    elif location_for_writing_output == "REDSHIFT":
        sql = read_sql_file(name=object_name) if create_table_from_sql_or_df == "sql" else None
        df = df if create_table_from_sql_or_df == "df" else None
        actual_create_table(engine=engine,
                            conn=conn,
                            cur=cur,
                            schema=schema_name,
                            table_name=object_name,
                            create_table_from_sql_or_df=create_table_from_sql_or_df,
                            sql=sql,
                            df=df,
                            skip_if_exists=skip_if_exists,
                            drop_if_exists=drop_if_exists
                            )

    elif location_for_writing_output == "GCS" or location_for_writing_output == "GCS_VIA_LOCAL":
        df = prepare_df_for_json_file(df=df,
                                      database_name=database_name,
                                      collection_name=object_name)
        write_json_to_gcs_bucket(location_for_writing_output=location_for_writing_output,
                                 buckets=buckets,
                                 df=df,
                                 object_name=object_name,
                                 drop_if_exists=drop_if_exists,
                                 config=config
                                 )

    elif location_for_writing_output == "AWS" or location_for_writing_output == "AWS_VIA_LOCAL":
        _, file_name = write_df_to_csv_file(df=df, object_name=object_name)
        aws_file_path = configure_paths(bucket=buckets[0], file_name=file_name, config=config)
        upload_to_aws(location_for_writing_output=location_for_writing_output,
                      aws_bucket=buckets[0],
                      aws_file_path=aws_file_path,
                      file_name=file_name,
                      drop_if_exists=drop_if_exists)
        return aws_file_path

    else:
        raise ValueError("No location_for_writing_output was specified in the config file")
