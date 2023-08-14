import os, os.path
import sys
import pickle
import numpy as np
import configparser
import ast
from datetime import datetime
from src.load_and_process_data import data_processing, load_local_credentials
from matplotlib import pyplot as plt
from src.training_and_evaluation_pipeline.model_training import train_model
from src.evaluation_utils import model_evaluation
from src.evaluation_utils import plot_utils
from src.ml_projects_utils.writing_output_utils import configure_paths, upload_to_gcs
from sklearn.model_selection import train_test_split
from google.cloud import storage
from google.auth import compute_engine
import pandas as pd
import smogn

from src.vault import get_vault_secrets


# DATA PREPARATION FUNCTIONS #

def indices_selector(df: pd.DataFrame,
                     col: str,
                     threshold: float) -> list:
    list_of_indices = df.index[df[col] >= threshold].tolist()

    return list_of_indices


def prepare_data_for_classifier(y):
    # create copy of y
    y_clf = y.copy()

    # change target column from count to binary
    y_clf = y_clf.apply(lambda x: 1 if x > 0 else 0)
    y_clf = y_clf.astype('uint8')

    return y_clf


def add_classifier_output_as_input_to_the_regressor(X, trained_clf_model_object, reg_config):
    # retrieve target column from config file
    target = ast.literal_eval(reg_config["Data_Processing"].get("target"))

    # create copy of X
    X_reg = X.copy()

    # predict probability with the trained classifier and add it to the data set
    predict_proba_col = target + '_' + 'predicted_probability'
    X_reg[predict_proba_col] = trained_clf_model_object.predict_proba(X_reg).T[1]

    return X_reg


def prepare_data_for_regressor(X, y, reg_config, trained_clf_model_object=None):
    # retrieve objects from config file
    target = \
        ast.literal_eval(reg_config["Data_Processing"].get("target"))
    train_only_on_non_zero_target_data = \
        reg_config["Data_Processing"].get("train_only_on_non_zero_target_data")
    probability_threshold_that_filter_clf_output_passed_as_input_to_the_reg = \
        float(reg_config["Data_Processing"].get(
            "probability_threshold_that_filter_clf_output_passed_as_input_to_the_reg"))

    if trained_clf_model_object:
        # predict probabilities with the trained classifier
        X_reg = add_classifier_output_as_input_to_the_regressor(X, trained_clf_model_object, reg_config)
        # choose indices where either:
        # classifier's predictions = 1 (i.e. predicted probability>=0.5) OR all data (i.e. predicted probability>=0)
        predict_proba_col = target + '_' + 'predicted_probability'
        predictions_indices = indices_selector(df=X_reg,
                                               col=predict_proba_col,
                                               threshold=probability_threshold_that_filter_clf_output_passed_as_input_to_the_reg
                                               )
        # return data with non_zero_indices OR all_data_indices
        X_reg = X_reg.loc[predictions_indices]
        y_reg = y.loc[predictions_indices]
        # print status
        positive_target = len(y[y > 0].index)
        positive_target_caught_by_classifier = len(y_reg[y_reg.index.isin(y[y > 0].index)])
        positive_target_missed_by_classifier = positive_target - positive_target_caught_by_classifier
        total_samples_fed_into_regressor = len(X_reg)
        print(
            f"\n\033[1mStatus of the processed data (after classification) that will be fed into the regressor:\033[0m")
        print(
            f"* Positive target samples: \033[1m{positive_target}\033[0m")
        print(
            f"* Positive target samples caught by classifier: \033[1m{positive_target_caught_by_classifier}\033[0m")
        print(
            f"* Positive target samples missed by classifier (i.e. will not be fed into the regressor): \033[1m{positive_target_missed_by_classifier}\033[0m")
        print(
            f"Total samples that will be fed into the regressor: \033[1m{total_samples_fed_into_regressor}\033[0m\n")

    elif train_only_on_non_zero_target_data == "YES":
        # choose indices where target data > 0
        non_zero_target_data_indices = np.where(y > 0)[0]
        # return data with non_zero_target_data_indices
        X_reg = X.loc[non_zero_target_data_indices]
        y_reg = y.loc[non_zero_target_data_indices]

    elif train_only_on_non_zero_target_data == "NO":
        # return original dataset
        X_reg, y_reg = X.copy(), y.copy()

    else:
        raise ValueError(
            "\033[1m Process prepare_data_for_regressor failed due to config file or wrong parameter insertion\033[0m")

    return X_reg, y_reg


def split_dataset(X, y, test_size, validation_test_split_size=None, config=None):
    if config["Model_Parameters"].get("model_type") == "CLASSIFIER":
        if config["Data_Processing"].get("is_train_val_test") == "YES":
            X_train, X_rem, y_train, y_rem = train_test_split(X,
                                                              y,
                                                              test_size=test_size,
                                                              random_state=42,
                                                              stratify=y
                                                              )
            X_valid, X_test, y_valid, y_test = train_test_split(X_rem,
                                                                y_rem,
                                                                test_size=validation_test_split_size,
                                                                random_state=42,
                                                                stratify=y_rem
                                                                )
            return X_train, X_valid, X_test, y_train, y_valid, y_test
        elif config["Data_Processing"].get("is_train_val_test") == "NO":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            return X_train, X_test, y_train, y_test

    elif config["Model_Parameters"].get("model_type") == "REGRESSOR":
        if config["Data_Processing"].get("is_train_val_test") == "YES":
            X_train, X_rem, y_train, y_rem = train_test_split(X,
                                                              y,
                                                              test_size=test_size,
                                                              random_state=42,
                                                              )
            X_valid, X_test, y_valid, y_test = train_test_split(X_rem,
                                                                y_rem,
                                                                test_size=validation_test_split_size,
                                                                random_state=42,
                                                                )
            return X_train, X_valid, X_test, y_train, y_valid, y_test
        elif config["Data_Processing"].get("is_train_val_test") == "NO":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            return X_train, X_test, y_train, y_test
    else:
        raise ValueError("\033[1m An incorrect config file was selected \033[0m")


# MODEL EVALUATION  #

def plot_evaluation_graphs(trained_model,
                           X,
                           y,
                           str_title,
                           y_for_distribution,
                           X_train_shap,
                           X_reg_all=None,
                           y_reg_all=None,
                           config=None
                           ) -> dict:
    # initialize figs dictionary
    figs_dict = {}

    # data distribution graphs
    data_distribution_fig = plot_utils.plot_data_distribution(target=y_for_distribution)
    # save fig in dictionary
    figs_dict["Data_Distribution_Fig"] = data_distribution_fig

    # classifier evaluation graphs
    if config["Model_Parameters"].get("model_type") == "CLASSIFIER":
        # plot confusion matrix
        cm_fig = model_evaluation.plot_confusion_matrix(X=X,
                                                        y_clf=y,
                                                        clf_config=config,
                                                        trained_clf=trained_model,
                                                        str_title=str_title
                                                        )
        # save fig in dictionary
        figs_dict["Confusion_Matrix_Fig"] = cm_fig

    # regressor evaluation graphs
    elif config["Model_Parameters"].get("model_type") == "REGRESSOR":
        # plot regression results
        cols_features = [c for c in X.columns]
        res = model_evaluation.plot_regression_model_evaluation(model=trained_model,
                                                                data=X,
                                                                cols_features=cols_features,
                                                                col_y=y,
                                                                str_title=str_title,
                                                                alpha=0.5,
                                                                plot_improvement=True,
                                                                X_reg_all=X_reg_all,
                                                                y_reg_all=y_reg_all,
                                                                config=config
                                                                )
        # save figs in dictionary
        figs_dict["Predictions_by_Observations_Figs"] = \
            res['predictions_by_observations_fig']
        figs_dict["Residuals_by_Observations_Figs"] = \
            res['residuals_by_observations_fig']
        figs_dict["Normalized_and_Filtered_Residuals_by_Observations_Boxplot_Fig"] = \
            res['normalize_and_filtered_residuals_boxplot_fig']
        figs_dict["Normalized_Residuals_Buckets_by_Obs_Buckets_fig"] = \
            res['normalized_residuals_buckets_by_obs_buckets_fig']
        figs_dict["Improvement_Fig"] = \
            res['fig_improvement']

    else:
        raise ValueError("\033[1m An incorrect config file was selected \033[0m")

    # features importance graphs
    xgb_fig = model_evaluation.plot_xgb_features_importance(model=trained_model,
                                                            max_num_features=50
                                                            )
    shap_fig = model_evaluation.plot_shap_features_importance(model=trained_model,
                                                              X_train_shap=X_train_shap,
                                                              max_num_features=50
                                                              )
    # save fig in dictionary
    figs_dict["XGB_Fig"] = xgb_fig
    figs_dict["SHAP_Fig"] = shap_fig

    return figs_dict


def save_evaluation_graphs(fig_name, fig, trained_model_file_path, pipeline_name, pipeline):
    folder_to_save_plots = f"visualizations/{pipeline_name}/{pipeline}/"

    os.makedirs(os.path.dirname(folder_to_save_plots), exist_ok=True)

    fig_file_name = f"{trained_model_file_path[:-4]}_{fig_name}.png"
    fig_file_path = f"{folder_to_save_plots}{fig_file_name}"
    fig.savefig(fig_file_path, bbox_inches='tight')
    plt.close(fig)

    return fig_file_path


# EXECUTION FUNCTIONS #

def train_plot_save(X_for_learning,
                    y_for_learning,
                    X_for_tuning,
                    y_for_tuning,
                    X_for_evaluating,
                    y_for_evaluating,
                    y_for_distribution,
                    pipeline_name: str,
                    pipeline: str,
                    config,
                    X_reg_all=None,
                    y_reg_all=None
                    ):
    # configure paths
    gcs_bucket = config["GCS"].get("training_bucket")
    location_for_writing_model = config["Locations"].get("location_for_writing_model")

    # train model
    trained_model_object, trained_file_path = train_model(X_train=X_for_learning,
                                                          X_test=X_for_tuning,
                                                          y_train=y_for_learning,
                                                          y_test=y_for_tuning,
                                                          config=config)

    # upload model to GCS
    gcs_file_path = configure_paths(bucket=gcs_bucket,
                                    file_name=trained_file_path,
                                    config=config)

    upload_to_gcs(location_for_writing_output=location_for_writing_model,
                  gcs_bucket=gcs_bucket,
                  gcs_file_path=gcs_file_path,
                  file_name=trained_file_path,
                  drop_if_exists=False)

    # plot evaluation figs
    figs_dict = plot_evaluation_graphs(trained_model=trained_model_object,
                                       X=X_for_evaluating,
                                       y=y_for_evaluating,
                                       str_title="Testing",
                                       y_for_distribution=y_for_distribution,
                                       X_train_shap=X_for_learning,
                                       X_reg_all=X_reg_all,
                                       y_reg_all=y_reg_all,
                                       config=config)

    # upload evaluation figs to gcs
    for fig_name, fig in figs_dict.items():
        fig_file_path = save_evaluation_graphs(fig_name=fig_name,
                                               fig=fig,
                                               trained_model_file_path=trained_file_path,
                                               pipeline_name=pipeline_name,
                                               pipeline=pipeline
                                               )

        gcs_file_path = configure_paths(bucket=gcs_bucket,
                                        file_name=fig_file_path,
                                        config=config)

        upload_to_gcs(location_for_writing_output=location_for_writing_model,
                      gcs_bucket=gcs_bucket,
                      gcs_file_path=gcs_file_path,
                      file_name=fig_file_path,
                      drop_if_exists=False)

    return trained_model_object


def classifier_pipeline(clf_config):
    # retrieve pipeline from config file
    pipeline = clf_config["Pipeline"].get("pipeline")
    pipeline_name = clf_config["Pipeline"].get("pipeline_name")

    # retrieve location_for_writing_model from config file
    location_for_writing_model = clf_config["Locations"].get("location_for_writing_model")

    print("--------------------------------------------------------------------------------------------------------")
    print(
        f"Pipeline {pipeline_name} using {pipeline} - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")
    print("--------------------------------------------------------------------------------------------------------")

    # retrieve redshift credentials
    if location_for_writing_model == "GCS":
        redshift_credentials = get_vault_secrets()
    elif location_for_writing_model == "LOCAL" or location_for_writing_model == "GCS_VIA_LOCAL":
        redshift_credentials = load_local_credentials(file_name="redshift_config.json")
    else:
        raise ValueError("No location_for_writing_model was specified in the config file")

    # create dataset
    X, y = data_processing(config=clf_config, credentials=redshift_credentials)

    # prepare dataset for classifier
    y_clf = prepare_data_for_classifier(y)

    # split, train, plot, save
    if clf_config["Data_Processing"].get("is_train_val_test") == "YES":
        X_train_clf, X_valid_clf, X_test_clf, \
        y_train_clf, y_valid_clf, y_test_clf = \
            split_dataset(X=X,
                          y=y_clf,
                          test_size=0.3,
                          validation_test_split_size=0.5,
                          config=clf_config
                          )
        train_plot_save(X_for_learning=X_train_clf,
                        y_for_learning=y_train_clf,
                        X_for_tuning=X_valid_clf,
                        y_for_tuning=y_valid_clf,
                        X_for_evaluating=X_test_clf,
                        y_for_evaluating=y_test_clf,
                        y_for_distribution=y,
                        pipeline_name=pipeline_name,
                        pipeline=pipeline,
                        config=clf_config
                        )

    elif clf_config["Data_Processing"].get("is_train_val_test") == "NO":
        X_train_clf, X_test_clf, \
        y_train_clf, y_test_clf = \
            split_dataset(X=X,
                          y=y_clf,
                          test_size=0.2,
                          config=clf_config
                          )
        train_plot_save(X_for_learning=X_train_clf,
                        y_for_learning=y_train_clf,
                        X_for_tuning=X_test_clf,
                        y_for_tuning=y_test_clf,
                        X_for_evaluating=X_test_clf,
                        y_for_evaluating=y_test_clf,
                        y_for_distribution=y,
                        pipeline_name=pipeline_name,
                        pipeline=pipeline,
                        config=clf_config
                        )

    else:
        raise ValueError("Data splitting methodology was not specified in the config file")


def regressor_pipeline(reg_config):
    # retrieve pipeline from config file
    pipeline = reg_config["Pipeline"].get("pipeline")
    pipeline_name = reg_config["Pipeline"].get("pipeline_name")

    # retrieve location_for_writing_model from config file
    location_for_writing_model = reg_config["Locations"].get("location_for_writing_model")

    print("--------------------------------------------------------------------------------------------------------")
    print(
        f"Pipeline {pipeline_name} using {pipeline} - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")
    print("--------------------------------------------------------------------------------------------------------")

    # retrieve redshift credentials
    if location_for_writing_model == "GCS":
        redshift_credentials = get_vault_secrets()
    elif location_for_writing_model == "LOCAL" or location_for_writing_model == "GCS_VIA_LOCAL":
        redshift_credentials = load_local_credentials(file_name="redshift_config.json")
    else:
        raise ValueError("No location_for_writing_model was specified in the config file")

    # create dataset
    X, y = data_processing(config=reg_config, credentials=redshift_credentials)

    # prepare dataset for regressor
    X_reg, y_reg = prepare_data_for_regressor(X=X,
                                              y=y,
                                              reg_config=reg_config
                                              )

    # split, train, plot, save
    if reg_config["Data_Processing"].get("is_train_val_test") == "YES":
        X_train_reg, X_valid_reg, X_test_reg, \
        y_train_reg, y_valid_reg, y_test_reg = \
            split_dataset(X=X_reg,
                          y=y_reg,
                          test_size=0.3,
                          validation_test_split_size=0.5,
                          config=reg_config
                          )
        train_plot_save(X_for_learning=X_train_reg,
                        y_for_learning=y_train_reg,
                        X_for_tuning=X_valid_reg,
                        y_for_tuning=y_valid_reg,
                        X_for_evaluating=X_test_reg,
                        y_for_evaluating=y_test_reg,
                        y_for_distribution=y,
                        pipeline_name=pipeline_name,
                        pipeline=pipeline,
                        config=reg_config,
                        X_reg_all=X_reg,
                        y_reg_all=y_reg,
                        )

    elif reg_config["Data_Processing"].get("is_train_val_test") == "NO":
        X_train_reg, X_test_reg, \
        y_train_reg, y_test_reg = \
            split_dataset(X=X_reg,
                          y=y_reg,
                          test_size=0.2,
                          config=reg_config
                          )
        train_plot_save(X_for_learning=X_train_reg,
                        y_for_learning=y_train_reg,
                        X_for_tuning=X_test_reg,
                        y_for_tuning=y_test_reg,
                        X_for_evaluating=X_test_reg,
                        y_for_evaluating=y_test_reg,
                        y_for_distribution=y,
                        pipeline_name=pipeline_name,
                        pipeline=pipeline,
                        config=reg_config,
                        X_reg_all=X_reg,
                        y_reg_all=y_reg,
                        )

    else:
        raise ValueError("Data splitting methodology was not specified in the config file")


def classifier_and_regressor_pipeline(clf_config, reg_config):
    # retrieve pipeline (either from clf_config OR reg_config file)
    pipeline = reg_config["Pipeline"].get("pipeline")
    pipeline_name = reg_config["Pipeline"].get("pipeline_name")

    # retrieve location_for_writing_model from config file (either from clf_config OR reg_config)
    location_for_writing_model = clf_config["Locations"].get("location_for_writing_model")

    print("--------------------------------------------------------------------------------------------------------")
    print(
        f"Pipeline {pipeline_name} using {pipeline} - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")
    print("--------------------------------------------------------------------------------------------------------")

    # retrieve redshift credentials
    if location_for_writing_model == "GCS":
        redshift_credentials = get_vault_secrets()
    elif location_for_writing_model == "LOCAL" or location_for_writing_model == "GCS_VIA_LOCAL":
        redshift_credentials = load_local_credentials(file_name="redshift_config.json")
    else:
        raise ValueError("No location_for_writing_model was specified in the config file")

    # create dataset (either from clf_config OR reg_config file)
    X, y = data_processing(config=clf_config, credentials=redshift_credentials)

    # prepare dataset for classifier
    y_clf = prepare_data_for_classifier(y)

    # split, train, plot, save
    if clf_config["Data_Processing"].get("is_train_val_test") == "YES":
        X_train_clf, X_valid_clf, X_test_clf, \
        y_train_clf, y_valid_clf, y_test_clf = \
            split_dataset(X=X,
                          y=y_clf,
                          test_size=0.3,
                          validation_test_split_size=0.5,
                          config=clf_config
                          )
        trained_clf_model_object = train_plot_save(X_for_learning=X_train_clf,
                                                   y_for_learning=y_train_clf,
                                                   X_for_tuning=X_valid_clf,
                                                   y_for_tuning=y_valid_clf,
                                                   X_for_evaluating=X_test_clf,
                                                   y_for_evaluating=y_test_clf,
                                                   y_for_distribution=y,
                                                   pipeline_name=pipeline_name,
                                                   pipeline=pipeline,
                                                   config=clf_config
                                                   )

        if reg_config["Data_Processing"].get("is_train_val_test") == "YES":
            # prepare dataset for regressor (use regular split OR use clf split data to keep stratify in the reg too)
            if reg_config["Data_Processing"].get("use_clf_split_data_to_keep_stratify_in_reg") == "YES":
                X_train_reg, y_train_reg = prepare_data_for_regressor(X=X_train_clf,
                                                                      y=y.loc[y_train_clf.index],
                                                                      reg_config=reg_config,
                                                                      trained_clf_model_object=trained_clf_model_object
                                                                      )
                X_valid_reg, y_valid_reg = prepare_data_for_regressor(X=X_valid_clf,
                                                                      y=y.loc[y_valid_clf.index],
                                                                      reg_config=reg_config,
                                                                      trained_clf_model_object=trained_clf_model_object
                                                                      )
                X_test_reg, y_test_reg = prepare_data_for_regressor(X=X_test_clf,
                                                                    y=y.loc[y_test_clf.index],
                                                                    reg_config=reg_config,
                                                                    trained_clf_model_object=trained_clf_model_object
                                                                    )
                X_reg = pd.concat([X_train_reg, X_valid_reg, X_test_reg])
                y_reg = pd.concat([y_train_reg, y_valid_reg, y_test_reg])

            elif reg_config["Data_Processing"].get("use_clf_split_data_to_keep_stratify_in_reg") == "NO":
                X_reg, y_reg = prepare_data_for_regressor(X=X,
                                                          y=y,
                                                          reg_config=reg_config,
                                                          trained_clf_model_object=trained_clf_model_object
                                                          )
                X_train_reg, X_valid_reg, X_test_reg, \
                y_train_reg, y_valid_reg, y_test_reg = \
                    split_dataset(X=X_reg,
                                  y=y_reg,
                                  test_size=0.3,
                                  validation_test_split_size=0.5,
                                  config=reg_config
                                  )
            train_plot_save(X_for_learning=X_train_reg,
                            y_for_learning=y_train_reg,
                            X_for_tuning=X_valid_reg,
                            y_for_tuning=y_valid_reg,
                            X_for_evaluating=X_test_reg,
                            y_for_evaluating=y_test_reg,
                            y_for_distribution=y,
                            pipeline_name=pipeline_name,
                            pipeline=pipeline,
                            config=reg_config,
                            X_reg_all=X_reg,
                            y_reg_all=y_reg,
                            )
        else:
            raise ValueError(
                "A conflict was found between clf_config and reg_config files regarding is_train_val_test parameter")

    elif clf_config["Data_Processing"].get("is_train_val_test") == "NO":
        # split, train, plot, save
        X_train_clf, X_test_clf, \
        y_train_clf, y_test_clf = \
            split_dataset(X=X,
                          y=y_clf,
                          test_size=0.2,
                          config=clf_config
                          )
        trained_clf_model_object = train_plot_save(X_for_learning=X_train_clf,
                                                   y_for_learning=y_train_clf,
                                                   X_for_tuning=X_test_clf,
                                                   y_for_tuning=y_test_clf,
                                                   X_for_evaluating=X_test_clf,
                                                   y_for_evaluating=y_test_clf,
                                                   y_for_distribution=y,
                                                   pipeline_name=pipeline_name,
                                                   pipeline=pipeline,
                                                   config=clf_config
                                                   )

        if reg_config["Data_Processing"].get("is_train_val_test") == "NO":
            # prepare dataset for regressor (use regular split OR use clf split data to keep stratify in the reg too)
            if reg_config["Data_Processing"].get("use_clf_split_data_to_keep_stratify_in_reg") == "YES":
                X_train_reg, y_train_reg = prepare_data_for_regressor(X=X_train_clf,
                                                                      y=y.loc[y_train_clf.index],
                                                                      reg_config=reg_config,
                                                                      trained_clf_model_object=trained_clf_model_object
                                                                      )
                X_test_reg, y_test_reg = prepare_data_for_regressor(X=X_test_clf,
                                                                    y=y.loc[y_test_clf.index],
                                                                    reg_config=reg_config,
                                                                    trained_clf_model_object=trained_clf_model_object
                                                                    )
                X_reg = pd.concat([X_train_reg, X_test_reg])
                y_reg = pd.concat([y_train_reg, y_test_reg])

            elif reg_config["Data_Processing"].get("use_clf_split_data_to_keep_stratify_in_reg") == "NO":
                X_reg, y_reg = prepare_data_for_regressor(X=X,
                                                          y=y,
                                                          reg_config=reg_config,
                                                          trained_clf_model_object=trained_clf_model_object
                                                          )
                X_train_reg, X_test_reg, \
                y_train_reg, y_test_reg = \
                    split_dataset(X=X_reg,
                                  y=y_reg,
                                  test_size=0.2,
                                  config=reg_config
                                  )

            else:
                raise ValueError("Data splitting methodology for the regressor was not specified in the config file")

            train_plot_save(X_for_learning=X_train_reg,
                            y_for_learning=y_train_reg,
                            X_for_tuning=X_test_reg,
                            y_for_tuning=y_test_reg,
                            X_for_evaluating=X_test_reg,
                            y_for_evaluating=y_test_reg,
                            y_for_distribution=y,
                            pipeline_name=pipeline_name,
                            pipeline=pipeline,
                            config=reg_config,
                            X_reg_all=X_reg,
                            y_reg_all=y_reg,
                            )
        else:
            raise ValueError(
                "A conflict was found between clf_config and reg_config files regarding is_train_val_test parameter")
    else:
        raise ValueError("Data splitting methodology was not specified in the config file")


def ml_pipeline(model_types: list,
                config_objects: dict):
    for config in config_objects.values():

        if len(model_types) == 1 and model_types == ["clf"]:
            # retrieve config file
            clf_config = config[0]
            # initialize classifier pipeline
            if clf_config["Pipeline"].get("pipeline") == "CLASSIFIER":
                classifier_pipeline(clf_config)
            else:
                raise ValueError("\033[1m Pipeline object was not defined correctly in the config file \033[0m")

        elif len(model_types) == 1 and model_types == ["reg"]:
            # retrieve config file
            reg_config = config[0]
            # initialize regressor pipeline
            if reg_config["Pipeline"].get("pipeline") == "REGRESSOR":
                regressor_pipeline(reg_config)
            else:
                raise ValueError("\033[1m Pipeline object was not defined correctly in the config file \033[0m")

        elif len(model_types) == 2 and model_types == ["clf", "reg"]:
            # retrieve config files
            clf_config = config[0]
            reg_config = config[1]
            # check that the configs are the same
            if (clf_config["Pipeline"].get("pipeline") != reg_config["Pipeline"].get("pipeline")) or \
                    (clf_config["Pipeline"].get("pipeline_name") !=
                     reg_config["Pipeline"].get("pipeline_name")) \
                    or \
                    (clf_config["Locations"].get("location_for_writing_model") !=
                     reg_config["Locations"].get("location_for_writing_model")) \
                    or \
                    (clf_config["Dataset"].get("location") !=
                     reg_config["Dataset"].get("location")):
                raise ValueError(
                    "\033[1m A conflict was found between clf_config and reg_config files regarding CLASSIFIER+REGRESSOR pipeline \033[0m")
            # initialize classifier+regressor pipeline
            if reg_config["Pipeline"].get("pipeline") == "CLASSIFIER+REGRESSOR":
                classifier_and_regressor_pipeline(clf_config, reg_config)
            else:
                raise ValueError("\033[1m Pipeline object was not defined correctly in the config file \033[0m")

        else:
            raise ValueError("\033[1m models_name and model_types were not defined correctly \033[0m")
