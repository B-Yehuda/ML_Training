import os
import ast
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
from src.evaluation_utils.loss_functions import AveragePrecisionScore, F1Score, RecallScore, PrecisionScore, \
    LogLoss, RMSE, D2TweedieScore, R2, MAE, GammaScore, MeanTweedieScore
import optuna
from optuna.samplers import TPESampler
from datetime import datetime
from sklearn.model_selection import StratifiedKFold


# MODELS EVALUATION #

def model_performance(eval_model, model_object, X_test, y_test):
    # predict with the model
    y_pred = model_object.predict(X_test)

    if eval_model == xgb.XGBRegressor:
        # regression model scoring
        rmse = RMSE(squared=False, direction=None).score(y_test=y_test, y_pred=y_pred)
        r2 = R2(direction=None).score(y_test=y_test, y_pred=y_pred)
        mae = MAE(direction=None).score(y_test=y_test, y_pred=y_pred)
        # regression scores
        regression_res = {'RMSE': rmse, 'R2': r2, 'MAE': mae}

        return regression_res

    elif eval_model == xgb.XGBClassifier:
        # classification model scoring
        precision = PrecisionScore(beta_value=0.0, direction=None).score(y_test=y_test, y_pred=y_pred)
        recall = RecallScore(beta_value=np.inf, direction=None).score(y_test=y_test, y_pred=y_pred)
        fb = F1Score(beta_value=2.0, direction=None).score(y_test=y_test, y_pred=y_pred)
        # classification scores
        classification_res = {'Precision': precision, 'Recall': recall, 'F_beta': fb}

        return classification_res


# HYPERPARAMETERS OPTIMIZATION #

def custom_objective(trial, eval_model, is_cross_validation, param, score_func, score_name, X_train, y_train, X_test, y_test):
    # create objective function which evaluate model performance based on the hyperparameters combination

    # hyperparameters to be tuned
    if eval_model == xgb.XGBRegressor:
        hyperparameters_candidates = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5),
            # determines how fast the model learns
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            # maximum depth of a tree
            "min_child_weight": trial.suggest_float("min_child_weight", 10, 1000),
            # minimum sum of weights of all observations required in a child
            "subsample": trial.suggest_float("subsample", 0.5, 1),
            # the fraction of observations to be randomly samples for each tree
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1),
            # the subsample ratio of columns when constructing each tree
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0),
            # L1 regularization term on weights (analogous to Lasso regression)
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0),
            # L2 regularization term on weights (analogous to Ridge regression)
            "gamma": trial.suggest_float("lambda", 1e-8, 10.0),
            # the minimum loss reduction required to make a split
        }
    elif eval_model == xgb.XGBClassifier:
        hyperparameters_candidates = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5),
            # determines how fast the model learns
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            # maximum depth of a tree
            "min_child_weight": trial.suggest_float("min_child_weight", 10, 1000),
            # minimum sum of weights of all observations required in a child
            "scale_pos_weight": trial.suggest_int('scale_pos_weight', 1, 100),
            # controls the balance of positive and negative weights
            "subsample": trial.suggest_float("subsample", 0.5, 1),
            # the fraction of observations to be randomly samples for each tree
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1),
            # the subsample ratio of columns when constructing each tree
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0),
            # L1 regularization term on weights (analogous to Lasso regression)
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0),
            # L2 regularization term on weights (analogous to Ridge regression)
            "gamma": trial.suggest_float("lambda", 1e-8, 10.0)
            # the minimum loss reduction required to make a split
        }
    else:
        raise ValueError(
            "\033[1m eval_model parameter was not defined correctly\033[0m")

    # Add a callback for pruning (ensure unpromising trials are stopped early)
    evaluation_metric = param["eval_metric"]
    pruning_validation_func = f"validation_0-{evaluation_metric}"
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, pruning_validation_func)

    if is_cross_validation == "YES" and eval_model == xgb.XGBClassifier:
        # TODO:
        #  is_cross_validation param - is a temporary solution for cross validation process.
        #  Note: To be fixed after optuna will fix the issue of enabling pruning_callback in folds.

        # instantiate the model
        xgb_optuna = eval_model(**param,
                                **hyperparameters_candidates
                                )

        # combine back train and test data
        X = pd.concat([X_train, X_test], axis=0)
        y = pd.concat([y_train, y_test], axis=0)

        # create stratified k folds for cross validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # create scores array for cross validation
        cv_scores = []

        for train_index, test_index in skf.split(X, y):
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

            # train the model
            print(f"XGBoost fit - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")
            xgb_optuna.fit(X_train_fold,
                           y_train_fold,
                           verbose=False,
                           eval_set=[(X_test_fold, y_test_fold)])
            print(f"XGBoost fit - finished at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")

            # predict with the model
            y_pred_fold = xgb_optuna.predict(X_test_fold)

            # score the model
            score = score_func(y_test=y_test_fold, y_pred=y_pred_fold)

            # store the scores of all folds
            cv_scores.append(score)

        # score the model
        score = np.mean(cv_scores)

    else:
        # TODO:
        #  IF train with log transformer THEN:
        #  xgb_optuna = TransformedTargetRegressor(regressor=pre_transform_xgb_optuna, func=np.log, inverse_func=np.exp)
        #  Note: Can try also - func=np.log1p, inverse_func=np.expm1

        # instantiate the model
        xgb_optuna = eval_model(**param,
                                **hyperparameters_candidates,
                                callbacks=[pruning_callback]
                                )

        # train the model
        print(f"XGBoost fit - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")
        xgb_optuna.fit(X_train,
                       y_train,
                       verbose=False,
                       eval_set=[(X_test, y_test)])
        print(f"XGBoost fit - finished at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")

        # predict with the model
        y_pred = xgb_optuna.predict(X_test)

        # score the model
        score = score_func(y_test=y_test, y_pred=y_pred)

        # TODO:
        #  Save best iteration model (i.e. "x.pkl" instead of "x+100.pkl").
        #  best_iteration = xgb_optuna.get_booster().best_ntree_limit
        #  print(best_iteration)

        # TODO:
        #  IF train with log transformer THEN:
        #  best_iteration = xgb_optuna.regressor_.get_booster().best_ntree_limit
        #  print(best_iteration)
        #  y_pred = xgb_optuna.predict(X_test, ntree_limit=best_iteration)

    # for each trial - create naming convention
    model_name = "Regressor_Model" if eval_model == xgb.XGBRegressor else "Classifier_Model"

    # tag each trial - with its naming convention
    trial.set_user_attr(key="model_name", value=f"{model_name}_{score_name}")

    # for each model object - create a path based on the trial tag
    model_file_path = f"{trial.user_attrs['model_name']}_{trial.number}.pkl"

    # open (and close) a file where we store each model object
    with open(model_file_path, 'wb') as f:
        # dump best model to the file
        pickle.dump(xgb_optuna, f)

    return score


def callback(study, trial):
    # create a callback function to delete all models besides the best one
    file_prefix = f"{trial.user_attrs.get('model_name', -1)}"
    best_model_file_path = f"{file_prefix}_{study.best_trial.number}.pkl"
    model_files_to_delete = [f for f in os.listdir('.') if
                             os.path.isfile(f) and f.startswith(file_prefix) and f != best_model_file_path]
    for f in model_files_to_delete:
        os.remove(f)
    print(
        f"Optuna {trial.user_attrs.get('model_name', -1)} trial number {trial.number} - finished at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")


# MODEL CREATION FUNCTIONS #

def tune_models_hyperparams(eval_model, X_train, y_train, X_test, y_test, config):
    # retrieve param from config file
    param = ast.literal_eval(config["Model_Parameters"]["param"])
    param["missing"] = np.nan  # whenever a null value is encountered it is treated as missing value
    param["objective"] = config["Model_Parameters"]["objective"]
    param["eval_metric"] = config["Model_Parameters"]["eval_metric"]

    # retrieve n_trials from config file
    n_trials = int(config["Model_Parameters"]["n_trials"])

    # retrieve is_cross_validation from config file
    is_cross_validation = config["Model_Parameters"]["is_cross_validation"]

    # dictionary for saving the scoring_functions
    scoring_functions = {}

    # dictionary for saving the best models
    grid = {}

    # define scoring method
    if eval_model == xgb.XGBRegressor:
        # retrieve Scoring_Functions values from config file
        regressor_scoring_functions = ast.literal_eval(config["Scoring_Functions"]["regressor_scoring_functions"])
        for reg_score_obj in regressor_scoring_functions:
            if reg_score_obj == "AveragePrecisionScore":
                scoring_functions[reg_score_obj] = AveragePrecisionScore(direction="maximize")
            elif reg_score_obj == "LogLoss":
                scoring_functions[reg_score_obj] = LogLoss(direction="minimize")
            elif reg_score_obj == "RMSE":
                scoring_functions[reg_score_obj] = RMSE(squared=False,
                                                        direction="minimize"
                                                        )
            elif reg_score_obj == "GammaScore":
                scoring_functions[reg_score_obj] = GammaScore(direction="minimize")
            elif reg_score_obj == "TweedieScore":
                param["tweedie_variance_power"] = float(config["Model_Parameters"]["tweedie_variance_power"])
                param["eval_metric"] = f'{param["eval_metric"]}{param["tweedie_variance_power"]}'
                if config["Model_Parameters"]["tweedie_loss_function"] == "D2TweedieScore":
                    scoring_functions[reg_score_obj] = D2TweedieScore(power=param["tweedie_variance_power"],
                                                                      direction="maximize"
                                                                      )
                elif config["Model_Parameters"]["tweedie_loss_function"] == "MeanTweedieScore":
                    scoring_functions[reg_score_obj] = MeanTweedieScore(power=param["tweedie_variance_power"],
                                                                        direction="minimize"
                                                                        )
                else:
                    raise ValueError(
                        "\033[1m tweedie_loss_function object was not defined correctly in the config file \033[0m")
            else:
                raise ValueError(
                    "\033[1m regressor_scoring_functions object was not defined correctly in the config file \033[0m")

    elif eval_model == xgb.XGBClassifier:
        # retrieve Scoring_Functions values from config file
        classifier_scoring_functions = ast.literal_eval(config["Scoring_Functions"]["classifier_scoring_functions"])
        for clf_score_obj in classifier_scoring_functions:
            if clf_score_obj == "AveragePrecisionScore":
                scoring_functions[clf_score_obj] = AveragePrecisionScore(direction="maximize")
            elif clf_score_obj == "F1Score":
                scoring_functions[clf_score_obj] = F1Score(beta_value=1.0,
                                                           direction="maximize"
                                                           )
            elif clf_score_obj == "RecallScore":
                scoring_functions[clf_score_obj] = RecallScore(beta_value=2.0,
                                                               # beta_value=inf considers only recall
                                                               direction="maximize"
                                                               )
            elif clf_score_obj == "PrecisionScore":
                scoring_functions[clf_score_obj] = PrecisionScore(beta_value=0.5,
                                                                  # beta_value=0 considers only precision
                                                                  direction="maximize"
                                                                  )
            else:
                raise ValueError(
                    "\033[1m classifier_scoring_functions object was not defined correctly in the config file \033[0m")

    # create a sampler object to find more efficiently the best hyperparameters
    sampler = TPESampler()  # by default the sampler = TPESampler()

    for score_obj in scoring_functions.values():
        # create a study object (to set the direction of optimization and the sampler)
        study = optuna.create_study(sampler=sampler,
                                    pruner=optuna.pruners.HyperbandPruner(),
                                    direction=score_obj.direction,
                                    storage="sqlite:///study_state.db"
                                    )

        # create objective function (to calculate for each hyperparameter's combinations its score)
        objective = lambda trial: custom_objective(trial=trial,
                                                   eval_model=eval_model,
                                                   is_cross_validation=is_cross_validation,
                                                   param=param,
                                                   score_func=score_obj.score,
                                                   score_name=score_obj.name,
                                                   X_train=X_train,
                                                   y_train=y_train,
                                                   X_test=X_test,
                                                   y_test=y_test)

        # run the study object
        study.optimize(func=objective,
                       n_trials=n_trials,  # try hyperparameters combinations n_trials times
                       callbacks=[callback],  # callback delete all models but the best one
                       gc_after_trial=True)  # garbage collector

        # print study hyperparameters by importance
        print(
            f"\n\033[1mStudy hyperparameters by importance:\033[0m \n{optuna.importance.get_param_importances(study)}\n")

        # store best model name
        model_name = "Regressor_Model_" + score_obj.name \
            if eval_model == xgb.XGBRegressor \
            else "Classifier_Model_" + score_obj.name
        grid[model_name] = {}

        # store best model path
        model_file_path = f"{model_name}_{study.best_trial.number}.pkl"
        grid[model_name]["model_file_path"] = model_file_path

        # store best model object
        with open(model_file_path, 'rb') as f:
            model_object = pickle.load(f)
        grid[model_name]["model_object"] = model_object

        # store best model score
        model_scores = model_performance(eval_model, model_object, X_test, y_test)
        grid[model_name]["model_scores"] = model_scores

    return grid


def get_best_models(eval_model, grid, config):
    # dictionary for saving the filtered best models
    best_models_grid = {}

    if eval_model == xgb.XGBRegressor:
        # retrieve Scoring_Functions filters from config file
        r2_filter = float(config["Scoring_Functions"]["r2_filter"])
        # filter out models with negative R2 (models worse than a constant function that predicts the mean of the data)
        filtered_models_grid = {model_name: model_data for model_name, model_data in grid.items()
                                if model_data["model_scores"]['R2'] > r2_filter}

        # return best models and delete the rest
        if filtered_models_grid:
            min_rmse_model = min(filtered_models_grid.values(), key=lambda x: x["model_scores"]['RMSE'])
            min_rmse_value = min_rmse_model["model_scores"]['RMSE']
            best_models_grid = {model_name: model_data for model_name, model_data in filtered_models_grid.items()
                                if model_data["model_scores"]['RMSE'] == min_rmse_value}

    elif eval_model == xgb.XGBClassifier:
        # retrieve Scoring_Functions filters from config file
        precision_filter = float(config["Scoring_Functions"]["precision_filter"])
        # filter out models with precision<10%
        best_models_grid = {model_name: model_data for model_name, model_data in grid.items()
                            if model_data["model_scores"]['Precision'] > precision_filter}

    # return best models
    return best_models_grid


def print_best_grid_results(best_model_name, best_model_data, config):
    # retrieve pipeline_name from config file
    pipeline_name = config["Pipeline"]["pipeline_name"]
    # print models results
    print(f"\033[1m{'{:-^70}'.format(' [' + pipeline_name + '_' + best_model_name + '] ')}\033[0m")
    print(best_model_data["model_scores"])


def generate_model_file_name(best_model_name, config):
    # retrieve pipeline_name from config file
    pipeline_name = config["Pipeline"]["pipeline_name"]
    # generate new model file name
    new_model_file_path = str(pipeline_name) + "_" + str(best_model_name) + "_" + str(
        datetime.now().strftime("%Y%m%d_%H%M")) + ".pkl"

    return new_model_file_path


def rename_model_pickle_file(new_model_file_path, best_model_data):
    # retrieve best model path
    src = best_model_data["model_file_path"]

    # rename the original file
    os.rename(src, new_model_file_path)


# EXECUTION FUNCTIONS #

def train_model(X_train, X_test, y_train, y_test, config):
    print(f"Main function - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")

    # define model type
    if config["Model_Parameters"]["model_type"] == "CLASSIFIER":
        eval_model = xgb.XGBClassifier
    elif config["Model_Parameters"]["model_type"] == "REGRESSOR":
        eval_model = xgb.XGBRegressor
    else:
        raise ValueError("\033[1m An incorrect config file was selected \033[0m")

    # train, optimize and save models in a dictionary
    grid = tune_models_hyperparams(eval_model, X_train, y_train, X_test, y_test, config)

    # extract best models
    best_models_grid = get_best_models(eval_model, grid, config)
    if not best_models_grid:
        raise ValueError("\033[1m No model has fitted the data well \033[0m")

    # rename best model pkl file and upload it to gcs bucket
    for best_model_name, best_model_data in best_models_grid.items():
        # print best model
        print_best_grid_results(best_model_name=best_model_name, best_model_data=best_model_data, config=config)

        # rename best model pkl file
        new_model_file_path = generate_model_file_name(best_model_name=best_model_name, config=config)
        rename_model_pickle_file(new_model_file_path=new_model_file_path, best_model_data=best_model_data)

        # delete unnecessary objects from memory
        del X_train
        del X_test
        del y_train
        del y_test

        return best_model_data["model_object"], new_model_file_path
