import ast
from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from src.evaluation_utils.loss_functions import F1Score, RecallScore, PrecisionScore, RMSE, R2, MAE
from src.evaluation_utils import plot_utils
from sklearn.metrics import ConfusionMatrixDisplay
import xgboost as xgb
import shap


# GENERAL MODEL EVALUATION #

def plot_xgb_features_importance(model, max_num_features=50):
    # initialize fig object
    fig_xgb, ax_xgb = plt.subplots(figsize=(25 + 2, 15), dpi=120)

    # extract xgboost features importance
    xgb.plot_importance(model, ax=ax_xgb, max_num_features=max_num_features)

    # remove grid and spines from plot
    ax_xgb.grid(False)
    sns.despine(ax=ax_xgb)

    # set title
    ax_xgb.set_title(f"XGBoost Features Importance")
    ax_xgb.title.set_size(22)

    # set labels
    ax_xgb.set_xlabel("F Score", fontsize=18)
    ax_xgb.set_ylabel("Features", fontsize=18)
    ax_xgb.tick_params(axis='both', which='major', labelsize=14)

    # automatically adjust subplot parameters to give specified padding (i.e. prevent plots overlap)
    fig_xgb.tight_layout()

    return fig_xgb


def plot_shap_features_importance(model, X_train_shap, max_num_features=50):
    # initialize fig object
    fig_shap, ax_shap = plt.subplots(figsize=(25 + 2, 15))

    # initialize shap explainer
    explainer = shap.TreeExplainer(model)

    # calculate shap values - "how much" each feature impacts the prediction
    shap_values = explainer.shap_values(X_train_shap)

    # create fig object
    fig_shap = plt.gcf()

    # # extract shap features importance
    shap.summary_plot(shap_values, X_train_shap, show=False, max_display=None)

    # set title
    fig_shap.suptitle(f"SHAP Features Importance")

    # automatically adjust subplot parameters to give specified padding (i.e. prevent plots overlap)
    fig_shap.tight_layout()

    return fig_shap


# REGRESSION MODEL EVALUATION #

def plot_regression_model_evaluation(model,
                                     data: pd.DataFrame,
                                     cols_features: List[str],
                                     col_y="score",
                                     do_plot=True,
                                     str_title="",
                                     color="C0",
                                     alpha=0.75,
                                     print_func=True,
                                     decimal_precision=2,
                                     plot_improvement=False,
                                     X_reg_all=None,
                                     y_reg_all=None,
                                     config=None
                                     ):
    # define observations and predictions data
    obs = col_y
    pred = model.predict(data[cols_features])
    all_obs = y_reg_all
    all_pred = model.predict(X_reg_all)

    # define residuals
    residuals = obs - pred

    # calculate residuals percentiles
    ps = np.percentile(residuals, [5, 95]).tolist()

    # plot/not
    if do_plot:
        # plot predictions by observations graphs
        predictions_by_observations_fig = plot_utils.plot_predictions_by_observations_graphs(observations=obs,
                                                                                             predictions=pred,
                                                                                             alpha=alpha,
                                                                                             decimal_precision=decimal_precision,
                                                                                             color=color,
                                                                                             config=config
                                                                                             )
        if print_func:
            # regression model scoring
            rmse = RMSE(squared=False, direction=None).score(y_test=obs, y_pred=pred)
            r2 = R2(direction=None).score(y_test=obs, y_pred=pred)
            mae = MAE(direction=None).score(y_test=obs, y_pred=pred)
            # add title
            plot_title = f"Target: {obs.name}\n\nDisplayed data: {str_title}"
            # add scoring metrics
            plot_rmse = f"RMSE: {rmse:.{decimal_precision}f}"
            plot_r2 = f"R2: {r2:.{decimal_precision}f}"
            plot_mae = f"MAE: {mae:.{decimal_precision}f}"
            # add text
            plot_text = f"{plot_title}\n\n{plot_rmse}\n{plot_r2}\n{plot_mae}"
            # get upper axis from the fig
            predictions_by_observations_ax = predictions_by_observations_fig.axes[0]
            # position text
            predictions_by_observations_fig.text(0.45,
                                                 0.95,
                                                 plot_text,
                                                 transform=predictions_by_observations_ax.transAxes,
                                                 fontsize=16,
                                                 verticalalignment='top',
                                                 bbox=dict(alpha=0.5)
                                                 )
            # automatically adjust subplot parameters to give specified padding (i.e. prevent plots overlap)
            predictions_by_observations_fig.tight_layout()

        # plot residuals by observations graphs
        residuals_by_observations_fig, \
        normalize_and_filtered_residuals_boxplot_fig, \
        normalized_residuals_buckets_by_obs_buckets_fig = \
            plot_utils.plot_residuals_by_observations_graphs(observations=obs,
                                                             predictions=pred,
                                                             alpha=alpha,
                                                             decimal_precision=decimal_precision,
                                                             color=color,
                                                             all_observations=all_obs,
                                                             all_predictions=all_pred,
                                                             config=config
                                                             )

        if print_func:
            # add title
            plot_title = f"Target: {obs.name}\n\nDisplayed data: {str_title}"
            # add residuals confidence interval
            plot_residuals_ci = f"Residuals 95% Confidence Interval:  {ps[0]:.{decimal_precision}f} to {ps[1]:.{decimal_precision}f}"
            # add text
            plot_text = f"{plot_title}\n\n{plot_residuals_ci}"
            # get upper axis from the fig
            residuals_by_observations_ax = residuals_by_observations_fig.axes[0]
            residuals_by_observations_fig.text(0.45,
                                               0.95,
                                               plot_text,
                                               transform=residuals_by_observations_ax.transAxes,
                                               fontsize=16,
                                               verticalalignment='top',
                                               bbox=dict(alpha=0.5)
                                               )
            # automatically adjust subplot parameters to give specified padding (i.e. prevent plots overlap)
            residuals_by_observations_fig.tight_layout()

        # plot improvement graph
        if plot_improvement:
            fig_improvement, ax_improvement = plt.subplots(figsize=(25 + 2, 15))
            plot_utils.plot_observations_batches_indexed_by_predictions(observations_arr=obs,
                                                                        predictions_arr=pred,
                                                                        do_plot=True,
                                                                        color=color,
                                                                        cutoff=None,
                                                                        ax=ax_improvement,
                                                                        normalize_to_rand=False,
                                                                        str_ylabel=f"{col_y.name}\n(Relative to Random Selection)",
                                                                        str_xlabel=f"Predicted score rank\n(The lower the better)",
                                                                        )
            # automatically adjust subplot parameters to give specified padding (i.e. prevent plots overlap)
            fig_improvement.tight_layout()
        else:
            fig_improvement = None
        if str_title:
            if fig_improvement:
                fig_improvement.suptitle(f"Model Performance vs. Random Selection\n({str_title})", fontsize=20)

    else:
        fig_improvement = fig = None

    return dict(rmse=rmse,
                r2=r2,
                residual_limits=ps,
                predictions_by_observations_fig=predictions_by_observations_fig,
                residuals_by_observations_fig=residuals_by_observations_fig,
                normalize_and_filtered_residuals_boxplot_fig=normalize_and_filtered_residuals_boxplot_fig,
                normalized_residuals_buckets_by_obs_buckets_fig=normalized_residuals_buckets_by_obs_buckets_fig,
                fig_improvement=fig_improvement
                )


# CLASSIFICATION MODEL EVALUATION #

def plot_confusion_matrix(X, y_clf, clf_config, trained_clf=None, str_title="", decimal_precision=2):
    # retrieve target column from config file
    target = ast.literal_eval(clf_config["Data_Processing"]["target"])

    # retrieve pipeline from config file
    pipeline = clf_config["Pipeline"]["pipeline"]
    pipeline_name = clf_config["Pipeline"]["pipeline_name"]

    # plot confusion matrix
    cmp = ConfusionMatrixDisplay.from_estimator(trained_clf,
                                                X,
                                                y_clf,
                                                values_format='d',
                                                display_labels=[f"NO {target}", f"YES {target}"]
                                                )

    # initialize fig object
    fig, ax = plt.subplots(figsize=(25 + 2, 15))

    # set title and labels font size
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    # plot fig
    cmp.plot(ax=ax)

    # set text font size
    for labels in cmp.text_.ravel():
        labels.set_fontsize(22)

    # define observations and predictions data
    obs = y_clf
    pred = trained_clf.predict(X)

    # classification model scoring
    precision = PrecisionScore(beta_value=0.0, direction=None).score(y_test=obs, y_pred=pred)
    recall = RecallScore(beta_value=np.inf, direction=None).score(y_test=obs, y_pred=pred)
    fb = F1Score(beta_value=2.0, direction=None).score(y_test=obs, y_pred=pred)

    # add title
    plot_title = f"{str_title}"

    # add scoring metrics
    plot_precision = f"Precision: {precision:.{decimal_precision}f}"
    plot_recall = f"Recall: {recall:.{decimal_precision}f}"
    plot_fb = f"FBeta: {fb:.{decimal_precision}f}"

    # add text
    plot_text = f"{plot_title}\n\n{plot_precision}\n{plot_recall}\n{plot_fb}"
    fig.text(1, 1, plot_text, bbox=dict(alpha=0.5), fontsize=22)

    # automatically adjust subplot parameters to give specified padding (i.e. prevent plots overlap)
    fig.tight_layout()

    return fig
