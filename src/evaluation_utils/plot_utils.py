import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import ast


# PLOT DATA DISTRIBUTION GRAPHS #

def plot_data_distribution(target):
    # filter out 0's
    target = target[target > 0]
    # create subplot and axes object containing the plot
    fig, axs = plt.subplots(2, figsize=(25 + 2, 15))

    # plot target data distribution
    axs[0].hist(target, bins=np.arange(min(target), max(target) + 1, 1))
    axs[0].set_title(f"Histogram of {target.name}\n(where {target.name}>0)")
    axs[0].title.set_size(22)
    axs[0].tick_params(axis='both', which='major', labelsize=14)

    # plot log target data distribution
    axs[1].hist(np.log(target), bins=np.arange(min(np.log(target)), max(np.log(target)) + 0.1, 0.01))
    axs[1].set_title(f"Histogram of log {target.name}\n(where {target.name}>0)")
    axs[1].title.set_size(22)
    axs[1].tick_params(axis='both', which='major', labelsize=14)

    return fig


# PLOT PREDICTIONS BY OBSERVATIONS GRAPHS #

def plot_linear_regression_fit(observations,
                               predictions,
                               alpha,
                               ax=None,
                               ylabel=True,
                               decimal_precision=2,
                               color="C0"
                               ):
    # create subplot and axes object containing the plot
    if ax is None:
        fig, ax = plt.subplots()

    # plot data and a linear regression model fit
    sns.regplot(x=observations, y=predictions, ax=ax, ci=99, color=color, scatter_kws={"alpha": alpha})

    # define uniform axis size
    mn = min((min(observations), min(predictions)))
    mx = max((max(observations), max(predictions)))

    # identity line for better visual aid
    ax.plot([mn, mx], [mn, mx], "--", color="k", zorder=-1)

    # equal axis limits
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)

    # remove the top and right spines from plot(s)
    sns.despine(ax=ax)

    # set labels
    ax.set_xlabel("Observations", fontsize=18)
    ax.set_ylabel("Predictions", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # set title
    ax.set_title(f"Predictions by Observations")
    ax.title.set_size(22)

    return ax


def plot_predictions_by_observations_graphs(observations,
                                            predictions,
                                            alpha=1.0,
                                            fig: plt.Figure = None,
                                            decimal_precision=2,
                                            color="C0",
                                            config=None
                                            ) -> plt.Figure:
    # creates a figure (but with no axes in it)
    fig = plt.figure(figsize=(25 + 2, 15))

    # create an axis (for each sub-plot) at specific location inside a regular grid
    ax_obs_pred = plt.subplot2grid((4, 5), (0, 1), rowspan=2, colspan=5)
    ax_box_plot_pred = plt.subplot2grid((4, 5), (2, 1), rowspan=2, colspan=5)

    # plot_linear_regression_fit graph
    plot_linear_regression_fit(observations=observations,
                               predictions=predictions,
                               alpha=alpha,
                               ax=ax_obs_pred,
                               decimal_precision=decimal_precision,
                               color=color,
                               )

    # plot_boxplot graph (predictions)
    plot_boxplot(observations=observations,
                 predictions=predictions,
                 ax=ax_box_plot_pred,
                 config=config
                 )

    # automatically adjust subplot parameters to give specified padding (i.e. prevent plots overlap)
    fig.tight_layout()

    # # tune the subplot layout
    # fig.subplots_adjust(hspace=0)

    return fig


# PLOT RESIDUALS BY OBSERVATIONS GRAPHS #

def plot_residuals(observations, predictions, ax=None, ylabel=True, alpha=1.0, decimal_precision=2, color="C0"):
    # create subplot and axes object containing the plot
    if ax is None:
        fig, ax = plt.subplots()

    # plot residuals vs. predictions
    residuals = predictions - observations
    ax.plot(predictions, residuals, "o", color=color, alpha=alpha)

    # remove the top and right spines from plot(s)
    sns.despine(ax=ax)

    # add a horizontal line across the axis
    ax.axhline(0, color="k", zorder=-1, ls="--")

    # calculate residuals percentiles (5%/95% i.e. over/under estimations)
    perc_upper = 95
    perc_lower = 5
    residuals_percentiles = np.percentile(residuals, [perc_lower, perc_upper]).tolist()

    # add a horizontal line across the residuals percentiles axis
    for p in residuals_percentiles:
        ax.axhline(p, color="gray", zorder=-1, ls="--", lw=0.5)

    # add text across the residuals percentiles axis
    ax.text(
        x=ax.get_xlim()[1],
        y=residuals_percentiles[1],
        s=f"Overestimation\n{perc_upper}$^{{th}}$perc.: ${residuals_percentiles[1]:.{decimal_precision}f}$",
        color="gray",
        size=14,
        ha="right",
        va="bottom",  # above the line
        zorder=999,
    )
    ax.text(
        x=ax.get_xlim()[1],
        y=residuals_percentiles[0],
        s=f"Underestimation\n{perc_lower}${{th}}$perc.: ${residuals_percentiles[0]:.{decimal_precision}f}$",
        color="gray",
        size=14,
        ha="right",
        va="top",  # below the line, so that there is no overlap
        zorder=999,
    )

    # set labels
    ax.set_xlabel("Observations", y=-1, fontsize=18)
    ax.set_ylabel("Residuals (=Pred-Obs)", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # set title
    ax.set_title("Residuals by Observations")
    ax.title.set_size(22)

    # hide grid lines
    ax.grid(False)

    return ax


def plot_residuals_kde_on_the_side(residuals, ax, decimal_precision=2, color="C0"):
    # residuals = residuals.reshape(-1)

    # KDE analogous to a histogram, represents the data using a continuous probability density curve
    sns.kdeplot(y=residuals, ax=ax, color=color)

    # remove the top and right spines from plot(s)
    sns.despine(ax=ax, left=True)

    # moving bottom spine up to y=0 position
    ax.spines["bottom"].set_position("zero")

    # calculate residuals percentiles (5%/95% i.e. over/under estimations)

    residuals_percentiles = np.percentile(residuals, [5, 95]).tolist()

    # add avg residuals
    residuals_percentiles.append(np.mean(residuals))

    # round an array to the given number of decimals
    tks = np.round(residuals_percentiles, decimal_precision)

    # add secondary axis
    second_ax = ax.secondary_yaxis("right")

    # edit secondary axis
    second_ax.set_yticks(residuals_percentiles)  # set the y ticks (with list of ticks)
    second_ax.set_yticklabels(tks)  # set the y-tick labels (with list of string labels)
    # second_ax.set_xticks([])  # set the x ticks (with list of ticks)
    # second_ax.set_xlabel(None)  # set the x ticks labels (with list of string labels)

    # edit primary axis
    # ax.set_yticks([0])  # set the y ticks (with list of ticks)
    # ax.set_yticklabels([0])  # set the y-tick labels (with list of string labels)
    ax.set_xticks([])  # set the x ticks (with list of ticks)
    ax.set_xlabel(None)  # set the x ticks labels (with list of string labels)
    ax.set_xlim(reversed(ax.get_xlim()))  # set the x-axis view limits
    ax.tick_params(left=False, labelleft=False)  # remove multiple labels/ticks

    # hide grid lines
    ax.grid(False)
    # second_ax.grid(False)

    return ax


def plot_residuals_by_observations_graphs(observations,
                                          predictions,
                                          alpha=1.0,
                                          fig: plt.Figure = None,
                                          decimal_precision=2,
                                          color="C0",
                                          all_observations=None,
                                          all_predictions=None,
                                          config=None
                                          ) -> plt.Figure:
    # creates a figure (but with no axes in it)
    if fig is None:
        fig = plt.figure(figsize=(25 + 2, 15))

    # create an axis (for each sub-plot) at specific location inside a regular grid
    ax_residuals = plt.subplot2grid((4, 5), (0, 1), rowspan=2, colspan=4)
    ax_residual_kde = plt.subplot2grid((4, 5), (0, 0), rowspan=2, colspan=1, sharey=ax_residuals)
    ax_box_plot_residuals = plt.subplot2grid((4, 5), (2, 1), rowspan=2, colspan=5)

    # plot_residuals graph
    plot_residuals(observations,
                   predictions,
                   alpha=alpha,
                   ax=ax_residuals,
                   decimal_precision=decimal_precision,
                   color=color,
                   )

    # plot_residual_kde graph
    plot_residuals_kde_on_the_side(residuals=predictions - observations,
                                   ax=ax_residual_kde,
                                   decimal_precision=decimal_precision,
                                   color=color,
                                   )

    # plot_boxplot graph (residuals)
    plot_boxplot(observations=observations,
                 # passing predictions in order to calculate residuals in 1 of 2 ways:
                 # 1. predictions-observations , 2. (predictions-observations)/observations
                 predictions=predictions,
                 ax=ax_box_plot_residuals,
                 is_residuals=True,
                 config=config
                 )

    # set the zorder for the artist
    ax_residual_kde.set_zorder(-1)

    # automatically adjust subplot parameters to give specified padding (i.e. prevent plots overlap)
    fig.tight_layout()

    # # tune the subplot layout
    # fig.subplots_adjust(hspace=0)

    # plot_boxplot graph (normalize and filtered residuals)
    normalize_and_filtered_residuals_boxplot_fig, normalize_and_filtered_residuals_boxplot_ax = plt.subplots(
        figsize=(25 + 2, 15))
    plot_boxplot(observations=observations,
                 # passing predictions in order to calculate residuals in 1 of 2 ways:
                 # 1. predictions-observations , 2. (predictions-observations)/observations
                 predictions=predictions,
                 ax=normalize_and_filtered_residuals_boxplot_ax,
                 is_residuals=True,
                 is_normalize_residuals=True,
                 config=config
                 )
    normalize_and_filtered_residuals_boxplot_fig.tight_layout()

    # plot_normalized_residuals_buckets_by_obs_buckets graph
    normalized_residuals_buckets_by_obs_buckets_fig = \
        plot_normalized_residuals_buckets_by_obs_buckets(observations=all_observations,
                                                         predictions=all_predictions,
                                                         config=config
                                                         )

    return fig, normalize_and_filtered_residuals_boxplot_fig, normalized_residuals_buckets_by_obs_buckets_fig


# PLOT IMPROVEMENTS GRAPH #

def plot_observations_batches_indexed_by_predictions(observations_arr,
                                                     predictions_arr,
                                                     window_size=250,
                                                     estimation_func=np.mean,
                                                     do_plot=True,
                                                     color="C0",
                                                     cutoff=None,
                                                     ax=None,
                                                     normalize_to_rand=False,
                                                     str_ylabel=None,
                                                     str_xlabel=None,
                                                     str_model_name="Our model",
                                                     ):
    """
    Background:
    This script is used to generate a plot displaying a graph which gives a score to batches of observations
    according to indices of sorted predictions. For each batch of a given size (currently 250) of observations the
    average value is calculated and placed on the graph along with its position in the array of locations.
    So a descending graph indicates a good fit between observations and predictions.
    Example:
    predictions=[17,15,15,3,100] -->descending_predictions=[100,17,15,15,3] --> predictions_index=[4,0,1,2,3]
    observations=[9,8,8,2,35] --> observations[predictions_index]=[35,9,8,8,2]
    --> PERFECT match between observations and predictions order MEANING descending graph
    Terms and Explanations:
    1. x_axis = score = ordered index (i.e. position of observations according to descending_predictions)
        * left side = good score
        * right side = bad score
    2. y_axis = performance = avg. target value (i.e. avg. of observations positioned according to descending_predictions)
    3. estimations_rand = random selection = ~avg. target of the population (i.e. the gray value on y_axis =~ avg target data)
    4. cutoff = the performance of X%th (currently X=10) of the population = vertical line of a score (x_axis) with performance (y_axis)
        * i.e. if we use the model and choose the best 10% creators - we will get performance of CUTOFF
        * where CUTOFF = performance (y_axis value) = the intersection point between the v. line and the graph
    5. normalize_to_rand (=estimations_rand.mean()) = normalize both estimation_actual and estimations_rand by the population rdm value
        * that way we can compare DIFFERENT  models with the same GENERIC graph
    """
    # create observations array
    observations_arr = np.asanyarray(observations_arr)

    # create predictions array
    predictions_arr = np.asanyarray(predictions_arr).copy()

    # create predictions' index array and sort it by descending values (i.e. predictions)
    lix = np.argsort(predictions_arr)[-1::-1]

    # given size (=250) - create sliding window view into the "observations array" indexed by "predictions array decending index"
    # i.e. extracts subsets of the array
    # example: predictions=[127,115,100] --> predictions_index=[0,1,2] --> observations[predictions_index]=[9,8,8]
    sliding_window = np.lib.stride_tricks.sliding_window_view(observations_arr[lix], window_size)

    # create array of "avg observations value" for each sliding_window (to get a sense of relationiship between sorted indexed observations and predictions)
    # i.e. actual avg value for every observations' window which sorted by predictions index
    # example: predictions[observations_index]=[9,8,8] --> avg(predictions[observations_index]) = 8.33
    estimation_actual = estimation_func(sliding_window, axis=1)

    # create array of numbers from size (250) to size+number of sliding windows (250+X)
    x_actual = np.arange(window_size, window_size + len(estimation_actual))

    estimations_rand = []
    for _ in range(50):
        # randomly shuffle the sorted predictions' index array
        np.random.shuffle(lix)
        # given size (=250) - create a sliding window view into the the "observations array" indexed by "shuffled predictions array"
        sliding_window = np.lib.stride_tricks.sliding_window_view(observations_arr[lix], window_size)
        # create array of "avg observations value" for each sliding_window (to get a sense of relationiship between shuffled indexed observations and predictions)
        # i.e. actual avg value for every observations' window which sorted by shuffled predictions index
        estimation = estimation_func(sliding_window, axis=1)
        estimations_rand.append(estimation.copy())

    #  join a sequence of arrays along a new axis
    estimations_rand = np.stack(estimations_rand)

    if normalize_to_rand:
        # calculate avg observations values of the random shuffled windowed array
        ref = estimations_rand.mean()
        # normalize each array by dividing with constant
        estimations_rand = estimations_rand / ref
        estimation_actual = estimation_actual / ref

    if not cutoff:
        # create cutoff with the size of 10% from number of predictions
        cutoff = int(0.1 * len(predictions_arr))

    # retrieve index of the smallest value in x_actual array minus cutoff value (10% of number of predictions)
    # example: cutoff=622, x_actual=[250,251,...,6223] --> (x_actual - cutoff)=[-372,-371,...,0,...5601] --> abs(...)=0 --> ix=372
    ix = np.argmin(np.abs(x_actual - cutoff))

    # retrieve the value of this index from estimation_actual array i.e. min(avg(predictions[observations_index]))
    # i.e. the highest avg(predictions[observations_index]) values of the remaining 90% of the windows (since 10% were removed by the cutoff)
    improvement = estimation_actual[ix]

    if do_plot:
        if ax is None:
            # initialize fig object
            _, ax = plt.subplots(figsize=(25 + 2, 15))

        # create array of numbers from size (250) to size+number of sliding windows (250+X)
        x_rand = np.arange(estimations_rand.shape[1]) + window_size

        for estimation in estimations_rand:
            # plot scatter plot of:
            # x_axis = x_rand (running number from size to size (250) to size+number of sliding windows (250+X)
            # y_axis = random estimation ("avg observations value" for each sliding_window sorted by shuffled predictions index)
            ax.plot(x_rand, estimation, color="gray", lw=0.1, alpha=0.25)

        # add text
        ax.text(x_rand[-1], estimations_rand.mean(), "Random Selection", color="gray", fontsize=18)

        # plot scatter plot of:
        # x_axis = x_actual (running number from size to size (250) to size+number of sliding windows (250+X)
        # y_axis = actual estimation (actual avg value for every observations' window which sorted by predictions index)
        ax.plot(x_actual, estimation_actual, "-", color=color)

        # add axis labels
        if str_ylabel:
            ax.set_ylabel(str_ylabel, rotation=90, fontsize=18)
        if str_xlabel:
            ax.set_xlabel(str_xlabel, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)

        # add text
        ax.text(x_actual[-1], estimation_actual[-1], str_model_name, color=color, va="center")

        if cutoff:
            # add cutoff vertical line (in the value of 10% from number of predictions)
            ax.axvline(cutoff, color="gray", lw=0.75, zorder=-1, ls="--")
            ax.text(x=x_actual[ix], y=improvement, s=f"{improvement:.1f}-", ha="right", va="center", color="gray", )
            # remove the top and right spines from plot(s)
            sns.despine(ax=ax)

        return improvement, ax

    else:
        return improvement


# PLOT BOXPLOT GRAPH #

def create_buckets(values_for_bucketing_observations):
    # pair 2 consecutive items in given list
    list_of_paired_values = [[values_for_bucketing_observations[i], values_for_bucketing_observations[i + 1]] for i in
                             range(len(values_for_bucketing_observations) - 1)]

    # create text buckets out of pairs
    list_of_buckets = []
    for paired_value in list_of_paired_values:
        text_range = f"""{paired_value[0]}-{paired_value[1]}"""
        list_of_buckets.append(text_range)
    zipped_values_and_buckets = list(zip(list_of_paired_values, list_of_buckets))

    return zipped_values_and_buckets


def transform_values_to_buckets(x, zipped_values_and_buckets):
    # apply bucketing transformation on given value
    for item in zipped_values_and_buckets:
        if item[0][0] <= x < item[0][1]:
            return item[1]


def plot_boxplot(observations,
                 predictions,
                 ax=None,
                 str_xlabel="Observations Buckets",
                 is_residuals=None,
                 is_normalize_residuals=None,
                 config=None
                 ):
    # retrieve target column from config file
    target = ast.literal_eval(config["Data_Processing"]["target"])

    # define buckets and filters for each pipeline
    if target == 'tutorials_d7':
        values_for_bucketing_observations = [0, 1, 5, 10, 30, 50, 60, 70, 80, 90, 100, np.inf]
        is_filter_observations_low_values = True
        filter_observations_low_values = 10
    elif target == 'deposits_count_d7':
        values_for_bucketing_observations = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]
        is_filter_observations_low_values = False
    else:
        raise ValueError("\033[1m An incorrect config file was selected,"
                         " values_for_bucketing_observations parameter cannot be defined \033[0m")

    # create df of observations and predictions
    df = pd.DataFrame(data=np.column_stack((observations, predictions)),
                      columns=['observations', 'predictions'])

    # create observations buckets based on given values
    observations_zipped_values_and_buckets = create_buckets(values_for_bucketing_observations)

    # add buckets column to the df (by applying bucketing transformation on observations)
    df["observations_buckets"] = [transform_values_to_buckets(x, observations_zipped_values_and_buckets) for x in
                                  observations]

    # extract buckets order from observations_zipped_values_and_buckets
    boxplot_x_axis_order = [item[-1] for item in observations_zipped_values_and_buckets]

    # decide which boxplot to plot (predictions OR residuals)
    if not is_residuals:
        boxplot_y_axis_name = 'predictions'
        str_ylabel = "Predictions"

    elif is_residuals:
        # decide which boxplot to plot (absolute residuals OR normalized residuals)
        if not is_normalize_residuals:
            boxplot_y_axis_name = 'residuals'
            str_ylabel = f"Residuals\n= (Pred-Obs)"
            df[boxplot_y_axis_name] = df['predictions'] - df['observations']

        elif is_normalize_residuals:
            boxplot_y_axis_name = 'normalized residuals'
            str_ylabel = f"Normalized Residuals (%)"
            df[boxplot_y_axis_name] = 100 * (df['predictions'] - df['observations']) / df['observations']
            ax.set_title("Residuals (Normalized) by Observations\n\nNormalized: (Pred-Obs)/Obs")
            ax.title.set_size(22)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            plt.xticks(range(1, 20))

            if is_filter_observations_low_values:
                df = df[df["observations"] >= filter_observations_low_values]
                str_ylabel = f"Normalized and Filtered Residuals (%)"
                ax.set_title(
                    "Residuals (Normalized and Filtered) by Observations\n\nNormalized: (Pred-Obs)/Obs\nFiltered: Observations>=10")
                ax.title.set_size(22)
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                plt.xticks(range(1, 20))
    # plot
    sns.set_style("whitegrid")
    sns.boxplot(x="observations_buckets",
                y=boxplot_y_axis_name,
                ax=ax,
                data=df,
                order=boxplot_x_axis_order,
                )

    # set labels
    ax.set_xlabel(str_xlabel, fontsize=18)
    ax.set_ylabel(str_ylabel, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # # hide grid lines
    # ax.grid(True)

    return ax


def plot_normalized_residuals_buckets_by_obs_buckets(observations,
                                                     predictions,
                                                     config=None
                                                     ):
    # retrieve target column from config file
    target = ast.literal_eval(config["Data_Processing"]["target"])

    # define buckets and filters for each pipeline
    if target == 'tutorials_d7':
        values_for_bucketing_observations = [0, 1, 5, 10, 30, 50, 60, 70, 80, 90, 100, np.inf]
        values_for_bucketing_abs_normalized_residuals = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, np.inf]

    elif target == 'deposits_count_d7':
        values_for_bucketing_observations = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]
        values_for_bucketing_abs_normalized_residuals = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, np.inf]
    else:
        raise ValueError("\033[1m An incorrect config file was selected,"
                         " values_for_bucketing_observations parameter cannot be defined \033[0m")

    # create observations and predictions df
    df = pd.DataFrame(data=np.column_stack((observations, predictions)),
                      columns=['observations', 'predictions']
                      )
    # filter out observations=0 since the normalized residual will be ERROR (e.g. obs=0, pred=5 ---> (5-0)/0=ERROR)
    df = df[df["observations"] > 0]

    # create observations buckets
    observations_zipped_values_and_buckets = create_buckets(values_for_bucketing_observations)

    # add buckets column to the df
    df["observations_buckets"] = [transform_values_to_buckets(x, observations_zipped_values_and_buckets) for x in
                                  df["observations"]]

    # add abs_normalized_residuals column to the df
    df['abs_normalized_residuals'] = 100. * (df['predictions'] - df['observations']) / df['observations']
    df['abs_normalized_residuals'] = df['abs_normalized_residuals'].abs()

    # create abs_normalized_residuals buckets
    abs_normalized_residuals_zipped_values_and_buckets = create_buckets(values_for_bucketing_abs_normalized_residuals)

    # add buckets column to the df
    df["abs_normalized_residuals_buckets"] = [
        transform_values_to_buckets(x, abs_normalized_residuals_zipped_values_and_buckets) for x in
        df['abs_normalized_residuals']]

    # add to abs_normalized_residuals_buckets values the '%' sign
    df['abs_normalized_residuals_buckets'] = df['abs_normalized_residuals_buckets'].astype(str) + '%'

    # extract observations_buckets order
    sorter = [item[-1] for item in observations_zipped_values_and_buckets]
    sorter = [value for value in sorter if value in df['observations_buckets'].unique()]

    # extract abs_normalized_residuals_buckets order
    sorter_2 = [item[-1] for item in abs_normalized_residuals_zipped_values_and_buckets]
    sorter_2 = [s + "%" for s in sorter_2]
    sorter_2 = [value for value in sorter_2 if value in df['abs_normalized_residuals_buckets'].unique()]

    # create function to add percentage to columns
    def barPerc(df, xVar, ax):
        """
        barPerc(): Add percentage for hues to bar plots
        args:
            df: pandas dataframe
            xVar: (string) X variable
            ax: Axes object (for Seaborn Countplot/Bar plot or pandas bar plot)
        """

        # 1. how many X categories
        # check for NaN and remove
        numX = len([x for x in df[xVar].unique() if x == x])

        # 2. The bars are created in hue order, organize them
        bars = ax.patches
        # 2a. For each X variable
        for ind in range(numX):
            # 2b. Get every hue bar
            # ex. 8 X categories, 4 hues => [0, 8, 16, 24] are hue bars for 1st X category
            hueBars = bars[ind:][::numX]
            # 2c. Get the total height (for percentages)
            total = sum([x.get_height() for x in hueBars])

            # 3. Print the percentage on the bars
            for bar in hueBars:
                ax.text(bar.get_x() + bar.get_width() / 2.,
                        bar.get_height(),
                        f'{bar.get_height() / total:.0%}',
                        ha="center",
                        va="bottom"
                        )

    df = df[['observations_buckets', 'abs_normalized_residuals_buckets']]

    # create subplot and axes object containing the plot
    fig, ax = plt.subplots(figsize=(25 + 2, 15))

    # plot
    ax = sns.countplot(x="observations_buckets",
                       hue="abs_normalized_residuals_buckets",
                       data=df,
                       order=sorter,
                       hue_order=sorter_2,
                       palette="RdYlGn_r"
                       )

    # set title
    ax.set_title(f"Normalized Residuals\nby Observations")
    ax.title.set_size(22)

    # set labels
    ax.set_xlabel("Observations Buckets", fontsize=18)
    ax.set_ylabel("Count (#)", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # # set legend
    ax.legend(title="Normalized Residuals Buckets", fontsize=20, title_fontsize=20)

    # add annotations
    barPerc(df, 'observations_buckets', ax)

    # automatically adjust subplot parameters to give specified padding (i.e. prevent plots overlap)
    fig = ax.get_figure()
    plt.tight_layout()

    # delete unnecessary objects from memory
    del df

    return fig
