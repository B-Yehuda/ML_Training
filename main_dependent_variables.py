import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from scipy.stats import gaussian_kde


# Define process
multivariate_normal_process = False
LinearRegression_process = True
gaussian_kde_process = False


# Load CSV file containing the dependent variables
df = pd.read_csv('data.csv')
col_1 = df['tutorials_d7']
col_2 = df['depositors_d7']


if multivariate_normal_process:
    # Step 1: Estimate the Joint Distribution
    mean = [col_1.mean(), col_2.mean()]  # means for the 2 variables
    cov = df[['tutorials_d7', 'depositors_d7']].cov().values  # covariance matrix (relationship between the 2 variables)

    # Step 2: Formulate the Joint Probability Density Function (PDF)
    joint_dist = multivariate_normal(mean=mean, cov=cov)

    # Step 3: Calculate the probability of obtaining values
    values = np.column_stack((col_1, col_2))
    probabilities = joint_dist.pdf(values)

    # Step 4: Add Probability data to a new column in the DataFrame
    df['probability'] = probabilities

    df.to_csv("multivariate_normal_process.csv", index=False)


if LinearRegression_process:
    # Step 1: Find the Dependency between the columns
    X = col_1.values.reshape(-1, 1)
    y = col_2.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)

    # Step 2: Calculate the probability of obtaining values
    predicted_col_2 = model.predict(X)
    residuals = y - predicted_col_2
    mean_residuals = residuals.mean()
    probabilities = residuals

    # Step 3: Add Probability data to a new column in the DataFrame
    df['predicted_col_2'] = predicted_col_2
    df['probability'] = probabilities

    df.to_csv("LinearRegression_process.csv", index=False)


if gaussian_kde_process:
    # Step 1: Perform Kernel Density Estimation (KDE)
    kde = gaussian_kde([col_1, col_2])

    # Step 2: Calculate the probability of obtaining values
    probabilities = kde.pdf([col_1, col_2])

    # Step 3: Add Probability data to a new column in the DataFrame
    df['probability'] = probabilities

    df.to_csv("gaussian_kde_process.csv", index=False)
