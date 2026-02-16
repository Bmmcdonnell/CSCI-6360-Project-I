import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def LatexTable(y_actual, y_pred_in_sample, y_pred_split, m, k, data_name):
    # Calculate SSR (Sum of Squared Residuals)
    ssr_in_sample = np.sum((y_actual - y_pred_in_sample)**2)
    ssr_split = np.sum((y_actual - y_pred_split)**2)

    # Calculate SST (Total Sum of Squares)
    sst = np.sum((y_actual - np.mean(y_actual))**2)

    # Compute R-squared
    r_squared_in_sample = 1 - (ssr_in_sample / sst)
    r_squared_split = 1 - (ssr_split / sst)

    # Calculate Adjusted R-squared
    r_squared_adj_in_sample = 1 - (1 - r_squared_in_sample) * (m) / (m - k)
    r_squared_adj_split = 1 - (1 - r_squared_split) * (m) / (m - k)

    # Standard Deviation of the Errors (SDE)
    sde_in_sample = np.sqrt(ssr_in_sample / (m - k))
    sde_split = np.sqrt(ssr_split / (m - k))

    # Calculate Mean Squared Error of the Null Model (MSE0)
    mse0 = np.sum((y_actual - np.mean(y_actual))**2) / m

    # Root Mean Squared Error
    rmse_in_sample = np.sqrt(ssr_in_sample / (m - k))
    rmse_split = np.sqrt(ssr_split / (m - k))

    # Mean Absolute Error
    mae_in_sample = np.mean(np.abs(y_actual - y_pred_in_sample))
    mae_split = np.mean(np.abs(y_actual - y_pred_split))

    # Symmetric Mean Absolute Percentage Error (SMAPE)
    smape_in_sample = np.mean(2 * np.abs(y_actual - y_pred_in_sample) / (np.abs(y_actual) + np.abs(y_pred_in_sample))) * 100
    smape_split = np.mean(2 * np.abs(y_actual - y_pred_split) / (np.abs(y_actual) + np.abs(y_pred_split))) * 100  

    # Calculate F-statistic
    f_stat_in_sample = (mse0 - (ssr_in_sample / (m - k))) / (ssr_in_sample / (m - k))
    f_stat_split = (mse0 - (ssr_split / (m - k))) / (ssr_split / (m - k))

    # Calculate AIC
    aic_in_sample = m * np.log(ssr_in_sample / m) + 2 * k
    aic_split = m * np.log(ssr_split / m) + 2 * k

    # Calculate BIC
    bic_in_sample = m * np.log(ssr_in_sample / m) + k * np.log(m)
    bic_split = m * np.log(ssr_split / m) + k * np.log(m)

    # Print the results in a LaTeX table format
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Statsmodels - {data_name} Linear Regression}}")
    print(f"\\label{{tab:Statsmodels - {data_name} Linear Regression}}")
    print("\\begin{tabular}{|c|c|c|}\\hline")
    print("Regression & In-Sample & 80-20 Split \\\\ \\hline \\hline")
    print(f"rSq & {r_squared_in_sample:.4f} & {r_squared_split:.4f} \\\\ \\hline")
    print(f"rSqBar & {r_squared_adj_in_sample:.4f} & {r_squared_adj_split:.4f} \\\\ \\hline")
    print(f"sst & {sst:.4f} & {sst:.4f} \\\\ \\hline")
    print(f"sse & {ssr_in_sample:.4f} & {ssr_split:.4f} \\\\ \\hline")
    print(f"sde & {sde_in_sample:.4f} & {sde_split:.4f} \\\\ \\hline")
    print(f"mse0 & {ssr_in_sample:.4f} & {ssr_split:.4f} \\\\ \\hline")
    print(f"rmse & {rmse_in_sample:.4f} & {rmse_split:.4f} \\\\ \\hline")
    print(f"mae & {mae_in_sample:.4f} & {mae_split:.4f} \\\\ \\hline")
    print(f"smape & {smape_in_sample:.4f} & {smape_split:.4f} \\\\ \\hline")
    print(f"m & {m:.4f} & {m:.4f} \\\\ \\hline")
    print(f"dfr & {k-1:.4f} & {k-1:.4f} \\\\ \\hline")
    print(f"df & {m - k:.4f} & {m - k:.4f} \\\\ \\hline")
    print(f"fStat & {f_stat_in_sample:.4f} & {f_stat_split:.4f} \\\\ \\hline")
    print(f"aic & {aic_in_sample:.4f} & {aic_split:.4f} \\\\ \\hline")
    print(f"bic & {bic_in_sample:.4f} & {bic_split:.4f} \\\\ \\hline")
    print("\\end{tabular}")
    print("\\end{table}")



def LinRegAutoMPG():

    # Load the dataset
    Auto_MPG = pd.read_csv('auto_mpg_cleaned.csv')

    # Define the independent variable (X) and the dependent variable (y)
    X = Auto_MPG[['displacement', 'cylinders', 'horsepower', 'weight', 'acceleration', 'model_year']]
    y = Auto_MPG['MPG']

    # Add a constant to the independent variables (for the intercept)
    X = sm.add_constant(X)

    ## In-Sample Regression Analysis
    # Fit the OLS regression model
    reg_in_sample = sm.OLS(y, X).fit()

    # Print the summary of the regression results
    print("In-Sample Regression Results:")
    print(reg_in_sample.summary())
    print("-" * 88)
    print("-" * 88)
    print("-" * 88)

    ## 80-20 Train-Test Split
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the OLS regression model on the training data
    reg_train = sm.OLS(y_train, X_train).fit()

    # Print the summary of the regression results for the training data
    print("Training Set Regression Results:")
    print(reg_train.summary())
    print("-" * 88)
    print("-" * 88)
    print("-" * 88)

    # Make predictions on the original data set to compare to the in-sample regression results
    y_pred_in_sample = reg_in_sample.predict(X)
    y_pred_split = reg_train.predict(X)

    # Get the number of observations (m) and the number of independent variables (k)
    m = len(y)  # Number of observations
    k = X.shape[1] - 1  # Number of independent variables

    # Print the qof statistics in a LaTeX table format
    LatexTable(y, y_pred_in_sample, y_pred_split, m, k, "Auto MPG")





def LinReghouse():

    # Load the dataset
    House_Price = pd.read_csv('house_price_regression_dataset.csv')

    # Define the independent variable (X) and the dependent variable (y)
    X = House_Price[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']]
    y = House_Price['House_Price']

    # Add a constant to the independent variables (for the intercept)
    X = sm.add_constant(X)

    ## In-Sample Regression Analysis
    # Fit the OLS regression model
    reg_in_sample = sm.OLS(y, X).fit()

    # Print the summary of the regression results
    print("In-Sample Regression Results:")
    print(reg_in_sample.summary())
    print("-" * 88)
    print("-" * 88)
    print("-" * 88)

    ## 80-20 Train-Test Split
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the OLS regression model on the training data
    reg_train = sm.OLS(y_train, X_train).fit()

    # Print the summary of the regression results for the training data
    print("Training Set Regression Results:")
    print(reg_train.summary())
    print("-" * 88)
    print("-" * 88)
    print("-" * 88)

    # Make predictions on the original data set to compare to the in-sample regression results
    y_pred_in_sample = reg_in_sample.predict(X)
    y_pred_split = reg_train.predict(X)

    # Get the number of observations (m) and the number of independent variables (k)
    m = len(y)  # Number of observations
    k = X.shape[1] - 1  # Number of independent variables

    # Print the qof statistics in a LaTeX table format
    LatexTable(y, y_pred_in_sample, y_pred_split, m, k, "House Price")




def LinReginsurance():

    # Load the dataset
    Insurance_Charges = pd.read_csv('insurance_cat2num.csv')

    # Define the independent variable (X) and the dependent variable (y)
    X = Insurance_Charges[['intercept', 'age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']]
    y = Insurance_Charges['charges']

    ## In-Sample Regression Analysis
    # Fit the OLS regression model
    reg_in_sample = sm.OLS(y, X).fit()

    # Print the summary of the regression results
    print("In-Sample Regression Results:")
    print(reg_in_sample.summary())
    print("-" * 88)
    print("-" * 88)
    print("-" * 88)

    ## 80-20 Train-Test Split
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the OLS regression model on the training data
    reg_train = sm.OLS(y_train, X_train).fit()

    # Print the summary of the regression results for the training data
    print("Training Set Regression Results:")
    print(reg_train.summary())
    print("-" * 88)
    print("-" * 88)
    print("-" * 88)

    # Make predictions on the original data set to compare to the in-sample regression results
    y_pred_in_sample = reg_in_sample.predict(X)
    y_pred_split = reg_train.predict(X)

    # Get the number of observations (m) and the number of independent variables (k)
    m = len(y)  # Number of observations
    k = X.shape[1] - 1  # Number of independent variables

    # Print the qof statistics in a LaTeX table format
    LatexTable(y, y_pred_in_sample, y_pred_split, m, k, "Insurance Charges")




# LinRegAutoMPG()
# LinReghouse()
# LinReginsurance()