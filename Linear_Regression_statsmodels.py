from wsgiref import validate
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def LatexTable(y_actual, y_test, y_pred_in_sample, y_pred_split_test, k, data_name):
    # Get the number of observations (m) or (t) for the test set
    m = len(y_actual)  # Number of observations in the original dataset
    t = len(y_test)  # Number of observations in the test set

    # Calculate SSR (Sum of Squared Residuals)
    ssr_in_sample = np.sum((y_actual - y_pred_in_sample)**2)
    ssr_test = np.sum((y_test - y_pred_split_test)**2)

    # Calculate SST (Total Sum of Squares)
    sst_in_sample = np.sum((y_actual - np.mean(y_actual))**2)
    sst_test = np.sum((y_test - np.mean(y_test))**2)

    # Compute R-squared
    r_squared_in_sample = 1 - (ssr_in_sample / sst_in_sample)
    r_squared_test = 1 - (ssr_test / sst_test)

    # Calculate Adjusted R-squared
    r_squared_adj_in_sample = 1 - (1 - r_squared_in_sample) * (m) / (m - k)
    r_squared_adj_test = 1 - (1 - r_squared_test) * (t) / (t - k)

    # Standard Deviation of the Errors (SDE)
    sde_in_sample = np.sqrt(ssr_in_sample / (m - k))
    sde_test = np.sqrt(ssr_test / (t))

    # Calculate Mean Squared Error
    mse0_in_sample = sst_in_sample / m
    mse0_test = sst_test / t

    # Root Mean Squared Error
    rmse_in_sample = np.sqrt(ssr_in_sample / (m - k))
    rmse_test = np.sqrt(ssr_test / (t))

    # Mean Absolute Error
    mae_in_sample = np.mean(np.abs(y_actual - y_pred_in_sample))
    mae_test = np.mean(np.abs(y_test - y_pred_split_test))

    # Symmetric Mean Absolute Percentage Error (SMAPE)
    smape_in_sample = np.mean(2 * np.abs(y_actual - y_pred_in_sample) / (np.abs(y_actual) + np.abs(y_pred_in_sample))) * 100
    smape_test = np.mean(2 * np.abs(y_test - y_pred_split_test) / (np.abs(y_test) + np.abs(y_pred_split_test))) * 100  

    # Calculate F-statistic
    f_stat_in_sample = (mse0_in_sample - (ssr_in_sample / (m - k))) / (ssr_in_sample / (m - k))
    f_stat_test = (mse0_test - (ssr_test / t)) / (ssr_test / t)

    # Calculate AIC
    aic_in_sample = m * np.log(ssr_in_sample / m) + 2 * k
    aic_test = t * np.log(ssr_test / t) + 2 * k

    # Calculate BIC
    bic_in_sample = m * np.log(ssr_in_sample / m) + k * np.log(m)
    bic_test = t * np.log(ssr_test / t) + k * np.log(t)

    # Print the results in a LaTeX table format
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Statsmodels - {data_name} Linear Regression}}")
    print(f"\\label{{tab:Statsmodels - {data_name} Linear Regression}}")
    print("\\begin{tabular}{|c|c|c|}\\hline")
    print("Regression & In-Sample & 80-20 Split \\\\ \\hline \\hline")
    print(f"rSq & {r_squared_in_sample:.4f} & {r_squared_test:.4f} \\\\ \\hline")
    print(f"rSqBar & {r_squared_adj_in_sample:.4f} & {r_squared_adj_test:.4f} \\\\ \\hline")
    print(f"sst & {sst_in_sample:.4f} & {sst_test:.4f} \\\\ \\hline")
    print(f"sse & {ssr_in_sample:.4f} & {ssr_test:.4f} \\\\ \\hline")
    print(f"sde & {sde_in_sample:.4f} & {sde_test:.4f} \\\\ \\hline")
    print(f"mse0 & {mse0_in_sample:.4f} & {mse0_test:.4f} \\\\ \\hline")
    print(f"rmse & {rmse_in_sample:.4f} & {rmse_test:.4f} \\\\ \\hline")
    print(f"mae & {mae_in_sample:.4f} & {mae_test:.4f} \\\\ \\hline")
    print(f"smape & {smape_in_sample:.4f} & {smape_test:.4f} \\\\ \\hline")
    # print(f"m & {m:.4f} & {t:.4f} \\\\ \\hline")
    # print(f"dfr & {k-1:.4f} & {k-1:.4f} \\\\ \\hline")
    # print(f"df & {m - k:.4f} & {t - k:.4f} \\\\ \\hline")
    # print(f"fStat & {f_stat_in_sample:.4f} & {f_stat_test:.4f} \\\\ \\hline")
    # print(f"aic & {aic_in_sample:.4f} & {aic_test:.4f} \\\\ \\hline")
    # print(f"bic & {bic_in_sample:.4f} & {bic_test:.4f} \\\\ \\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def sorted_plot(y_actual, y_pred, data_name, model_name, validate=False):
    # Create a DataFrame with actual and predicted values
    df = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred})
    # Sort the DataFrame by actual values
    df_sorted = df.sort_values(by='Actual')
    x_end = df_sorted.shape[0]  # Number of observations
    x = np.arange(0, x_end)  # X-axis values from 1 to number of observations
    y_start = min(df_sorted['Actual'].min(), df_sorted['Predicted'].min())
    y_end = max(df_sorted['Actual'].max(), df_sorted['Predicted'].max())
    y_dist = y_end - y_start
    y_start = y_start - 0.1 * y_dist
    y_end = y_end + 0.1 * y_dist
    x_end_2 = int(1.1 * x_end) if x_end > 0 else int(0.9 * x_end)
    x_start = int(-0.1 * x_end)
    # Plot the sorted data
    plt.figure(figsize=(10, 6))
    plt.plot(x, df_sorted['Predicted'], label='Predicted Values', color='red')
    plt.plot(x, df_sorted['Actual'], color='black', label='Actual Values')
    plt.title(f'{data_name} Linear Regression, {"80-20 Split" if validate else "In-Sample"}: yy black/actual vs. yp red/predicted')
    plt.legend()
    plt.ylim(y_start, y_end)
    plt.xlim(x_start, x_end_2)
    plt.savefig(f'statsmodels_{model_name}_{"80_20" if validate else "In_Sample"}.png')


def CV_Latex_Table(X, y, k, data_name):
    # Initialize lists to store the results of each fold
    rSq_list = []
    rSqBar_list = []
    sst_list = []
    sse_list = []
    sde_list = []
    mse0_list = []
    rmse_list = []
    mae_list = []
    smape_list = []

    # Initialize KFold with the specified number of splits
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Loop through each fold
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        # Split data
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        
        # Fit model
        model = sm.OLS(y_train, X_train).fit()
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics and append to lists
        ssr = np.sum((y_test - y_pred)**2)
        sst = np.sum((y_test - np.mean(y_test))**2)
        r_squared = 1 - (ssr / sst)
        r_squared_adj = 1 - (1 - r_squared) * (len(y_test) / (len(y_test) - X.shape[1]+1))
        sde = np.sqrt(ssr / (len(y_test) - X.shape[1]+1))
        mse0 = sst / len(y_test)
        rmse = np.sqrt(ssr / (len(y_test) - X.shape[1]+1))
        mae = np.mean(np.abs(y_test - y_pred))
        smape = np.mean(2 * np.abs((y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred)))) * 100

        rSq_list.append(r_squared)
        rSqBar_list.append(r_squared_adj)
        sst_list.append(sst)
        sse_list.append(ssr)
        sde_list.append(sde)
        mse0_list.append(mse0)
        rmse_list.append(rmse)
        mae_list.append(mae)
        smape_list.append(smape)
    
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Statsmodels - {data_name} Linear Regression CV}}")
    print(f"\\label{{tab:Statsmodels - {data_name} Linear Regression CV}}")
    print("\\begin{tabular}{|c|c|c|c|c|c|}\\hline")
    print("Name & In-num folds & min & max & mean & stdev \\\\ \\hline \\hline")
    print(f"rSq & {len(rSq_list)} & {min(rSq_list):.4f} & {max(rSq_list):.4f} & {np.mean(rSq_list):.4f} & {np.std(rSq_list):.4f} \\\\ \\hline")
    print(f"rSqBar & {len(rSqBar_list)} & {min(rSqBar_list):.4f} & {max(rSqBar_list):.4f} & {np.mean(rSqBar_list):.4f} & {np.std(rSqBar_list):.4f} \\\\ \\hline")
    print(f"sst & {len(sst_list)} & {min(sst_list):.4f} & {max(sst_list):.4f} & {np.mean(sst_list):.4f} & {np.std(sst_list):.4f} \\\\ \\hline")
    print(f"sse & {len(sse_list)} & {min(sse_list):.4f} & {max(sse_list):.4f} & {np.mean(sse_list):.4f} & {np.std(sse_list):.4f} \\\\ \\hline")
    print(f"sde & {len(sde_list)} & {min(sde_list):.4f} & {max(sde_list):.4f} & {np.mean(sde_list):.4f} & {np.std(sde_list):.4f} \\\\ \\hline")
    print(f"mse0 & {len(mse0_list)} & {min(mse0_list):.4f} & {max(mse0_list):.4f} & {np.mean(mse0_list):.4f} & {np.std(mse0_list):.4f} \\\\ \\hline")
    print(f"rmse & {len(rmse_list)} & {min(rmse_list):.4f} & {max(rmse_list):.4f} & {np.mean(rmse_list):.4f} & {np.std(rmse_list):.4f} \\\\ \\hline")
    print(f"mae & {len(mae_list)} & {min(mae_list):.4f} & {max(mae_list):.4f} & {np.mean(mae_list):.4f} & {np.std(mae_list):.4f} \\\\ \\hline")
    print(f"smape & {len(smape_list)} & {min(smape_list):.4f} & {max(smape_list):.4f} & {np.mean(smape_list):.4f} & {np.std(smape_list):.4f} \\\\ \\hline")
    print("\\end{tabular}")
    print("\\end{table}")


        





def LinRegAutoMPG():

    # Load the dataset
    Auto_MPG = pd.read_csv('auto_mpg_cleaned.csv')

    # Drop rows with missing values and the 'origin' column
    Auto_MPG = Auto_MPG.dropna()
    Auto_MPG = Auto_MPG.drop('origin', axis=1)

    # Define the independent variable (X) and the dependent variable (y)
    X = Auto_MPG[['displacement', 'cylinders', 'horsepower', 'weight', 'acceleration', 'model_year']]
    y = Auto_MPG['mpg']

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
    y_pred_split_test = reg_train.predict(X_test)

    # Get the number of observations (m) and the number of independent variables (k)
    k = X.shape[1] - 1  # Number of independent variables

    # Print the qof statistics in a LaTeX table format
    LatexTable(y, y_test, y_pred_in_sample, y_pred_split_test, k, "Auto MPG")

    print("-" * 88)
    print("-" * 88)
    print("-" * 88)

    CV_Latex_Table(X, y, 5, "Auto MPG")

    sorted_plot(y, y_pred_in_sample, "Auto MPG", validate=False)
    sorted_plot(y_test, y_pred_split_test, "Auto MPG", validate=True)
    





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
    y_pred_split_test = reg_train.predict(X_test)

    # Get the number of observations (m) and the number of independent variables (k)
    m = len(y)  # Number of observations
    k = X.shape[1] - 1  # Number of independent variables

    # Print the qof statistics in a LaTeX table format
    LatexTable(y, y_test, y_pred_in_sample, y_pred_split_test, k, "House Price")

    print("-" * 88)
    print("-" * 88)
    print("-" * 88)

    CV_Latex_Table(X, y, 5, "House Price")

    sorted_plot(y, y_pred_in_sample, "House Price", validate=False)
    sorted_plot(y_test, y_pred_split_test, "House Price", validate=True)




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
    y_pred_split_test = reg_train.predict(X_test)

    # Get the number of observations (m) and the number of independent variables (k)
    m = len(y)  # Number of observations
    k = X.shape[1] - 1  # Number of independent variables

    # Print the qof statistics in a LaTeX table format
    LatexTable(y, y_test, y_pred_in_sample, y_pred_split_test, k, "Insurance Charges")

    print("-" * 88)
    print("-" * 88)
    print("-" * 88)

    CV_Latex_Table(X, y, 5, "Insurance Charges")

    sorted_plot(y, y_pred_in_sample, "Insurance Charges", validate=False)
    sorted_plot(y_test, y_pred_split_test, "Insurance Charges", validate=True)




# LinRegAutoMPG()
# LinReghouse()
# LinReginsurance()