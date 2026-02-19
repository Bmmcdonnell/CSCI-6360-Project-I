from wsgiref import validate
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def getQoF(y_actual, y_pred, k, mod):
    # Get the number of observations (m)
    m = len(y_actual)  # Number of observations in the original dataset

    # Calculate SSE (Sum of Squared Errors)
    try:
        sse = mod.ssr
    except:
        sse = np.sum((y_actual - y_pred)**2)

    # Calculate SST (Total Sum of Squares)
    try:
        sst = mod.centered_tss
    except:
        sst= np.sum((y_actual - np.mean(y_actual))**2)

    # Compute R-squared
    try:
        rsquared = mod.rsquared
    except:
        rsquared = 1 - (sse / sst)

    # Calculate Adjusted R-squared
    try:
        rsquared_adj = mod.rsquared_adj
    except:
        rsquared_adj = 1 - (1 - rsquared) * (m - 1) / (m - k)

    # Standard Deviation of the Errors (SDE)
    try:
        sde = np.sqrt(mod.mse_resid)
    except:
        sde = np.sqrt(sse / (m - k))

    # Calculate Mean Squared Error
    try:
        mse0 = mod.mse_resid
    except:
        mse0 = sse / m

    # Root Mean Squared Error
    try:
        rmse = np.sqrt(mod.mse_resid)
    except:
        rmse = np.sqrt(mse0)

    # Mean Absolute Error
    try:
        mae = mod.mae
    except:
        mae = np.mean(np.abs(y_actual - y_pred))

    # Symmetric Mean Absolute Percentage Error (SMAPE)
    try:
        smape = mod.smape
    except:
        smape = np.mean(2 * np.abs(y_actual - y_pred) / (np.abs(y_actual) + np.abs(y_pred))) * 100

    # Calculate F-statistic
    try:
        f_stat = mod.fvalue
    except:
        f_stat = ((sst - sse) / k) / (sse / (m - k))

    # Get the degree of freedom for the model
    try:
        dfr = mod.df_model
    except:
        dfr = k

    # Get the degree of freedom for the residuals
    try:
        df = mod.df_resid
    except:
        df = m - k

    # Calculate AIC
    try:
        aic = mod.aic
    except:
        aic = m * np.log(mse0) + 2 * k

    # Calculate BIC
    try:
        bic = mod.bic
    except:
        bic = m * np.log(mse0) + k * np.log(m)

    QoF = list(range(15))

    QoF[0] = rsquared
    QoF[1] = rsquared_adj
    QoF[2] = sst
    QoF[3] = sse
    QoF[4] = sde
    QoF[5] = mse0
    QoF[6] = rmse
    QoF[7] = mae
    QoF[8] = smape
    QoF[9] = m
    QoF[10] = dfr
    QoF[11] = df
    QoF[12] = f_stat
    QoF[13] = aic
    QoF[14] = bic

    return QoF


def save_sorted_plot(y_actual, y_pred, data_name, model_name, validate=False):
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

def Compare_IS_OOS(X, y, data_name, testp=0.2, random_state=42):
    k = X.shape[1] - 1  # Number of predictors (excluding the intercept)

    reg_IS = sm.OLS(y, X).fit()
    reg_IS_yp = reg_IS.predict(X)
    reg_IS_QoF = getQoF(y, reg_IS_yp, k, reg_IS)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testp, random_state=random_state)

    reg_OOS = sm.OLS(y_train, X_train).fit()
    reg_OOS_yp = reg_OOS.predict(X_test)
    reg_OOS_QoF = getQoF(y_test, reg_OOS_yp, k, None)

    # Print the results in a LaTeX table format
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Statsmodels - {data_name} Linear Regression}}")
    print(f"\\label{{tab:Statsmodels - {data_name} Linear Regression}}")
    print("\\begin{tabular}{|c|c|c|}\\hline")
    print("Metric & In-Sample & 80-20 Split \\\\ \\hline \\hline")
    print(f"rSq & {reg_IS_QoF[0]:.4f} & {reg_OOS_QoF[0]:.4f} \\\\ \\hline")
    print(f"rSqBar & {reg_IS_QoF[1]:.4f} & {reg_OOS_QoF[1]:.4f} \\\\ \\hline")
    print(f"sst & {reg_IS_QoF[2]:.4f} & {reg_OOS_QoF[2]:.4f} \\\\ \\hline")
    print(f"sse & {reg_IS_QoF[3]:.4f} & {reg_OOS_QoF[3]:.4f} \\\\ \\hline")
    print(f"sde & {reg_IS_QoF[4]:.4f} & {reg_OOS_QoF[4]:.4f} \\\\ \\hline")
    print(f"mse0 & {reg_IS_QoF[5]:.4f} & {reg_OOS_QoF[5]:.4f} \\\\ \\hline")
    print(f"rmse & {reg_IS_QoF[6]:.4f} & {reg_OOS_QoF[6]:.4f} \\\\ \\hline")
    print(f"mae & {reg_IS_QoF[7]:.4f} & {reg_OOS_QoF[7]:.4f} \\\\ \\hline")
    print(f"smape & {reg_IS_QoF[8]:.4f} & {reg_OOS_QoF[8]:.4f} \\\\ \\hline")
    print(f"m & {reg_IS_QoF[9]:.4f} & {reg_OOS_QoF[9]:.4f} \\\\ \\hline")
    print(f"dfr & {reg_IS_QoF[10]:.4f} & {reg_OOS_QoF[10]:.4f} \\\\ \\hline")
    print(f"df & {reg_IS_QoF[11]:.4f} & {reg_OOS_QoF[11]:.4f} \\\\ \\hline")
    print(f"fStat & {reg_IS_QoF[12]:.4f} & {reg_OOS_QoF[12]:.4f} \\\\ \\hline")
    print(f"aic & {reg_IS_QoF[13]:.4f} & {reg_OOS_QoF[13]:.4f} \\\\ \\hline")
    print(f"bic & {reg_IS_QoF[14]:.4f} & {reg_OOS_QoF[14]:.4f} \\\\ \\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    return (reg_IS_yp, reg_OOS_yp, y_test)



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

    Compare_IS_OOS(X, y, "Auto MPG", testp=0.2, random_state=42)
    



def LinReghouse():

    # Load the dataset
    House_Price = pd.read_csv('house_price_regression_dataset.csv')

    # Define the independent variable (X) and the dependent variable (y)
    X = House_Price[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']]
    y = House_Price['House_Price']

    # Add a constant to the independent variables (for the intercept)
    X = sm.add_constant(X)

    Compare_IS_OOS(X, y, "House Price", testp=0.2, random_state=42)




def LinReginsurance():

    # Load the dataset
    Insurance_Charges = pd.read_csv('insurance_cat2num.csv')

    # Define the independent variable (X) and the dependent variable (y)
    X = Insurance_Charges[['intercept', 'age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']]
    y = Insurance_Charges['charges']

    Compare_IS_OOS(X, y, "Insurance Charges", testp=0.2, random_state=42)




# LinRegAutoMPG()
# LinReghouse()
# LinReginsurance()



def CV_Latex_Table(X, y, num_folds=5, data_name=None):
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
    m_list = []
    dfr_list = []
    df_list = []
    fStat_list = []
    aic_list = []
    bic_list = []

    k = X.shape[1] - 1  # Number of predictors (excluding the intercept)

    # Initialize KFold with the specified number of splits
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

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
        Qof = getQoF(y_test, y_pred, k, None)

        
        rSq_list.append(Qof[0])
        rSqBar_list.append(Qof[1])
        sst_list.append(Qof[2])
        sse_list.append(Qof[3])
        sde_list.append(Qof[4])
        mse0_list.append(Qof[5])
        rmse_list.append(Qof[6])
        mae_list.append(Qof[7])
        smape_list.append(Qof[8])
        m_list.append(Qof[9])
        dfr_list.append(Qof[10])
        df_list.append(Qof[11])
        fStat_list.append(Qof[12])
        aic_list.append(Qof[13])
        bic_list.append(Qof[14])

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
    print(f"m & {len(m_list)} & {min(m_list):.4f} & {max(m_list):.4f} & {np.mean(m_list):.4f} & {np.std(m_list):.4f} \\\\ \\hline")
    print(f"dfr & {len(dfr_list)} & {min(dfr_list):.4f} & {max(dfr_list):.4f} & {np.mean(dfr_list):.4f} & {np.std(dfr_list):.4f} \\\\ \\hline")
    print(f"df & {len(df_list)} & {min(df_list):.4f} & {max(df_list):.4f} & {np.mean(df_list):.4f} & {np.std(df_list):.4f} \\\\ \\hline")
    print(f"fStat & {len(fStat_list)} & {min(fStat_list):.4f} & {max(fStat_list):.4f} & {np.mean(fStat_list):.4f} & {np.std(fStat_list):.4f} \\\\ \\hline")
    print(f"aic & {len(aic_list)} & {min(aic_list):.4f} & {max(aic_list):.4f} & {np.mean(aic_list):.4f} & {np.std(aic_list):.4f} \\\\ \\hline")
    print(f"bic & {len(bic_list)} & {min(bic_list):.4f} & {max(bic_list):.4f} & {np.mean(bic_list):.4f} & {np.std(bic_list):.4f} \\\\ \\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def AutoMPG_CV():
    Auto_MPG = pd.read_csv('auto_mpg_cleaned.csv')
    Auto_MPG = Auto_MPG.dropna()
    Auto_MPG = Auto_MPG.drop('origin', axis=1)
    X = Auto_MPG[['displacement', 'cylinders', 'horsepower', 'weight', 'acceleration', 'model_year']]
    y = Auto_MPG['mpg']
    X = sm.add_constant(X)
    CV_Latex_Table(X, y, 5, "Auto MPG")
    



def house_CV():
    House_Price = pd.read_csv('house_price_regression_dataset.csv')
    X = House_Price[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']]
    y = House_Price['House_Price']
    X = sm.add_constant(X)
    CV_Latex_Table(X, y, 5, "House Price")



def insurance_CV():
    Insurance_Charges = pd.read_csv('insurance_cat2num.csv')
    X = Insurance_Charges[['intercept', 'age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']]
    y = Insurance_Charges['charges']
    CV_Latex_Table(X, y, 5, "Insurance Charges")



# AutoMPG_CV()
# house_CV()
# insurance_CV()