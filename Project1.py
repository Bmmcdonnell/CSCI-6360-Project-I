import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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




def InSample(X, y, data_name):
    k = X.shape[1] - 1  # Number of predictors (excluding the intercept)

    reg = sm.OLS(y, X).fit()
    regyp = reg.predict(X)
    reg_QoF = getQoF(y, regyp, k, reg)

    ridge = sm.OLS(y, X).fit_regularized(method='elastic_net', L1_wt=0, alpha=1.0)
    ridgeyp = ridge.predict(X)
    ridge_QoF = getQoF(y, ridgeyp, k, ridge)

    lasso = sm.OLS(y, X).fit_regularized(method='elastic_net', L1_wt=1, alpha=1.0)
    lassoyp = lasso.predict(X)
    lasso_QoF = getQoF(y, lassoyp, k, lasso)

    Sqrtf = sm.OLS(np.sqrt(y), X).fit()
    Sqrtfyp_tran = Sqrtf.predict(X)
    Sqrtfyp = Sqrtfyp_tran ** 2
    Sqrtf_QoF = getQoF(y, Sqrtfyp, k, Sqrtf)

    log1p = sm.OLS(np.log1p(y), X).fit()
    log1pyp_tran = log1p.predict(X)
    log1pyp = np.expm1(log1pyp_tran)
    log1p_QoF = getQoF(y, log1pyp, k, log1p)

    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Statsmodels - {data_name} In-Sample QoF Comparison}}")
    print(f"\\label{{tab:Statsmodels - {data_name} In-Sample QoF Comparison}}")
    print("\\begin{tabular}{|c|c|c|c|c|c|}\\hline")
    print("Metric & Regression & Ridge & Lasso & Sqrt & Log1p \\\\ \\hline \\hline")
    print(f"rSq & {reg_QoF[0]:.4f} & {ridge_QoF[0]:.4f} & {lasso_QoF[0]:.4f} & {Sqrtf_QoF[0]:.4f} & {log1p_QoF[0]:.4f} \\\\ \\hline")
    print(f"rSqBar & {reg_QoF[1]:.4f} & {ridge_QoF[1]:.4f} & {lasso_QoF[1]:.4f} & {Sqrtf_QoF[1]:.4f} & {log1p_QoF[1]:.4f} \\\\ \\hline")
    print(f"sst & {reg_QoF[2]:.4f} & {ridge_QoF[2]:.4f} & {lasso_QoF[2]:.4f} & {Sqrtf_QoF[2]:.4f} & {log1p_QoF[2]:.4f} \\\\ \\hline")
    print(f"sse & {reg_QoF[3]:.4f} & {ridge_QoF[3]:.4f} & {lasso_QoF[3]:.4f} & {Sqrtf_QoF[3]:.4f} & {log1p_QoF[3]:.4f} \\\\ \\hline")
    print(f"sde & {reg_QoF[4]:.4f} & {ridge_QoF[4]:.4f} & {lasso_QoF[4]:.4f} & {Sqrtf_QoF[4]:.4f} & {log1p_QoF[4]:.4f} \\\\ \\hline")
    print(f"mse0 & {reg_QoF[5]:.4f} & {ridge_QoF[5]:.4f} & {lasso_QoF[5]:.4f} & {Sqrtf_QoF[5]:.4f} & {log1p_QoF[5]:.4f} \\\\ \\hline")
    print(f"rmse & {reg_QoF[6]:.4f} & {ridge_QoF[6]:.4f} & {lasso_QoF[6]:.4f} & {Sqrtf_QoF[6]:.4f} & {log1p_QoF[6]:.4f} \\\\ \\hline")
    print(f"mae & {reg_QoF[7]:.4f} & {ridge_QoF[7]:.4f} & {lasso_QoF[7]:.4f} & {Sqrtf_QoF[7]:.4f} & {log1p_QoF[7]:.4f} \\\\ \\hline")
    print(f"smape & {reg_QoF[8]:.4f} & {ridge_QoF[8]:.4f} & {lasso_QoF[8]:.4f} & {Sqrtf_QoF[8]:.4f} & {log1p_QoF[8]:.4f} \\\\ \\hline")
    print(f"m & {reg_QoF[9]:.4f} & {ridge_QoF[9]:.4f} & {lasso_QoF[9]:.4f} & {Sqrtf_QoF[9]:.4f} & {log1p_QoF[9]:.4f} \\\\ \\hline")
    print(f"dfr & {reg_QoF[10]:.4f} & {ridge_QoF[10]:.4f} & {lasso_QoF[10]:.4f} & {Sqrtf_QoF[10]:.4f} & {log1p_QoF[10]:.4f}\\\\ \\hline")
    print(f"df & {reg_QoF[11]:.4f} & {ridge_QoF[11]:.4f} & {lasso_QoF[11]:.4f} & {Sqrtf_QoF[11]:.4f} & {log1p_QoF[11]:.4f}\\\\ \\hline")
    print(f"fStat & {reg_QoF[12]:.4f} & {ridge_QoF[12]:.4f} & {lasso_QoF[12]:.4f} & {Sqrtf_QoF[12]:.4f} &{log1p_QoF[12]:.4f}\\\\ \\hline")
    print(f"aic &{reg_QoF[13]:.4f} &{ridge_QoF[13]:.4f} & {lasso_QoF[13]:.4f} & {Sqrtf_QoF[13]:.4f} & {log1p_QoF[13]:.4f}\\\\ \\hline")
    print(f"bic & {reg_QoF[14]:.4f} & {ridge_QoF[14]:.4f} & {lasso_QoF[14]:.4f} & {Sqrtf_QoF[14]:.4f} & {log1p_QoF[14]:.4f}\\\\ \\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    return (regyp, ridgeyp, lassoyp, Sqrtfyp, log1pyp)



def P1_AutoMPG_IS():
    Auto_MPG = pd.read_csv('auto_mpg_cleaned.csv')
    Auto_MPG = Auto_MPG.dropna()
    Auto_MPG = Auto_MPG.drop('origin', axis=1)
    X = Auto_MPG[['displacement', 'cylinders', 'horsepower', 'weight', 'acceleration', 'model_year']]
    y = Auto_MPG['mpg']
    X = sm.add_constant(X)
    yp = InSample(X, y, "Auto MPG")
    save_sorted_plot(y, yp[0], "Auto MPG", "reg", validate=False)
    save_sorted_plot(y, yp[1], "Auto MPG", "ridge", validate=False)
    save_sorted_plot(y, yp[2], "Auto MPG", "lasso", validate=False)
    save_sorted_plot(y, yp[3], "Auto MPG", "sqrt", validate=False)
    save_sorted_plot(y, yp[4], "Auto MPG", "log1p", validate=False)



def P1_Housing_IS():
    House_Price = pd.read_csv('house_price_regression_dataset.csv')
    X = House_Price[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']]
    y = House_Price['House_Price']
    X = sm.add_constant(X)
    yp = InSample(X, y, "House Price")
    save_sorted_plot(y, yp[0], "House Price", "reg", validate=False)
    save_sorted_plot(y, yp[1], "House Price", "ridge", validate=False)
    save_sorted_plot(y, yp[2], "House Price", "lasso", validate=False)
    save_sorted_plot(y, yp[3], "House Price", "sqrt", validate=False)
    save_sorted_plot(y, yp[4], "House Price", "log1p", validate=False)



def P1_Insuarance_IS():
    Insurance_Charges = pd.read_csv('insurance_cat2num.csv')
    X = Insurance_Charges[['intercept', 'age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']]
    y = Insurance_Charges['charges']
    yp = InSample(X, y, "Insurance Charges")
    save_sorted_plot(y, yp[0], "Insurance Charges", "reg", validate=False)
    save_sorted_plot(y, yp[1], "Insurance Charges", "ridge", validate=False)
    save_sorted_plot(y, yp[2], "Insurance Charges", "lasso", validate=False)
    save_sorted_plot(y, yp[3], "Insurance Charges", "sqrt", validate=False)
    save_sorted_plot(y, yp[4], "Insurance Charges", "log1p", validate=False)




# P1_AutoMPG_IS()
# P1_Housing_IS()
# P1_Insuarance_IS()




def OutOfSample(X, y, data_name, testp=0.2, random_state=42):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testp, random_state=random_state)

    k = X.shape[1] - 1  # Number of predictors (excluding the intercept)

    reg = sm.OLS(y_train, X_train).fit()
    regyp = reg.predict(X_test)
    reg_QoF = getQoF(y_test, regyp, k, None)

    ridge = sm.OLS(y_train, X_train).fit_regularized(method='elastic_net', L1_wt=0, alpha=1.0)
    ridgeyp = ridge.predict(X_test)
    ridge_QoF = getQoF(y_test, ridgeyp, k, None)

    lasso = sm.OLS(y_train, X_train).fit_regularized(method='elastic_net', L1_wt=1, alpha=1.0)
    lassoyp = lasso.predict(X_test)
    lasso_QoF = getQoF(y_test, lassoyp, k, None)

    Sqrtf = sm.OLS(np.sqrt(y_train), X_train).fit()
    Sqrtfyp_tran = Sqrtf.predict(X_test)
    Sqrtfyp = Sqrtfyp_tran ** 2
    Sqrtf_QoF = getQoF(y_test, Sqrtfyp, k, None)

    log1p = sm.OLS(np.log1p(y_train), X_train).fit()
    log1pyp_tran = log1p.predict(X_test)
    log1pyp = np.expm1(log1pyp_tran)
    log1p_QoF = getQoF(y_test, log1pyp, k, None)

    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Statsmodels - {data_name} Out-of-Sample QoF Comparison}}")
    print(f"\\label{{tab:Statsmodels - {data_name} Out-of-Sample QoF Comparison}}")
    print("\\begin{tabular}{|c|c|c|c|c|c|}\\hline")
    print("Metric & Regression & Ridge & Lasso & Sqrt & Log1p \\\\ \\hline \\hline")
    print(f"rSq & {reg_QoF[0]:.4f} & {ridge_QoF[0]:.4f} & {lasso_QoF[0]:.4f} & {Sqrtf_QoF[0]:.4f} & {log1p_QoF[0]:.4f} \\\\ \\hline")
    print(f"rSqBar & {reg_QoF[1]:.4f} & {ridge_QoF[1]:.4f} & {lasso_QoF[1]:.4f} & {Sqrtf_QoF[1]:.4f} & {log1p_QoF[1]:.4f} \\\\ \\hline")
    print(f"sst & {reg_QoF[2]:.4f} & {ridge_QoF[2]:.4f} & {lasso_QoF[2]:.4f} & {Sqrtf_QoF[2]:.4f} & {log1p_QoF[2]:.4f} \\\\ \\hline")
    print(f"sse & {reg_QoF[3]:.4f} & {ridge_QoF[3]:.4f} & {lasso_QoF[3]:.4f} & {Sqrtf_QoF[3]:.4f} & {log1p_QoF[3]:.4f} \\\\ \\hline")
    print(f"sde & {reg_QoF[4]:.4f} & {ridge_QoF[4]:.4f} & {lasso_QoF[4]:.4f} & {Sqrtf_QoF[4]:.4f} & {log1p_QoF[4]:.4f} \\\\ \\hline")
    print(f"mse0 & {reg_QoF[5]:.4f} & {ridge_QoF[5]:.4f} & {lasso_QoF[5]:.4f} & {Sqrtf_QoF[5]:.4f} & {log1p_QoF[5]:.4f} \\\\ \\hline")
    print(f"rmse & {reg_QoF[6]:.4f} & {ridge_QoF[6]:.4f} & {lasso_QoF[6]:.4f} & {Sqrtf_QoF[6]:.4f} & {log1p_QoF[6]:.4f} \\\\ \\hline")
    print(f"mae & {reg_QoF[7]:.4f} & {ridge_QoF[7]:.4f} & {lasso_QoF[7]:.4f} & {Sqrtf_QoF[7]:.4f} & {log1p_QoF[7]:.4f} \\\\ \\hline")
    print(f"smape & {reg_QoF[8]:.4f} & {ridge_QoF[8]:.4f} & {lasso_QoF[8]:.4f} & {Sqrtf_QoF[8]:.4f} & {log1p_QoF[8]:.4f} \\\\ \\hline")
    print(f"m & {reg_QoF[9]:.4f} & {ridge_QoF[9]:.4f} & {lasso_QoF[9]:.4f} & {Sqrtf_QoF[9]:.4f} & {log1p_QoF[9]:.4f} \\\\ \\hline")
    print(f"dfr & {reg_QoF[10]:.4f} & {ridge_QoF[10]:.4f} & {lasso_QoF[10]:.4f} & {Sqrtf_QoF[10]:.4f} & {log1p_QoF[10]:.4f}\\\\ \\hline")
    print(f"df & {reg_QoF[11]:.4f} & {ridge_QoF[11]:.4f} & {lasso_QoF[11]:.4f} & {Sqrtf_QoF[11]:.4f} & {log1p_QoF[11]:.4f}\\\\ \\hline")
    print(f"fStat & {reg_QoF[12]:.4f} & {ridge_QoF[12]:.4f} & {lasso_QoF[12]:.4f} & {Sqrtf_QoF[12]:.4f} &{log1p_QoF[12]:.4f}\\\\ \\hline")
    print(f"aic &{reg_QoF[13]:.4f} &{ridge_QoF[13]:.4f} & {lasso_QoF[13]:.4f} & {Sqrtf_QoF[13]:.4f} & {log1p_QoF[13]:.4f}\\\\ \\hline")
    print(f"bic & {reg_QoF[14]:.4f} & {ridge_QoF[14]:.4f} & {lasso_QoF[14]:.4f} & {Sqrtf_QoF[14]:.4f} & {log1p_QoF[14]:.4f}\\\\ \\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    return (regyp, ridgeyp, lassoyp, Sqrtfyp, log1pyp, y_test)




def P1_AutoMPG_OOS():
    Auto_MPG = pd.read_csv('auto_mpg_cleaned.csv')
    Auto_MPG = Auto_MPG.dropna()
    Auto_MPG = Auto_MPG.drop('origin', axis=1)
    X = Auto_MPG[['displacement', 'cylinders', 'horsepower', 'weight', 'acceleration', 'model_year']]
    y = Auto_MPG['mpg']
    X = sm.add_constant(X)
    yp = OutOfSample(X, y, "Auto MPG", testp=0.2, random_state=42)
    save_sorted_plot(yp[5], yp[0], "Auto MPG", "reg", validate=True)
    save_sorted_plot(yp[5], yp[1], "Auto MPG", "ridge", validate=True)
    save_sorted_plot(yp[5], yp[2], "Auto MPG", "lasso", validate=True)
    save_sorted_plot(yp[5], yp[3], "Auto MPG", "sqrt", validate=True)
    save_sorted_plot(yp[5], yp[4], "Auto MPG", "log1p", validate=True)



def P1_Housing_OOS():
    House_Price = pd.read_csv('house_price_regression_dataset.csv')
    X = House_Price[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']]
    y = House_Price['House_Price']
    X = sm.add_constant(X)
    yp = OutOfSample(X, y, "House Price", testp=0.2, random_state=42)
    save_sorted_plot(yp[5], yp[0], "House Price", "reg", validate=True)
    save_sorted_plot(yp[5], yp[1], "House Price", "ridge", validate=True)
    save_sorted_plot(yp[5], yp[2], "House Price", "lasso", validate=True)
    save_sorted_plot(yp[5], yp[3], "House Price", "sqrt", validate=True)
    save_sorted_plot(yp[5], yp[4], "House Price", "log1p", validate=True)



def P1_Insuarance_OOS():
    Insurance_Charges = pd.read_csv('insurance_cat2num.csv')
    X = Insurance_Charges[['intercept', 'age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']]
    y = Insurance_Charges['charges']
    yp = OutOfSample(X, y, "Insurance Charges", testp=0.2, random_state=42)
    save_sorted_plot(yp[5], yp[0], "Insurance Charges", "reg", validate=True)
    save_sorted_plot(yp[5], yp[1], "Insurance Charges", "ridge", validate=True)
    save_sorted_plot(yp[5], yp[2], "Insurance Charges", "lasso", validate=True)
    save_sorted_plot(yp[5], yp[3], "Insurance Charges", "sqrt", validate=True)
    save_sorted_plot(yp[5], yp[4], "Insurance Charges", "log1p", validate=True)




# P1_AutoMPG_OOS()
# P1_Housing_OOS()
# P1_Insuarance_OOS()