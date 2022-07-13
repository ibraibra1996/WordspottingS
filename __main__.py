import numpy as np
import pandas as pd

from sklearn import linear_model
from scipy.stats import spearmanr
def main():
    df_pie = pd.read_csv('ResultData.csv',
                         sep=';')

    print('Summary:')
    print(df_pie[['n_centroids','step_size','cell_size','totalAvg']].describe())




    print(' \nBedingt')
    print(df_pie[['totalAvg','n_centroids']].groupby("n_centroids").mean())

    print('\nBedingt ')
    print(df_pie[['totalAvg','step_size']].groupby("step_size").mean())

    print('\nBedingt ')
    print(df_pie[['totalAvg','cell_size']].groupby("cell_size").mean())


    print('\nBedingt ')
    print(df_pie[['totalAvg','dataName']].groupby("dataName").mean())


    print('\n')
    print('Correlation:')
    rho, pval=spearmanr(df_pie['totalAvg'],df_pie[['n_centroids','step_size','cell_size']])
    print(rho[0])

    print('\n')
    print('LinearRegression:')
    X = df_pie[['n_centroids','step_size','cell_size']]
    y = df_pie['totalAvg']
    # n_centroids;step_size;cell_size;totalAvg
    # Initialize model from sklearn and fit it into our data
    regr = linear_model.LinearRegression()
    model = regr.fit(X, y)


    print('Intercept:', model.intercept_)
    print('Coefficients:', model.coef_)

if __name__ == "__main__":
    main()
