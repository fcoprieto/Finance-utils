import pandas as pd
import numpy as np
import datetime
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression

"""
Based on Avellaneda and Lee "Statistical arbitrage in the US equities market"
"""

def demean(x):
    deme = x - x.mean()
    s = x.std()
    return (deme/s)


class PCA_pairs:

    """
    Take a pandas dataframe with daily returns as input
    """
    def __init__(self, asset_returns):
        self.asset_returns = asset_returns
        self.normed_returns = asset_returns.apply(demean, axis = 1)
        self.stock_tickers = self.normed_returns.columns.values
        self.n_tickers = len(self.stock_tickers)
        self.pc_w = pd.DataFrame(index=self.stock_tickers)

    def eigenret(self, n, inicio, fin, cov_matrix, cov_matrix_raw):
        eigen_prtf = pd.DataFrame(index = self.stock_tickers)
        eigen_prtf_returns = pd.DataFrame()
        pca = PCA()
        pca.fit(cov_matrix)
        pcs = pca.components_
        std = np.diag(cov_matrix_raw)
        for i in range(n):
            self.pc_w = pcs[i]/std
            self.pc_w = (self.pc_w/self.pc_w.sum())
            eigen_prtf['port_'+ str(i)] =  np.nan_to_num(self.pc_w.squeeze()*100) #It is multiply by 100 to deal with very small numbers that would be transformed to 0 or NaN. This explains the divition by 100 in the following line
            eigen_prtf_returns['port_'+ str(i)] = np.dot(self.asset_returns[inicio : fin], eigen_prtf['port_'+ str(i)] / 100)
        return(eigen_prtf_returns, self.pc_w.T)
    
    def pca(self, n, tiempo, offset):

        """
        # n: Number of PCA portfolios to use
        tiempo: Pandas daterange specifing the time frame to use
        offset: Number of month taken to calculate the covariance matrix

        returns a pandas dataframe with the signals for each stock each day
        """

        signals = pd.DataFrame(columns = self.normed_returns.columns, index = tiempo)
        for i in tiempo:
            inicio = i - pd.DateOffset(months = offset)
            cov_matrixa = self.normed_returns[inicio:i].cov()
            cov_matrixr = self.asset_returns[inicio:i].cov()
            eigen_prtf_returns, self.pc_w = self.eigenret(n, cov_matrix = cov_matrixa, inicio = inicio, fin = i, cov_matrix_raw = cov_matrixr) # Choose number of eigenportfolios to use and create returns
            eigen_prtf_returns = eigen_prtf_returns.set_index(self.normed_returns[inicio:i].index) #Dates as index
            x = eigen_prtf_returns[inicio:i].replace([np.inf, -np.inf], np.nan).fillna(0) # Initialize x replace +/- infs with nan and then to 0s
            for s in self.stock_tickers:
                # Regression of each stock against each eigenportfolio
                y = self.normed_returns[s][inicio:i].replace([np.inf, -np.inf], np.nan).fillna(0)
                reg = Ridge(alpha = 0.5).fit(X= x, y = y)    # alternative: LinearRegression().fit(X= x, y = y) 
            
                # Regression errors
                errors = y - reg.predict(x)
                ye = errors.shift(1)
            
                ### Regression against the errors
                rege = LinearRegression().fit(X= errors[1:].values.reshape(-1,1), y = ye[1:])       #Alternative: Ridge(alpha = 0.5).fit(X= errors[1:].values.reshape(-1,1), y = ye[1:])
                a = rege.intercept_
                b = rege.coef_
                errors2 = ye[1:] - rege.predict(errors[1:].values.reshape(-1,1))
                epsi = errors2.std()
                # Calculate signals
                signal =( -a * (1-b**2)**0.5 ) / ( (1-b) * (epsi) ) * 100
                signals[s][i] = float(signal)
        return signals