import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Capm:
    def __init__(self, df):
        self.df = df
        self.ivec = np.ones(len(self.df.columns)).tolist()
        self.sigma = self.df.cov()*252
        self.sigma = self.sigma.values
        self.vinv = np.linalg.inv(self.sigma)

        #Vector de volatilidad y retorno anualizado y transformado a numpy
        self.sigmavec = np.diag(self.sigma) ** 0.5
        self.muvec = self.df.mean(axis = 0).fillna(-1000)
        self.muvec = self.muvec.values * 252

    def minvar(self):
        wmvp = np.matmul(self.vinv, self.ivec)
        wmvp = wmvp/wmvp.sum()
        mumvp = np.matmul(wmvp, self.muvec)
        sdmvp = (np.matmul(np.matmul(wmvp.transpose(), self.sigma), wmvp))**0.5
        return [mumvp,sdmvp]
    
    def tang(self):
        wtan  = np.matmul(self.vinv, self.muvec)
        wtan  = wtan/sum(wtan)
        mutan = np.matmul(wtan.transpose(), self.muvec)
        sdtan = (np.matmul(np.matmul(wtan.transpose(), self.sigma), wtan))**0.5
        return [mutan, sdtan]

    def frontier(self):
        AA  = self.vinv.sum()
        BB = np.matmul(np.matmul(self.muvec.transpose(), self.vinv), self.ivec)
        CC = np.matmul(np.matmul(self.muvec.transpose(), self.vinv), self.muvec)
        DEL = AA*CC-BB**2

        mufrontb = np.linspace(start = self.minvar()[0],stop = max(self.muvec.max()+0.3,self.tang()[0]),num=200)
        sdfrontb = ((AA*mufrontb**2-2*BB*mufrontb + CC)/DEL)**0.5

        return mufrontb, sdfrontb

    def graph(self):
        mu, sd = self.frontier()
        plt.scatter(x=sd, y = mu)
        plt.scatter(x=self.sigmavec, y = self.muvec)
        for i, txt in enumerate(self.df.columns):
            plt.annotate(txt, (self.sigmavec[i], self.muvec[i]))
        plt.show()