import numpy as np
from scipy.stats import norm
from math import log, sqrt, exp, pi

class Opcion:
    def __init__(self,s,k,r,sigma,T, kind):
        """ s = spot, k = strike , r = tasa (decimal), sigma = volatilidad (decimal), T = dias """
        self.s = float(s)
        self.k = float(k)
        self.r = float(r)
        self.sigma = float(sigma)
        self.T = float(T/360)
        self.kind = kind
        self.d1 =( log(self.s/self.k) + ( self.r + (self.sigma**2 /2) )*self.T ) /   (self.sigma * sqrt(self.T) )
        self.d2 = self.d1 - (self.sigma * sqrt(self.T))
        self.discount = exp(-(self.r) * self.T)
    
    def value(self):
        if (self.kind == "call"):
            call = self.s*norm.cdf(self.d1) - self.k*self.discount*norm.cdf(self.d2)
            return(call)
        else:
            put = self.k*self.discount*norm.cdf(-self.d2) - self.s*norm.cdf(-self.d1)
            return(put)

    def delta(self):
        if (self.kind == "call"):
            return(norm.cdf(self.d1))
        else:
            return(norm.cdf(self.d1)-1)

    def gamma(self):
        gamma = exp((-self.d1 ** 2)/2)  / (self.s * self.sigma * sqrt(self.T) * 2 * pi)
        return(gamma)
    
    def vega(self):
        vega = self.s * sqrt(self.T) * exp((-self.d1 ** 2)/2) / sqrt(2 * pi)
        return(vega)
    
    def valuation_curve(self):
    #Establish uper and lower limit for range
        if self.s > self.k:
            liminf = 0.9 * self.s
            limsup = 1.1 * self.k
        else:
            liminf = 0.9 * self.k
            limsup = 1.1 * self.s
        rango = np.arange(start = liminf, stop = limsup, step = self.s/100).tolist()
        valores = []

        for i in rango:
            x = Opcion(s = i,k = self.k ,r = self.r ,sigma = self.sigma, T = self.T, kind = self.kind)
            valores.append(x.value())
        val_cur = [rango, valores]
        return val_cur



a = Opcion(s = 50.6 ,k = 50 ,r = 0.05 ,sigma = 0.39 ,T = 3, kind = "put")
print(a.value())
print(a.delta())
print(a.gamma())
print(a.vega())
res = a.valuation_curve()
import matplotlib.pyplot as plt
plt.plot(res[0], res[1])
plt.show()