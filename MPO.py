from __future__ import division
import pandas_datareader as pdr
from pandas_datareader.quandl import QuandlReader 
from collections import defaultdict
import numpy as np
from numpy import dot, sqrt
import pandas as pd
import datetime
from random import uniform
from scipy.stats import linregress
from scipy.optimize import minimize, fsolve
from scipy.interpolate import splrep, splev
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import re

"""
Mess:
    Market returns in pct: 
        self.market_returns = self.data[self.market_indecies].pct_change().mean(axis=0)
    Covariance in pct: 
        self.cov_matrix = self.data.pct_change().cov() 
"""

"""
TODO:
 - generate and plot sharpe-ratio of a whole bunch of random portfolio weight combinations
 - calculate and plot capital market line
    - page 333 in python finance book
    - First derivative of efficient frontier??
- ADD required return
- ADD sanity check to see if wndow size and window move matches

"""

class Calcualtion_pack():
    "L4NT0W"
    
    def __init__(self, stock_ticks=["AAPL","MMM"], market_indecies=['^GSPC',"^DJI"], 
                 start=datetime.datetime(1997, 1, 1), end=datetime.date.today(), 
                 risk_free_rate= 0.02, shorting_allowed=False, 
                 window_size=None, window_move=None, source="quandl"):

        self.stock_ticks = stock_ticks
        self.market_indecies = market_indecies #Concatenate different markets indecies to get "real" market
                                                # Possibly do this with different weights
        self.start = start
        self.end = end
        self.risk_free_rate = risk_free_rate
        self.shorting_allowed=shorting_allowed
        self.window_move = window_move
        self.window_size = window_size
        self.source = source

    def get_monthly_data(self):

        if self.source in ["google", "yahoo"]: #depricated 
            raw_data = pdr.DataReader(self.market_indecies + self.stock_ticks, 
                                      self.source, self.start, self.end)
            adj_data = raw_data["Close"]

        if self.source == "quandl":
            adj_data = defaultdict()
            for ticker in self.stock_ticks + self.market_indecies:
                data = QuandlReader(symbols="WIKI/{}".format(ticker), start=self.start, end=self.end).read()
                adj_data[ticker] =  data["AdjClose"]

            adj_data = pd.DataFrame(adj_data)

        self.data = adj_data.groupby(pd.Grouper(freq='MS')).mean() #adjusted monthly data

    def calculate_log_change(self):
        self.log_change_data = (np.log(self.data) - np.log(self.data).shift(1)).dropna()


    def calculate_covariance_and_var(self):
        self.cov_matrix = self.log_change_data.cov() * 12
        self.var = pd.Series(np.diag(self.cov_matrix), index=[self.cov_matrix.columns])
        

    def calculate_beta(self): #can be done vith linalg cov_matrix * var 
        #getting beta as covar/var
        d = defaultdict(list)
        for index in self.market_indecies:
            var = self.cov_matrix.loc[index, index]
            for tick in self.stock_ticks:
                covar = self.cov_matrix.loc[index, tick]
                d[index] += [covar/var]
        self.beta1 = pd.DataFrame(data=d, index=self.stock_ticks)

    
    def calculate_regress_params(self):
        #getting alpha and beta with linear regression
        a = defaultdict(list)
        b = defaultdict(list)
        
        for market in self.market_indecies:
            for tick in self.stock_ticks:
                slope, intercept, _, _, _ = linregress(self.log_change_data[tick], self.log_change_data[market])
                a[market] += [intercept]
                b[market] += [slope]
     
        self.alfa = pd.DataFrame(data=a, index=self.stock_ticks)
        self.beta = pd.DataFrame(data=b, index=self.stock_ticks)

                
    def calculate_expected_market_return(self):
        #Using plane mean value
        self.market_returns = self.log_change_data[self.market_indecies].mean()
        #scaling to yearly using eulers
        self.market_returns_yr = np.exp(self.market_returns*12)-1
        

    def calculate_exp_return(self):
        # #Using CAPM
        # self.exp_return = self.risk_free_rate + (self.market_returns-self.risk_free_rate) * self.beta
        # Using plane mean value
        self.exp_return = self.log_change_data[self.stock_ticks].mean()
        self.exp_return_yr = np.exp(self.exp_return*12)-1

        
    def solve_quadratic_weights_of_portfolio(self):
        """
            where:
                R is the vector of CAPM expected returns
                C is the var-covariance matrix
                W is the weights
        """            
        R = self.exp_return_yr.values
        C = self.cov_matrix.iloc[:-1,:-1].values
        W = R*0 + 1/len(R) #Initialize equal procent weights
        
        def quad_var(W, C):
            return dot(dot(W.T, C), W) # Quadratic expression to calculate portfolio risk
        
        def exp_return(W, R):
            return np.dot(W.T, R).sum() # Expectd_portfolio_return
        
        def fitness(W, C, r):
            Pv = quad_var(W, C) 
            return Pv

        #Bounds (inequality constraints)
        b = [(0.,1.) for i in W] # weights between 0%..100%. -No shorting - no borrowing
        # Equality constraints
        h = ({'type':'eq', 'fun': lambda W: sum(W) -1.}, # Sum of weights = 100%
             {'type':'eq', 'fun': lambda W: exp_return(W, R) - r})  # equalizes portfolio return to r
        
        Rf = np.linspace(min(R), max(R), num=100)
        Vf, Wf = [], []
       
        for r in Rf:
            # For given level of return r, find weights which minimizes portfolio variance.
            optimized = minimize(fitness, W, args=(C, r), method='SLSQP', #Sequential Least SQuares Programming 
                                 constraints=h, bounds=b)
            X = optimized.x
            Wf.append(X)
            Vx = quad_var(X,C)
            Vf.append(Vx)

        self.frontier_exp_return = Rf #Y axis of EFF
        self.frontier_risk = Vf #X axis of EFF
        self.frontier_weights = [[round(w*100,2) for w in ws] for ws in Wf] #TODO might be done directly in pandas
    
    def calculate_CML_and_efficient_portfolio(self):

        Vf = self.frontier_risk
        Rf = self.frontier_exp_return

        def sharpe_ratio(Rp, Vp, rf):
            return (Rp - rf) / Vp

        self.EFFsr = sharpe_ratio(Rf, Vf, self.risk_free_rate)

        idx = np.argmax(self.EFFsr) # index of "market" portfolio
        self.EFFpx = Vf[idx]
        self.EFFpy = Rf[idx]

        self.CMLy = [self.risk_free_rate, self.EFFpy]
        self.CMLx = [0, self.EFFpx]

        def CML():
            risk_free_rate + market_return * Sharpe_ratio_of_the_market_portfolio

       

        # tck = splrep(EFF_Vf, EFF_Rf)

        # def f(x, tck=tck):
        #     # Efficient frontier function (splines approximation)
        #     return splev(x, tck, der=0)
            
        # def df(x, tck=tck):
        #     # First derivative of efficient frontier function
        #     return splev(x, tck, der=1)

        # def equations(p, rf=self.risk_free_rate):
        #     b, a, x = p
        #     intercept = rf - b
        #     slope = a - df(x)
        #     EFPx = a*x + rf - f(x) # Risk where the CML is the tangent of the EFF ie. x of efficient portfolio 
        #     return intercept, slope, EFPx

        # init_b = uniform(0.1, 5.0)
        # init_a = uniform(0.1, 5.0)
        # init_x = uniform(0.1, 5.0)
        # b,a, EFPx = fsolve(equations, [(init_b, init_a, init_x)]) # Using a Newton-Raphson optimisation algorithm
        # EFPy = f(EFPx)

        # def CML(x, b, a):
        #     return x*a+b

        # self.EEF_func = f
        # self.CMLy = [CML(x, b, a) for x in Vf]
        # self.EFP = (EFPx, EFPy)

    
    def plot_EFF(self):
        Xeef = self.frontier_risk
        Yeef = self.frontier_exp_return

        # Xcml = Xeef
        # Ycml = self.CMLy

        plotly.tools.set_credentials_file(username="TheVizWiz", api_key="92x5KNp4VDPBDGNtLR2l")

        def annotaions():
            PD = pd.DataFrame(self.frontier_weights, columns=self.stock_ticks)
            T = [re.sub(r'\n', "% -- ", re.sub(r'[ ]+', " ", PD.iloc[i].to_string() )) for i in PD.index]
            return T

        EEF = go.Scatter(
                    x=Xeef,
                    y=Yeef,
                    mode='markers+lines',
                    marker = dict(colorscale="Electric", color=self.EFFsr, showscale=True),
                    text = annotaions()
                )

        CML = go.Scatter(
                    x = self.CMLx,
                    y = self.CMLy,
                    mode='lines',
                    text = "Captial Market Line"
                    #marker = make coler outside space of efficient frontier different collor
                )

        EFP = go.Scatter(
                x = self.EFFpx,
                y = self.EFFpy,
                mode = 'marker',
                marker = dict(size=10, symbol="circle")
                )


        # EEF_aprox = go.Scatter(
        # 		x = Xeef_aprox,
        # 		y = Yeef_aprox,
        # 		mode = "makers+lines",
        # 		)

        data = [EEF, EFP, CML] # EEF_aprox]

        start = "{0}/{1}-{2}".format(self.start.day, self.start.month, self.start.year)
        end = "{0}/{1}-{2}".format(self.end.day, self.end.month, self.end.year)
        layout = go.Layout(
            title= "Efficent Frontier: from {} to {}".format(start, end),
            showlegend=True,
            hovermode= 'closest',
            yaxis = dict(title="Portfolio Return"),
            xaxis = dict(title="Portfolio Variance"),
            height=600,
            width=600,
        )

        fig = go.Figure(data=data, layout=layout)
        plot_url = py.plot(fig, filename='efficent_frontier')  
    
    
    def analyze_data(self):
        self.calculate_log_change()
        self.calculate_covariance_and_var()
        self.calculate_expected_market_return()
        self.calculate_beta()
        self.calculate_regress_params()
        self.calculate_exp_return()
        self.solve_quadratic_weights_of_portfolio()
        self.calculate_CML_and_efficient_portfolio()
        

    def run_pack(self):

        def one_window():
            self.get_monthly_data()
            self.analyze_data()
            self.plot_EFF()

        if self.window_size and self.window_move:

            def with_moving_window(func):
                def func_wrapper():

                    time = self.end - self.start
                    window = datetime.timedelta(days=self.window_size)
                    window_m = datetime.timedelta(days=self.window_move)

                    while time >= datetime.timedelta(1):
                        self.end = self.start + window
                        func()
                        self.start = self.start + window_m
                        time -= window_m

                return func_wrapper
           
            with_moving_window(one_window)()

            
        else:
            one_window()

if __name__ == '__main__':
  
    CP = Calcualtion_pack(stock_ticks=["AAPL", "MMM", "GOOGL", "AMZN", "FRX", "NKE", "TSN"], 
                            market_indecies=["NDAQ"],
                            start=datetime.datetime(1999, 1, 1), 
                            end=datetime.datetime(2018,1,1), 
                            risk_free_rate= 0.02,
                            # window_size=3650, 
                            # window_move=365
                            )
  
    CP.run_pack()
