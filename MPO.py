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
import plotly.offline as offline
import pickle
from time import sleep

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
- ADD option to save 

"""

class Calcualtion_pack():
    "L4NT0W"
    
    def __init__(self, stock_ticks=None, stock_names=None, market_indecies=None, 
                 start=datetime.datetime(1997, 1, 1), end=datetime.date.today(), 
                 risk_free_rate= 0.02, window_size=None, window_move=None, 
                 source="quandl", online=False, n_sim=10000,
                 name_of_data="develop", stack_windows=None):

        self.stock_ticks = stock_ticks
        self.market_indecies = market_indecies #Concatenate different markets indecies to get "real" market
                                                # Possibly do this with different weights
        self.start = start
        self.end = end
        self.risk_free_rate = risk_free_rate
        self.window_move = window_move
        self.window_size = window_size
        self.source = source
        self.online = online
        self.n_sim = n_sim
        self.stock_names = stock_names
        self.save_data = True
        self.name_of_data = name_of_data
        self.stack_windows = stack_windows #ie. only one plot
        def sanity_check():
            self.plot_CML = True if self.risk_free_rate > 0 else False 
            self.plot_simulation = True if self.n_sim > 0 and not self.stack_windows else False
            self.plot_many_windows = True if self.window_size and self.window_move else False
            # stack_windows only if window_size and window_move
            # if not stock names use stock ticks
            # Maximum 20 moving windows
            # n_sim max 500.000 if offline and 30.000 if online
            # start must be before end

        sanity_check()   

    def get_monthly_data(self):
        #TODO add better module for data management and auto naming files

        if self.source == "pickle":
            self.data = pickle.load(open( "{}.p".format(self.name_of_data), "rb" ) )

        else:
            if self.source in ["google", "yahoo"]: #DEPRICATED
                raw_data = pdr.DataReader(self.market_indecies + self.stock_ticks, 
                                          self.source, self.start, self.end)
                adj_data = raw_data["Close"]

            if self.source == "quandl":
                adj_data = defaultdict()
                for ticker in self.stock_ticks + self.market_indecies:
                    data = QuandlReader(symbols=ticker, start=self.start, end=self.end).read()

                    if "AdjClose" in data.columns:
                        adj_data[ticker] = data["AdjClose"]
                    elif "IndexValue" in data.columns:
                        adj_data[ticker] = data["IndexValue"]

                    sleep(0.4)

                adj_data = pd.DataFrame(adj_data)

            self.data = adj_data.groupby(pd.Grouper(freq='MS')).mean() #adjusted monthly data
            if self.save_data: pickle.dump(self.data, open("{}.p".format(self.name_of_data), "wb")) 

    def calculate_log_change(self):
        self.log_change_data = (np.log(self.data) - np.log(self.data).shift(1)).dropna()[self.start:self.end]

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


    # def solve_quadratic_weights_of_portfolios(self): 
    def solve_elements_for_plot(self):

        """Operations"""
        def quad_var(W, C):
            return np.sqrt(dot(dot(W.T, C), W)) # Quadratic expression to calculate portfolio risk
    
        def exp_return(W, R):
            return np.dot(W.T, R).sum() # Expectd_portfolio_return

        def sharpe_ratio(Rf, Vf, rf):
            return (Rf - rf) / Vf

        def CML(Vf, rf, Sr):
            # risk_free_rate + risk * Sharpe_ratio_of_the_market_portfolio
            return Vf * Sr + rf

        def qsolve(R, Rf, C):
            """
            where:
                R is the vector of expected returns
                Rf is the 
                C is the var-covariance matrix
            """
            # TODO: add options to short and borrow

            W = R*0 + 1/len(R) #Initialize equal procent weights
            #Bounds (inequality constraints)
            b = [(0.,1.) for i in W] # weights between 0%..100%. - no borrowing
            # Equality constraints

            def fitness(W, C, r):
                Pv = quad_var(W, C) 
                return Pv
            
            Vf, Wf = [], []

            h = ({'type':'eq', 'fun': lambda W: sum(W)-1.}, # Sum of weights = 100% -No shorting 
                 {'type':'eq', 'fun': lambda W: exp_return(W, R) - r})  # equalizes portfolio return to r   

            for r in Rf:
                # For given level of return r, find weights which minimizes portfolio variance.
                optimized = minimize(fitness, W, args=(C, r), method='SLSQP', #Sequential Least SQuares Programming 
                                     constraints=h, bounds=b)
                X = optimized.x
                Wf.append(X)
                Vx = quad_var(X,C)
                Vf.append(Vx)

            return Vf, Wf

        R = self.exp_return_yr.values
        Rf = np.linspace(min(R), max(R), num=100) 
        C = self.cov_matrix.iloc[:-1,:-1].values #cov matrix without market index

        Vf, Wf = qsolve(R, Rf, C)

        self.frontier_exp_return = Rf #Y axis of EFF
        self.frontier_risk = Vf #X axis of EFF
        self.frontier_weights = [[round(w*100,2) for w in ws] for ws in Wf] #TODO might be done directly in pandas
        
        rf = self.risk_free_rate
        self.EFFsr = sharpe_ratio(Rf, Vf, rf) #sharpe ratio for portfolios on the eficient frontier

        idxmax = np.argmax(self.EFFsr) # index of "market" portfolio
        MPsr = self.EFFsr[idxmax] # sharpe ratio of "market" portfolio ie. slope of CML
        self.Wmp = self.frontier_weights[idxmax]# weights of market portfolio
        self.marketPx = Vf[idxmax] # "market" portfolio x and y
        self.marketPy = Rf[idxmax]

        idxmin  = self.idxmin = np.argmin(Vf) # index of minimum risk portfolio
        self.minriskPx = Vf[idxmin]
        self.minriskPy = Rf[idxmin]

        self.CMLx = [0] + Vf
        self.CMLy = [CML(x, rf, MPsr) for x in self.CMLx]       

        if self.plot_simulation:
            def MCsimulation(R, C, rf):
                returns, volatility, ratio = [], [], []
                for single_portfolio in range(self.n_sim):
                    W = np.random.normal(scale=3,size=len(self.stock_ticks))**2
                    W /= np.sum(W)
                    ret = exp_return(W, R)
                    vol = quad_var(W, C)
                    returns.append(ret)
                    volatility.append(vol)
                    ratio.append(sharpe_ratio(ret, vol, rf))

                self.MCx = volatility
                self.MCy = returns
                self.MCsr = ratio

            MCsimulation(R, C, rf)

    
    def prepare_plot(self):
        
        def annotaions():
            PD = pd.DataFrame(self.frontier_weights, columns=self.stock_names)
            T = [re.sub(r'\n', "% <br>", re.sub(r'[ ]+', " ", PD.iloc[i].to_string() )) for i in PD.index]
            return T

        start = "{0}-{1}-{2}".format(self.start.day, self.start.month, self.start.year)
        end = "{0}-{1}-{2}".format(self.end.day, self.end.month, self.end.year)
        self.name = name = "{0} - {1}".format(start, end)
        EFP_weights = str(zip(self.Wmp, self.stock_names))

        EFF = go.Scatter(
                x = self.frontier_risk[self.idxmin:],
                y = self.frontier_exp_return[self.idxmin:],
                mode = 'lines+markers',
                legendgroup = name,
                marker = dict(size=5),
                text = annotaions(),
                name = "Efficient frontier<br>" + name 
                )

        EFP = go.Scatter(
                x = self.marketPx,
                y = self.marketPy,
                mode = 'markers',
                legendgroup = name,
                marker = dict(size=10, symbol="circle"),
                text =  EFP_weights,
                name = "Market/Efficient portfolio"
                )

        MVP = go.Scatter(
                x = self.minriskPx,
                y = self.minriskPy,
                mode = "markers",
                legendgroup = name,
                marker = dict(size=10, symbol="circle"),
                name = "minimum variance portfolio"
                )

        data = [EFF, EFP, MVP]

        if self.plot_CML:
            CML = go.Scatter(
                x = self.CMLx,
                y = self.CMLy,
                mode='lines',
                legendgroup = name,
                #text = "Captial Market Line", #solve weights and show as text
                name = "Capital market line"
                #marker = make coler outside space of efficient frontier different collor
                )
            data.insert(0, CML)

        if self.plot_simulation:
            MonteCarlo = go.Scatter(
                x = self.MCx,
                y = self.MCy,
                mode = "markers",
                marker = dict(size=6, colorscale="Electric", color=self.MCsr, showscale=True, 
                              colorbar=dict(title="Sharpe Ratio", titleside="right")),
                name = "MonteCarlo Simulated portfolios"
                ) 
            data.insert(0, MonteCarlo)

        title = "Efficent Frontier"
        if not self.plot_many_windows:
            title = format("{0}<br>from {1} to {2}".format(title, start, end)) 
        self.layout = go.Layout(
            legend=dict(
                x=1.2,
                y=1.2,
                traceorder='grouped',
                tracegroupgap=20,
                font=dict(
                    family='sans-serif',
                    size=20,
                    color='#000'
                    ),
                bgcolor='#E2E2E2',
                bordercolor='#FFFFFF',
                borderwidth=2
                ),
            title=title,
            showlegend=True,
            font=dict(
                    size=20,
                    color='#000'
                    ),
            hovermode='closest',
            yaxis=dict(title="Portfolio Return"),
            xaxis=dict(title="Portfolio Variance"),
            height=1000,
            width=1200,  
            )

        self.plot_data += data
    
    def execute_plot(self):
        fig = go.Figure(data=self.plot_data, layout=self.layout)
        self.plot_data = list()

        if self.online:
            plotly.tools.set_credentials_file(username="TheVizWiz", api_key="92x5KNp4VDPBDGNtLR2l")
            py.plot(fig, filename='efficent_frontier')

        if not self.online:
            name = self.name_of_data + self.name
            plot_url =  offline.plot(fig, image='png',auto_open=True, image_filename=name,
                                     output_type='file', image_width=1200, image_height=1000, 
                                     filename=name+".html")
        

    def analyze_data(self):
        self.calculate_log_change()
        self.calculate_covariance_and_var()
        self.calculate_expected_market_return()
        self.calculate_beta()
        self.calculate_regress_params()
        self.calculate_exp_return()
        self.solve_elements_for_plot()


    def run_pack(self):
        self.get_monthly_data()
        self.plot_data = list()

        def one_window():
            self.analyze_data()
            self.prepare_plot()

        if self.plot_many_windows:

            def with_moving_window(generate_window):
                def func_wrapper():
                    time = self.end - self.start
                    window = datetime.timedelta(days=self.window_size)
                    window_m = datetime.timedelta(days=self.window_move)

                    while time-window >= datetime.timedelta(1):
                        self.end = self.start + window
                        generate_window()
                        self.execute_plot()
                        self.start = self.start + window_m
                        time -= window_m

                return func_wrapper

            with_moving_window(one_window)()
            
            
        else:
            one_window()
            self.execute_plot()

        

if __name__ == '__main__':
  
    CP = Calcualtion_pack(
                            stock_ticks = ["WIKI/AAPL", "WIKI/ABC", "WIKI/AGN", "WIKI/ADP", "WIKI/ADSK", "WIKI/IBM", "WIKI/GE"],
                            stock_names = "APL ABC AGN ADP ADSK IBM GE".split(),
                            # stock_ticks=["NASDAQOMX/NQDK4000DKK", "NASDAQOMX/NQDE", "NASDAQOMX/NQJP2000JPY",
                            #     "NASDAQOMX/NQHK2000HKD", "NASDAQOMX/NQGB", "NASDAQOMX/NQSE",
                            #     "NASDAQOMX/NQFI"],
                            market_indecies=["GOOGL"],
                            start=datetime.datetime(2003, 1, 1), 
                            end=datetime.datetime(2018,1,1), 
                            risk_free_rate= 0.02,
                            source = "pickle",
                            name_of_data = "USA",
                            n_sim = 10000,
                            online = False,
                            # window_size=3650, 
                            # window_move=365,
                            #stack_windows = True
                            )
  
    CP.run_pack()
