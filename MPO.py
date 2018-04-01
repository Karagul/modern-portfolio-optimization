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
- Make line and at markers same collor in plot

"""



class Calcualtion_pack():
    "L4NT0W"
    
    def __init__(self, stock_ticks=None, stock_names=None, market_indecies=None, 
                 start=datetime.datetime(1997, 1, 1), end=datetime.date.today(), 
                 risk_free_rate=0.0, window_size=None, window_move=None, 
                 source="quandl", online=False, n_sim=0, annotations=None,
                 name_of_data="develop", stack_windows=None, required_return=None):

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
        self.annotations =  annotations
        self.required_return = required_return


        def logic_gate():
            self.plot_as_windows = True if self.window_size and self.window_move else False
            if not self.plot_as_windows: self.stack_windows = None # dont stack if only one window
            self.plot_CML = True if self.risk_free_rate > 0 else False 
            self.plot_simulation = True if self.n_sim > 0 and not self.stack_windows else False
            if self.stack_windows: self.annotations = False

        def sanity_check():
            pass
            # stack_windows only if window_size and window_move
            # if not stock names use stock ticks
            # Maximum 20 moving windows
            # n_sim max 80.000 if offline and 30.000 if online
            # start must be before end
            #MAKE SURE ALL STRING INPUT IS CORRECT and that start/end is in datetime format

        logic_gate()
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
            # TODO: TEST FOR EMPTY DATA!


    def calculate_log_change(self):
        self.log_change_data = (np.log(self.data) - np.log(self.data).shift(1)).dropna()


    def assign_data_window(self, opperation_type=None):
        """the unelegance is here that there needs to be first a calculation of the backtest weights with one range of data held out
                followed then by a calculation of the backtest expected return calculated on the held out data
                this assign data method is fairly adhock"""

        if opperation_type == "backtest_weights":
            df1 = self.log_change_data
            df2 = self.log_change_data[self.start:self.end]
            self.data_window = pd.concat([df1, df2]).drop_duplicates(keep=False) # removes the window from the dataframe ie. hold one out

        elif opperation_type == "windows": 
            self.data_window = self.log_change_data[self.start:self.end]

        else: 
            self.data_window = self.log_change_data


    def calculate_covariance_and_var(self):
        self.cov_matrix = self.data_window.cov() * 12
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
                slope, intercept, _, _, _ = linregress(self.data_window[tick], self.data_window[market])
                a[market] += [intercept]
                b[market] += [slope]
     
        self.alfa = pd.DataFrame(data=a, index=self.stock_ticks)
        self.beta = pd.DataFrame(data=b, index=self.stock_ticks)

                
    def calculate_expected_market_return(self):
        #Using plane mean value
        self.market_returns = self.data_window[self.market_indecies].mean()
        #scaling to yearly using eulers
        self.market_returns_yr = np.exp(self.market_returns*12)-1
        

    def calculate_exp_return(self):
        # #Using CAPM
        # self.exp_return = self.risk_free_rate + (self.market_returns-self.risk_free_rate) * self.beta
        # Using plane mean value
        self.exp_return = self.data_window[self.stock_ticks].mean()
        self.exp_return_yr = np.exp(self.exp_return*12)-1


    def solve_elements_for_plot(self):

        """Operations"""
        def quad_var(W, C):
            return np.sqrt(dot(dot(W.T, C), W)) # Quadratic expression to calculate portfolio risk
    
        def exp_return(W, R):
            return np.dot(W.T, R).sum() # Expectd_portfolio_return

        def exp_return1(W, R, rf):
            return rf + np.dot(W.T, (R-rf)).sum()

        def sharpe_ratio(Fr, Vf, rf):
            # returns Sr
            return (Fr - rf) / Vf

        def CML(Vf, rf, Sr):
            # risk_free_rate + risk * Sharpe_ratio_of_the_market_portfolio
            return Vf * Sr + rf

        def qsolve(R, Fr, C):
            """
            where:
                R is the vector of expected returns
                Fr is the range expected returns on the EFF
                C is the var-covariance matrix
            """
            # TODO: add options to short and borrow

            W = R*0 + 1/len(R) #Initialize equal procent weights
            #Bounds (inequality constraints)
            b = [(0.,1.) for i in W] # weights between 0%..100%. - no borrowing  -No shorting 
            

            def fitness(W, C, r):
                Pv = quad_var(W, C) 
                return Pv
            
            Vf, Wf = [], []

            # Equality constraints
            h = ({'type':'eq', 'fun': lambda W: sum(W)-1.}, # Sum of weights = 100%
                 {'type':'eq', 'fun': lambda W: exp_return(W, R) - r})  # equalizes portfolio return to r   

            for r in Fr:
                # For given level of return r, find weights which minimizes portfolio variance.
                optimized = minimize(fitness, W, args=(C, r), method='SLSQP', #Sequential Least SQuares Programming 
                                     constraints=h, bounds=b)
                X = optimized.x
                Wf.append(X)
                Vx = quad_var(X,C)
                Vf.append(Vx)

            return Vf, Wf

        R = self.exp_return_yr.values
        Fr = np.linspace(min(R), max(R), num=100) 
        C = self.cov_matrix.iloc[:-1,:-1].values #cov matrix without market index

        Vf, Wf = qsolve(R, Fr, C) 

        rf = self.risk_free_rate
        self.EFFsr = sharpe_ratio(Fr, Vf, rf) #sharpe ratio for portfolios on the eficient frontier

        # FRONTIER
        self.frontier_exp_return = Fr #Y axis of EFF
        self.frontier_risk = Vf #X axis of EFF
        self.frontier_weights = [[round(w*100,2) for w in ws] for ws in Wf] #TODO might be done directly in pandas
        
        #MARKET PORTFOLIO
        idxmax = np.argmax(self.EFFsr) # index of "market" portfolio
        MPsr = self.EFFsr[idxmax] # sharpe ratio of "market" portfolio ie. slope of CML
        self.Wmp = self.frontier_weights[idxmax]# weights of market portfolio
        self.marketPx = Vf[idxmax] # "market" portfolio x and y
        self.marketPy = Fr[idxmax]

        #MINIMUM RISK PORTFOLIO
        idxmin  = self.idxmin = np.argmin(Vf) # index of minimum risk portfolio
        self.minriskPx = Vf[idxmin]
        self.minriskPy = Fr[idxmin]

        #CAPITAL MARKET LINE
        self.CMLx = np.linspace(0, max(Vf), num=100) 
        self.CMLy = [CML(x, rf, MPsr) for x in self.CMLx]

       
        def qsolve1(CMLy, CMLx, C, R): #TODO: make 1 Qsolver with intuitive changeable constraints

            W = R*0 + 1/len(C) #Initialize equal procent weights
            #Bounds (inequality constraints)
            b = [(0.,2.) for i in W] # weights between 0%..100%. - no borrowing  -No shorting 
            
            def fitness(W, x, y, re, ri):
                Pv = sharpe_ratio(x, y, self.risk_free_rate)
                return - Pv.sum()
            
            Vf, Wf = [], []

            # Equality constraints
            h = ({'type':'eq', 'fun': lambda W: sum(W)-1.}, # Sum of weights = 100%
                 {'type':'eq', 'fun': lambda W: quad_var(W, C) - ri}, # equalizes portfolio risk to ri
                 {'type':'eq', 'fun': lambda W: exp_return1(W, R, self.risk_free_rate) - re})  # equalizes portfolio return to re 

            for ri, re in zip(CMLx, CMLy):
                # For given level of return r, find weights which minimizes portfolio variance.
                optimized = minimize(fitness, W, args=(CMLx, CMLy, ri, re), method='SLSQP', #Sequential Least SQuares Programming 
                                     constraints=h, bounds=b)
                X = optimized.x
                Wf.append(X)
                Vx = quad_var(X,C)
                Vf.append(Vx)

            return Vf, Wf

        R1 = self.exp_return_yr
        R1["Rf"] = self.risk_free_rate
        R1 = R1.values

        C1 = self.cov_matrix.iloc[:-1,:-1]
        C1["Rf"] = 0.0
        C1.loc["Rf"] = 0.0
        C1 = C1.values

        _, Wcml = qsolve1(self.CMLy, self.CMLx, C1, R1)

        self.CML_weights = [[round(w*100,2) for w in ws] for ws in Wcml]


        #portfolio on CML with rr as return
        if self.required_return:
             # DANGER! Mess ahead

            rr = self.required_return
            risk = (rr-rf)/MPsr
            self.CMLPx = risk
            self.CMLPy = rr
            _, CMLpw = qsolve1(np.array([rr]), np.array([risk]), C1, R1)
            self.CMLpw = [round(w*100,2) for w in CMLpw[0]] #Fix: why index?

        if self.plot_simulation:
            def MCsimulation(R, C, rf):
                returns, volatility, ratio = [], [], []
                for single_portfolio in range(self.n_sim):
                    W = np.random.normal(scale=4,size=len(self.stock_ticks))**2
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

        #TODO: plot 100% in one stock for evry stock


    def prepare_plot(self):
        
        def weights_in_text(n):
            if n == "EFF":
                PD = pd.DataFrame(self.frontier_weights, columns=self.stock_names)
                T = [re.sub(r'\n', "% <br>", re.sub(r'[ ]+', " ", PD.iloc[i].to_string() ))+"%" for i in PD.index]

            if n == "EFP":
                T = "Efficient portfolio<br>"+"".join(["{0}: {1}%<br>".format(name, weight) 
                        for name, weight in zip(self.stock_names, self.Wmp)])

            if n == "CML":
                PD = pd.DataFrame(self.CML_weights, columns=self.stock_names+["Risk-free rate"])
                T = [re.sub(r'\n', "% <br>", re.sub(r'[ ]+', " ", PD.iloc[i].to_string() ))+"%" for i in PD.index]

            if n == "CMLp":
                PD = pd.DataFrame(self.CMLpw, index=self.stock_names+["Risk-free rate"])
                T = [re.sub(r'\n', "% <br>", re.sub(r'[ ]+', " ", PD.to_string() ))+"%"]

            return T

        def annotations(strings, placements): # TODO: better annotations
                annotations=list()
                for s, p in zip(strings, placements):
                    d = dict(
                            x=p[0], y=p[1],
                            xref='paper', yref='paper',
                            text=s,
                            showarrow=True,
                            arrowhead=20
                        )
                    annotations.append(d)
                return annotations

        start = "{0}-{1}-{2}".format(self.start.day, self.start.month, self.start.year)
        end = "{0}-{1}-{2}".format(self.end.day, self.end.month, self.end.year)
        self.name = name = "{0} - {1}".format(start, end)

        data = list() 

        EFF = go.Scatter(
                x = self.frontier_risk[self.idxmin:],
                y = self.frontier_exp_return[self.idxmin:],
                mode = 'markers+lines',
                legendgroup = name if self.stack_windows else None,
                showlegend = True,
                marker = dict(size=5, symbol="circle"), #, color=[1 for _ in self.frontier_risk[self.idxmin:]], colorscale="Electric"),
                text = weights_in_text("EFF")[self.idxmin:],
                name = "Efficient frontier:<br>{}".format(name) if not self.stack_windows else name
                )

        EFP = go.Scatter(
                x = [self.marketPx],
                y = [self.marketPy],
                mode = 'markers',
                legendgroup = name if self.stack_windows else None,
                showlegend = False if self.stack_windows else True,
                marker = dict(size=15, symbol="circle"), #color=[1], colorscale="Electric"),
                name = "Market/Efficient portfolio"

                )

        MVP = go.Scatter(
                x = [self.minriskPx],
                y = [self.minriskPy],
                mode = "markers",
                legendgroup = name if self.stack_windows else None,
                showlegend = False if self.stack_windows else True,
                marker = dict(size=15, symbol="diamond-x"), #, color=[1], colorscale="Electric"),
                name = "minimum variance portfolio"
                )

        if self.plot_CML:
            CML = go.Scatter(
                x = self.CMLx,
                y = self.CMLy,
                mode='lines+markers',
                legendgroup = name if self.stack_windows else None,
                text = weights_in_text("CML"),
                name = "Capital market line"
                #marker = make coler outside space of efficient frontier different collor
                )
            data.append(CML)

        if self.required_return:
            CMLp = go.Scatter(
                x = [self.CMLPx],
                y = [self.CMLPy],
                mode = "markers",
                legendgroup = name if self.stack_windows else None,
                text = weights_in_text("CMLp"),
                name = "optimal asset allocation on CML",
                marker = dict(size=15, symbol="diamond-x")
                )
            data.append(CMLp)

        if self.plot_simulation:
            MonteCarlo = go.Scatter(
                x = self.MCx,
                y = self.MCy,
                mode = "markers",
                marker = dict(size=6, colorscale="Electric", color=self.MCsr, showscale=True, 
                              colorbar=dict(title="Sharpe Ratio", titleside="right")),
                name = "MonteCarlo Simulated portfolios"
                ) 
            data.append(MonteCarlo)

        data += [EFF, EFP, MVP]

        title = "Efficent Frontier"
        if not self.plot_as_windows:
            title = format("{0}<br>from {1} to {2}".format(title, start, end))

        self.layout = go.Layout(
            annotations = annotations([weights_in_text("EFP")],[(0.2,0.8)]) if self.annotations else annotations("",(0,0)), #TODO: better (less hacky) annotations
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
       
        if self.online:
            plotly.tools.set_credentials_file(username="TheVizWiz", api_key="92x5KNp4VDPBDGNtLR2l")
            py.plot(fig, filename='efficent_frontier')

        if not self.online:
            name = self.name_of_data + self.name
            plot_url =  offline.plot(fig, image='png',auto_open=True, image_filename=name,
                                     output_type='file', image_width=1200, image_height=1000, 
                                     filename="figures/{0}.html".format(name) # run some sys to create folder
                                     )
        self.plot_data = list() # Clear plot data when plot is made


    def with_moving_windows(self, operation):
        def func_wrapper():
            time = self.end - self.start
            # self.absolute_start = self.start
            # self.absolute_end = self.end
            window = datetime.timedelta(days=self.window_size)
            window_m = datetime.timedelta(days=self.window_move)

            while time-window >= datetime.timedelta(1):
                self.end = self.start + window
                operation() 
                self.start = self.start + window_m 
                time -= window_m

        return func_wrapper


    def prepare_data(self):
        self.get_monthly_data()
        self.calculate_log_change()

    def analyze_data(self):
        self.calculate_covariance_and_var()
        self.calculate_expected_market_return()
        self.calculate_beta()
        self.calculate_regress_params()
        self.calculate_exp_return()
        self.solve_elements_for_plot()


    def run_backtest(self):
        #cross-validation of model (WARNING: NOT PRETTY! - gaffataped in last moment)
        self.window_size=365 
        self.window_move=365
        self.market_portfolios = list()
        self.expected_portfolio_returns = list()
        self.prepare_data()

        def one_window():
            self.assign_data_window("windows")
            self.analyze_data()
            self.expected_portfolio_returns.append(self.exp_return_yr)
            self.assign_data_window("backtest_weights")
            self.analyze_data()
            self.market_portfolios.append(self.Wmp)
        
        self.with_moving_windows(one_window)()
        self.backtest_results = [(i*x).sum() for i,x in zip(self.expected_portfolio_returns, self.market_portfolios)]

    def run_pack(self):
        self.plot_data = list()
        self.prepare_data()

        def one_window():
            self.assign_data_window("windows")
            self.analyze_data()
            self.prepare_plot()
            if not self.stack_windows: self.execute_plot()

        if self.plot_as_windows:
            self.with_moving_windows(one_window)()
            if self.stack_windows: self.execute_plot() 
            
        else:
            one_window()

        

if __name__ == '__main__':
  
    CP = Calcualtion_pack(
                            stock_ticks = ["WIKI/AAPL", "WIKI/ABC", "WIKI/AGN", "WIKI/ADP", "WIKI/ADSK", "WIKI/IBM", "WIKI/GE"],
                            stock_names = "APL ABC AGN ADP ADSK IBM GE".split(),
                            # stock_ticks=["NASDAQOMX/NQDK4000DKK", "NASDAQOMX/NQDE", "NASDAQOMX/NQJP2000JPY",
                            #     "NASDAQOMX/NQHK2000HKD", "NASDAQOMX/NQGB", "NASDAQOMX/NQSE",
                            #     "NASDAQOMX/NQFI"],
                            market_indecies=["NASDAQOMX/NDX"],
                            start=datetime.datetime(1999, 1, 1), 
                            end=datetime.datetime(2018,1,1), 
                            # risk_free_rate= 0.01,
                            source = "pickle",
                            name_of_data = "USA",
                            # n_sim = 10000,
                            # online = True,
                            window_size=3650, 
                            window_move=365,
                            stack_windows = True,
                            annotations=True
                            
                            )
  
    CP.run_pack()
    # CP 