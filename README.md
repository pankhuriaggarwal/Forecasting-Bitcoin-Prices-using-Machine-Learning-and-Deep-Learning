# Forecasting-using-Deep-Learning
Forecasting daily values of the Bitcoin using Machine Learning and Deep Learning Models

Methodology:
- Deep Learning Models used are Multi-layer Perceptron, Gated Recurrent Unit + Vanilla Recurrent Neural Network and Long Short-term Memory Model
- Machine Learning Models used are Support Vector Regressor and Random Forest Regressor
- Traditional Models used are Autoregressive Integrated Moving Averages and Naive Forecasting
- All these models are compared for predicition accuracy using the Root Mean Squared Error. 

Data:
The dataset consists of time series data of one output and 30 input variables collected daily from Sep 17, 2014 to Aug 17, 2020. 
All values are denominated in USD at the closing price of the given day.
The output variable is the closing price of the bitcoin. 

The predictors consists of closing  prices of the following classes of financial assets:
- Foreign Exchange (FX) Rates: Major currencies, i.e. – United States Dollar (USD), Pound Sterling, Euro, Swiss Franc, Japanese Yen, Australian Dollar, Canadian Dollar and New Zealand Dollar, have been included. Additionally, Chinese Yuan (offshore), South Korean Won and Romanian Ieu have been included due to high number of Bitcoin holders in the nations. Hong Kong Dollar and Singapore Dollar have been included due to the large number of crypto exchanges in these countries.
- Commodities: Gold, Silver, Copper, Natural Gas, Brent Crude Oil and WTI Crude Oil have been included as these are amongst the most actively traded commodities in the world.
- Indices: Major indices such as, S&P 500, Dow Jones Industrial Average, NASDAQ Composite, CBOE Volatility Index, Nikkei 225, Hang Seng Index, KOSPI Composite Index, SSE Composite Index, FTSE 100 and DAX Performance Index, have been included.
- Fixed Income: The US 10 year treasury yield rates and USD 3 month London Interbank Offered Rate (LIBOR) have also been included. 

The last available data has been used for missing values such as, Friday closing price of S&P 500 is used as values for the weekend whenstock market is closed .
All the data has been retrieved using Yahoo! Finance except LIBOR which is obtained from Federal Reserve Bank of St. Louis (FRED Economic Data). All values has been converted to log returns for analysis.

