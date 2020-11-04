# Forecasting-using-Deep-Learning
Forecasting daily values of the Bitcoin using Machine Learning and Deep Learning Models

Methodology:
- Deep Learning Models used are Multi-layer Perceptron, Vanilla Recurrent Neural Network and Long Short-term Memory Model
- Machine Learning Models used are Support Vector Regressor and Random Forest Regressor
- Traditional Models used are Random Walk, ARIMA and ARMAX
- All these models are compared for predicition accuracy using the Root Mean Squared Error
- Feature Selection method used is Mutual Information Regression
- Hyperparameter Tuning using Manual Search and Random Search

# Data
The undermentioned variables will be used as predictors to forecast the daily value of Bitcoin. The dataset consists of time series data of one output and 32 input variables collected daily from Sep 17, 2014 to Aug 17, 2020. All values are denominated in USD at the adjusted closing price of the given day. The exceptions are DAX Performance Index, which is denominated in USD at the closing price of the given day and LIBOR, which is denominated in USD at its daily value. The output variable is the adjusted closing price of the bitcoin denominated in USD.

The predictors consist of closing prices of the following classes of financial assets:
  1. Foreign Exchange (FX) Rates: Major currencies, i.e. – United States Dollar (USD), Pound Sterling, Euro, Swiss Franc, Japanese Yen, Australian Dollar, Canadian Dollar and New Zealand Dollar, have been included (Lee, 2019). Additionally, Chinese Yuan, South Korean Won and Romanian Leu have been included due to high number of Bitcoin holders in the nations (BIDITEX Exchange, 2020). Hong Kong Dollar and Singapore Dollar have been included due to the large number of crypto exchanges in these countries (Alexandre, 2019).
  2. Commodities: Gold, Silver, Copper, Natural Gas, Brent Crude Oil and WTI Crude Oil have been included as these are amongst the most actively traded commodities in the world (Plus500, n.d.).
  3. Indices: Major indices such as, S&P 500, Dow Jones Industrial Average, NASDAQ Composite, CBOE Volatility Index, Nikkei 225, Hang Seng Index, KOSPI Composite Index, SSE Composite Index, FTSE 100 and DAX Performance Index, have been included. The U.S. Dollar Index, which tracks the value of USD against a basket of global currencies, is also included,
  4. Fixed Income: The US 10-year treasury yield rates are “the most widely tracked rates” in financial markets. They impact multiple fixed income instruments and are used as a benchmark for other types of long-term debt (Franck, 2018). USD 3-month London Interbank Offered Rate (LIBOR) serves as a widely used reference for short-term interest rates (Federal Reserve Bank of St. Louis, 2020). Thus, these 2 rates have also been included as predictor variables.
Additionally, the daily trading volume of the Bitcoin is also included as an input.

The last available data has been used for missing values such as, Friday closing price of S&P 500 is used as values for the weekend when the stock market is closed. All the data has been retrieved using Yahoo! Finance, except LIBOR which is obtained from Federal Reserve Bank of St. Louis (FRED Economic Data) and DAX Performance Index which is obtained from Wall Street Journal.

All values have been converted to log returns for analysis. The input variables have been standardized to mean=0 and variance=1 for efficient implementation of machine learning and deep learning models (McNally et al., 2018). Note that the value of predictors at time t-1 are used to predict the value of Bitcoin at time t. Feature selection is conducted to identify the relevant predictors for machine learning and deep learning models has been detailed in the next section.
