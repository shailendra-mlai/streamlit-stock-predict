# streamlit-stock-price-predict
### Predictive Model for S&P 500 Stock App using Streamlit

**Author**
Shailendra Mruthunjaya

#### Executive summary
The Predictive Model for S&P 500 Stock endeavors to provide actionable insights derived from five years of historical data. Leveraging this data alongside external factors such as news and trends, the model aims to offer recommendations on whether to hold, buy, or sell stocks. By employing a combination of time series forecasting techniques and machine learning methods, the project seeks to streamline decision-making processes for traders while reducing the time required for stock analysis. Ultimately, the objective is to develop a user-friendly application that empowers traders to make efficient and informed decisions regarding their investment strategies.

#### Rationale
Can a predictive model, utilizing historical data and external factors such as news and trends, effectively recommend whether to hold, buy, or sell S&P 500 stocks? The goal is to determine if such a model can enhance decision-making for part-time or hobby traders by providing timely and actionable trading insights.

#### Data Sources
*	Extract the data using [APIs](https://pypi.org/project/yfinance/) from https://finance.yahoo.com/. There is no need to dowload the dataset. 


#### Methodology
The methodology for forecasting stock prices combines time series techniques with machine learning methods for buy, sell, or hold recommendations. 

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture designed to address the vanishing gradient problem, which can occur in traditional RNNs. LSTM networks excel at capturing long-term dependencies in sequential data due to their ability to selectively retain or forget information over time. This makes them particularly effective for tasks where context and temporal relationships are essential, such as predicting stock prices based on historical data or generating sequences of text.

 - Data Collection: Retrieve historical stock price data for selected stock using the yfinance library and preprocess the data.
 - Data Preprocessing: Extract the 'Close' prices for each stock. Normalize the data using MinMaxScaler to scale the values between 0 and 1.
 - Feature Engineering: Create training and testing datasets using a sliding window approach. Split the data into training and test sets with a 95% to 5% ratio.
 - Model Building: Build LSTM (Long Short-Term Memory) models with multiple layers and dropout regularization to prevent overfitting. Use a sequential model with four LSTM layers and a dense output layer.
 - Training the Model: Compile the model using the Adam optimizer and mean squared error as the loss function.
 - Train the model on the training data while validating on the test data, capturing loss and validation loss metrics.
 - Evaluation and Forecasting: Evaluate the model's performance on the test data. Predict future stock prices using the trained model. Inverse transform the predictions to get the actual scale.
 - Visualization: Plot the training loss and validation loss to understand model performance. Visualize the actual versus predicted stock prices for each company.

For complete analysis see [Forecasting stock price using LSTM](https://github.com/shailendra-mlai/Predictive-Model-for-SAP-500-Stock/blob/main/LSTM-Stock-price-prediction.ipynb)



#### Results

The LSTM model proved to be beneficial for traders by offering actionable insights and facilitating informed decision-making. By analyzing the trends and patterns in stock prices, the LSTM model provided accurate forecasts, thus empowering traders to make informed decisions efficiently. This resulted in a reduction in the time required for stock analysis while ensuring reliable predictions to guide trading strategies.

#### Next steps
- Explore additional features that may influence stock price movements, such as technical indicators such as moving averages, relative strength index, macroeconomic indicators like interest rates, GDP growth, and sentiment analysis of news and social media.
- Develop risk management strategies to mitigate potential losses, such as setting stop-loss orders, diversifying investment portfolios, and incorporating risk-adjusted return metrics into the decision-making process. 
- An app created using https://streamlit.io/ could allow users to input a stock symbol and receive a recommendation.


#### Outline of project

- [Notebook to develop steamlit app with user input](https://github.com/shailendra-mlai/Predictive-Model-for-SAP-500-Stock/blob/main/LSTM-App.ipynb)
