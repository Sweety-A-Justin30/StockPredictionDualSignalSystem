check in here:  https://stockpricepredictionlstm30.streamlit.app/

# StockPredictionDualSignalSystem
Stock price prediction using technical indicators with dual signal and alert system

The stock price prediction model used in this project is LSTM which is a type of recurrent neural network that is particularly efficient for sequential data. In essence, the goal of this model is to predict future stock prices and provide trading signals that existing and potential investors can use to make sound decisions.

To improve the efficiency of the prediction, the model includes a dual signal system. For existing investors, it produces ‘Buy,’ ‘Sell,’ or ‘Hold’ signals depending on the position of the short-term and long-term EMAs. For new investors, another signal is added, ‘Consider Entry,’ when the market is not bearish but not necessarily bullish for a strong buy signal. This system ensures that both groups of investors are provided with relevant information for market operations.


In a stock price prediction system, the architecture typically includes:

	•	Data Collection: Gather historical stock data from financial APIs
 
	•	Data Processing: Cleans and prepares the data, calculates technical indicators, and formats it for model training.
 
	•	Model Training: Uses historical data to train predictive models like LSTM (Long Short-Term Memory) networks.
 
	•	Forecasting: Applies the trained model to forecast future stock prices.
 
	•	Signal Detection: Analyzes the forecasted data to generate trading signals (Buy, Sell, Hold) based on technical indicators.
 
	•	Alert System: Monitors for significant market changes and triggers alerts for potential risks.

Criteria for the signals are:

For Existing Investors:

	•	Buy Signal: We recommend buying stocks when the short-term EMA is above the long-term EMA. This usually means the stock price is likely to go up.
 
	•	Sell Signal: We suggest selling when the short-term EMA is below the long-term EMA, indicating that the stock price might be going down.
 
	•	Hold Signal: If the short-term and long-term EMAs are close to each other, it’s a sign to hold onto the stocks since there’s no strong trend.
 
      2.	For New Investors:
      
	•	Consider Entry Signal: When the market is stable but not showing a strong buy signal, new investors get a “Consider Entry” signal. This means it might be a good time to think about buying stocks, but with caution.
