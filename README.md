check in here:  https://stockpricepredictionlstm30.streamlit.app/
app images:
<img width="330" alt="Screenshot 2024-09-20 at 12 49 55 PM" src="https://github.com/user-attachments/assets/0e5008d6-9e04-4137-a11f-a3a84236729d">
<img width="915" alt="Screenshot 2024-09-20 at 12 50 47 PM" src="https://github.com/user-attachments/assets/cef48cbd-b0ec-4bbc-ba82-4669c9bdeb20">
<img width="915" alt="Screenshot 2024-09-20 at 12 51 28 PM" src="https://github.com/user-attachments/assets/58d57dc8-bacb-4173-8f8a-b38c27452c11">
<img width="915" alt="Screenshot 2024-09-20 at 12 52 02 PM" src="https://github.com/user-attachments/assets/a71b8622-4d5d-4b36-9869-594bc0e51a0e">


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
