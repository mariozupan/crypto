
# An Experiment on Predicting Cryptocurrency prices with LSTM Networks

> This repository does not constitute financial advice and should not be taken as such. 
> While cryptocurrency investments will definitely go up in value forever, they may also go down.

## Problem Description

In this experiment I build an LSTM network in order to predict the price of Bitcoin based on scenario described at [1] and [3].

## Data
The data about Bitcoin was extracted from [coinmarketcap.com](coinmarketcap.com).
The training data consists of **multiple multivariate time series** with "day" as the time unit that start from **2013-12-27** to **2017-09-24**.
The testing data has the same data schema as the training data and start from **2017-09-25** to **2017-11-27**.
Finally, the model was blind tested using data from **2017-11-29** to **2017-12-06**.

## LSTM Configuration

The LSTM model will use previous data to predict the next day's closing price of bitcoin. 
So we set how many previous days it will have access to equals to 20.

LSTM params:   
- epochs = 100
- batch_size = 32
- num_of_neurons_lv1 = 50    
- num_of_neurons_lv2 = 25 
- activ_func="linear",
- dropout=0.5
- loss="mean_squared_error"
- optimizer="adam"

### Results:

|Mean Squared Error|Mean Absolute Error|Coefficient of Determination (R^2)|
|----|----|----|
|0.003|0.048|0.603| 

#### Blind Test:  
Single day prediction from **2017-11-28** to **2017-12-05**:
9669.40, 10035.91, 9830.83, 9993.55, 10367.92, 10770.16, 10858.19, 11521.98

The truth from **2017-11-29** to **2017-12-06**:
9888.61, 10233.60, 10975.60, 11074.60, 11323.20 , 11657.20, 11916.70, 14291.50
  
**Pearson correlation** between single day prediction and the truth: 0.92
  
## References:

- [1] Predicting the price of Bitcoin using Machine Learning [here](http://trap.ncirl.ie/2496/1/seanmcnally.pdf).
- [2] Automated Bitcoin Trading via Machine Learning Algorithms [here](http://ai2-s2-pdfs.s3.amazonaws.com/e065/3631b4a476abf5276a264f6bbff40b132061.pdf)
- [3] Predicting cryptocurrency prices with deep learning [here](https://github.com/dashee87/blogScripts/blob/master/Jupyter/2017-11-20-predicting-cryptocurrency-prices-with-deep-learning.ipynb)
- [4] Multidimensional LSTM Networks to Predict Bitcoin Price [here](http://www.jakob-aungiers.com/articles/a/Multidimensional-LSTM-Networks-to-Predict-Bitcoin-Price)
- [5] Kaggle Dataset [here](https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory)