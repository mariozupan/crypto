'''
Created on 24 nov 2017

@author: mantica

'''
import pandas as pd
import time
import numpy as np
from scipy.stats.stats import pearsonr 
import os
from PIL import Image
import io
import urllib
from PlotUtils import PlotUtils
from ModelUtils import ModelUtils

# import the relevant Keras modules
from keras.models import load_model

if __name__ == '__main__':   
    
    ########################
    # Configuration
    ######################## 
    
    model_path = "../Output/bt_model.h5"
    
    start_dates = ['20171108', '20171109', '20171110', '20171111', '20171112', '20171113', '20171114', '20171115']
    #end_date=time.strftime("%Y%m%d")
    end_dates = ['20171128', '20171129', '20171130', '20171201', '20171202', '20171203', '20171204', '20171205']

    ground_truth = [9888.61, 10233.60, 10975.60, 11074.60, 11323.20 , 11657.20, 11916.70, 14291.50]
    
    predictions = []
    
    window_len = 20  
    
    ###################################
    # Getting the Eth
    ###################################
    
    bt_img = urllib.request.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")    
    image_file = io.BytesIO(bt_img.read())
    bt_im = Image.open(image_file)
    width_bt_im , height_bt_im = bt_im.size
    bt_im = bt_im.resize((int(bt_im.size[0] * 0.8), int(bt_im.size[1] * 0.8)), Image.ANTIALIAS)
    
    for  start_date, end_date in zip(start_dates, end_dates):
        
        ################
        # Data Ingestion
        ################
        
        # get market info for ethereum from the start of 2016 to the current day
        time.strftime("%Y%m%d")
        bt_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=" + start_date + "&end=" + end_date)[0]
        # convert the date string to the correct date format
        bt_market_info = bt_market_info.assign(Date=pd.to_datetime(bt_market_info['Date']))
        '''
        # look at the first few rows
        print("ETH")
        print(bt_market_info.head())
        print('\nshape: {}'.format(bt_market_info.shape)) 
        print("\n")
        # PlotUtils.plotCoinTrend(bt_market_info, bt_im)
        '''
        # Feature Eng
        # print("Feature ENG")
        bt_market_info.columns = [bt_market_info.columns[0]] + ['bt_' + i for i in bt_market_info.columns[1:]]
        for coins in ['bt_']: 
            kwargs = { coins + 'day_diff': lambda x: (x[coins + 'Close'] - x[coins + 'Open']) / x[coins + 'Open']}
            bt_market_info = bt_market_info.assign(**kwargs)
        '''    
        print('\nshape: {}'.format(bt_market_info.shape))
        print(bt_market_info.head())
        print("\n")
        '''
        
        ###########################################################################################################
        # DATA PREPARATION
        # In time series models, we generally train on one period of time and then test on another separate period.
        # I've created a new data frame called model_data. 
        # I've removed some of the previous columns (open price, daily highs and lows) and reformulated some new ones.
        # close_off_high represents the gap between the closing price and price high for that day, where values of -1 and 1 
        # mean the closing price was equal to the daily low or daily high, respectively. 
        # The volatility columns are simply the difference between high and low price divided by the opening price.
        # You may also notice that model_data is arranged in order of earliest to latest. 
        # We don't actually need the date column anymore, as that information won't be fed into the model.
        ###########################################################################################################
        # close_off_high = 2 * (High - Close) / (High - Low) - 1
        # volatility = (High - Low) / Open
        for coins in ['bt_']: 
            kwargs = { coins + 'close_off_high': lambda x: 2 * (x[coins + 'High'] - x[coins + 'Close']) / (x[coins + 'High'] - x[coins + 'Low']) - 1,
                    coins + 'volatility': lambda x: (x[coins + 'High'] - x[coins + 'Low']) / (x[coins + 'Open'])}
            bt_market_info = bt_market_info.assign(**kwargs)
        model_data = bt_market_info[['Date'] + [coin + metric for coin in [ 'bt_'] 
                                       for metric in ['Close', 'Volume', 'close_off_high', 'volatility', 'day_diff', 'Market Cap']]]
        
        # need to reverse the data frame so that subsequent rows represent later timepoints
        model_data = model_data.sort_values(by='Date')
        '''
        print("Model Data")
        print('\nshape: {}'.format(model_data.shape))
        print(model_data.head())
        print("\n")
        '''
        model_data = model_data.drop('Date', 1) 
        
        norm_cols = [coin + metric for coin in ['bt_'] for metric in ['Close', 'Volume', 'Market Cap']]
        
        LSTM_test_inputs = ModelUtils.buildLstmInput(model_data, norm_cols, window_len)
    
        # print("\nNumber Of Input Test's sequences: {}".format(len(LSTM_test_inputs)))
    
        # I find it easier to work with numpy arrays rather than pandas dataframes
        # especially as we now only have numerical data
        
        LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
        LSTM_test_inputs = np.array(LSTM_test_inputs)
        
        # if best iteration's model was saved then load and use it
        if os.path.isfile(model_path):
                          
            estimator = load_model(model_path, custom_objects={'r2_keras': ModelUtils.r2_keras})
            print((((np.transpose(estimator.predict(LSTM_test_inputs)) + 1) * model_data['bt_Close'].values[:-window_len])[0])[0])
            predictions.append((((np.transpose(estimator.predict(LSTM_test_inputs)) + 1) * model_data['bt_Close'].values[:-window_len])[0])[0])
    
    print(predictions)
    print(pearsonr(predictions, ground_truth))
