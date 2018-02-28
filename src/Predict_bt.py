'''
Created on 24 nov 2017

@author: mantica

'''
import pandas as pd
import time
import datetime
from datetime import timedelta
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

import time, threading

def predikcija():
    if __name__ == '__main__':   
        
        ########################
        # Configuration
        ######################## 
        
        model_path = "../Output/bt_model.h5"
        
        start_dates = []
        for dana_unazad in range(30, 0, -1):
            danas = datetime.date.today() - timedelta(days=dana_unazad)
            start_dates.append(str(danas.strftime("%Y%m%d")))
        print(start_dates)
        
        end_dates = []
        for dana_unazad in range(1, -3, -1):
            danas = datetime.date.today() - timedelta(days=dana_unazad)
            end_dates.append(str(danas.strftime("%Y-%m-%d")))
        print(end_dates)


        # get market info for bitcoin from the start of 2016 to the current day
        bt_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=" + end_dates[0] + "&end=" + end_dates[len(end_dates)-1])[0]
        # convert the date string to the correct date format
        bt_market_info = bt_market_info.assign(Date=pd.to_datetime(bt_market_info['Date']))

        burza = bt_market_info
        # look at the first few rows
        print("BT")
        print(bt_market_info.head())
        print('\nshape: {}'.format(bt_market_info.shape)) 
        print("\n")
        print(bt_market_info.Date)
        print(bt_market_info.Close)


        ground_truth = bt_market_info.Close #[9888.61, 10233.60, 10975.60]

        
        predictions = []
        
        window_len = 20  
        
        ###################################
        # Getting the BTC
        ###################################
        
        bt_img = urllib.request.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")    
        #bt_img = open("../Output/Bitcoin-Logo-640x480.png")
        image_file = io.BytesIO(bt_img.read())
        bt_im = Image.open(image_file)
        width_bt_im , height_bt_im = bt_im.size
        bt_im = bt_im.resize((int(bt_im.size[0] * 0.8), int(bt_im.size[1] * 0.8)), Image.ANTIALIAS)
        
        for  start_date, end_date in zip(start_dates, end_dates):
            
            ################
            # Data Ingestion
            ################
            
            # get market info for BTC from the start of 2016 to the current day
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
                print(str(end_date) + '--> ')
                print((((np.transpose(estimator.predict(LSTM_test_inputs)) + 1) * model_data['bt_Close'].values[:-window_len])[0])[0])
                predictions.append((((np.transpose(estimator.predict(LSTM_test_inputs)) + 1) * model_data['bt_Close'].values[:-window_len])[0])[0])
        
        #zaokruzi na dvije decimale
        predictions = np.round(predictions,2)
        
        #datafreme predikcije stari datumi
        burza = pd.DataFrame(burza)
        burzaHtml = burza.to_html(justify=True)


        #datafreme predikcije tri dana
        dfPredikcija = pd.DataFrame()
        dfPredikcija['Datumi'] = end_dates
        dfPredikcija['Closing Price'] = predictions
        
        dfPredikcija = dfPredikcija.sort_values(by='Datumi', ascending=False)

        s = dfPredikcija.style.set_properties(**{'text-align': 'right'})
        s.render()
        
        predictionsHtml = dfPredikcija.to_html(justify='right', index=False) #justfiy poravna samo zaglavlje
        predictionsHtml = predictionsHtml.replace('<tr>','<tr style="text-align: right;">')
        
        
        print(predictionsHtml)
        print('pearson test: WAIT!!! ')
        #print(pearsonr(predictions, ground_truth))

        f = open('../Output/cijenaBTC.html','w', encoding='utf-16')

        message = """

            <br>
            ﻿@{
                <br>
                Layout = "~/_SiteLayout.cshtml";
                Page.Title = "Predikcijski modeli, deep learning analize";
                <br>
            }
            <br>
            <p>
                """ +str('Predikcija cijene za sljedeća tri dana')+"""
                <br>
                <br>
            </p>
            <br>
            """+str(predictionsHtml)+"""
    
            <br>
            <br>        
            

            """+str(burzaHtml)+"""

            <br>
            <br>        

            """+str(s.render())+"""
            <br>
            <br>        

            <p>
                <img src="../BTCoutput/BitcoinTrend.png" alt="BTC">
                <img src="../BTCoutput/BitcoinTrainTest.png" alt="BTC">
                <img src="../BTCoutput/BTCperformanceTraining.png" alt="BTC">
                <img src="../BTCoutput/BTCPredictionMoreDays.png" alt="BTC">
                <img src="../BTCoutput/BTCmoreDays_error.png" alt="BTC">
            </p>
        """
        #kraj varijable message



        f.write(message)
        f.close()







#start every XY minutes or hours
try:
    while True:
        predikcija()
        time.sleep(120) #216000 je 24 sata
except KeyboardInterrupt:
    print('Manual break by user')