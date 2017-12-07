'''
Created on 24 nov 2017

@author: mantica

'''

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from PIL import Image
import io
import urllib

import keras
from PlotUtils import PlotUtils
from ModelUtils import ModelUtils

if __name__ == '__main__':   
    
    ########################
    # Configuration
    ######################## 
    # random seed for reproducibility
    # np.random.seed(202)
    model_path = "../Output/eth_model.h5"
    
    start_date = '20150808'
    # end_date=time.strftime("%Y%m%d")
    end_date = '20171127'
    split_date = '2017-09-25'
    
    # Our LSTM model will use previous data to predict the next day's closing price of eth. 
    # We must decide how many previous days it will have access to
    window_len = 20   
    eth_epochs = 100
    eth_batch_size = 32
    num_of_neurons_lv1 = 50    
    num_of_neurons_lv2 = 25  
    
    ###################################
    # Getting the Eth
    ###################################
    
    eth_img = urllib.request.urlopen("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ethereum_logo_2014.svg/256px-Ethereum_logo_2014.svg.png")    
    image_file = io.BytesIO(eth_img.read())
    eth_im = Image.open(image_file)
    width_eth_im , height_eth_im = eth_im.size
    eth_im = eth_im.resize((int(eth_im.size[0] * 0.8), int(eth_im.size[1] * 0.8)), Image.ANTIALIAS)
    
    ################
    # Data Ingestion
    ################
    
    # get market info for ethereum from the start of 2016 to the current day
    eth_market_info = pd.read_html("https://coinmarketcap.com/currencies/ethereum/historical-data/?start=" + start_date + "&end=" + end_date)[0]
    # convert the date string to the correct date format
    eth_market_info = eth_market_info.assign(Date=pd.to_datetime(eth_market_info['Date']))
    # look at the first few rows
    print("ETH")
    print(eth_market_info.head())
    print('\nshape: {}'.format(eth_market_info.shape)) 
    print("\n")
    PlotUtils.plotCoinTrend(eth_market_info, eth_im,"Ethereum")
    
    # Feature Eng
    print("Feature ENG")
    eth_market_info.columns = [eth_market_info.columns[0]] + ['eth_' + i for i in eth_market_info.columns[1:]]
    for coins in ['eth_']: 
        kwargs = { coins + 'day_diff': lambda x: (x[coins + 'Close'] - x[coins + 'Open']) / x[coins + 'Open']}
        eth_market_info = eth_market_info.assign(**kwargs)
    print('\nshape: {}'.format(eth_market_info.shape))
    print(eth_market_info.head())
    print("\n")
    PlotUtils.plotCoinTrainingTest(eth_market_info, split_date, eth_im,"eth_Close","Ethereum")
    
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
    for coins in ['eth_']: 
        kwargs = { coins + 'close_off_high': lambda x: 2 * (x[coins + 'High'] - x[coins + 'Close']) / (x[coins + 'High'] - x[coins + 'Low']) - 1,
                coins + 'volatility': lambda x: (x[coins + 'High'] - x[coins + 'Low']) / (x[coins + 'Open'])}
        eth_market_info = eth_market_info.assign(**kwargs)
    model_data = eth_market_info[['Date'] + [coin + metric for coin in [ 'eth_'] 
                                   for metric in ['Close', 'Volume', 'close_off_high', 'volatility', 'day_diff', 'Market Cap']]]
    
    # need to reverse the data frame so that subsequent rows represent later timepoints
    model_data = model_data.sort_values(by='Date')
    print("Model Data")
    print('\nshape: {}'.format(model_data.shape))
    print(model_data.head())
    print("\n")
    
    # create Training and Test set    
    training_set, test_set = model_data[model_data['Date'] < split_date], model_data[model_data['Date'] >= split_date]
    # we don't need the date columns anymore
    training_set = training_set.drop('Date', 1)
    test_set = test_set.drop('Date', 1)
    
    norm_cols = [coin + metric for coin in ['eth_'] for metric in ['Close', 'Volume', 'Market Cap']]
    
    LSTM_training_inputs = ModelUtils.buildLstmInput(training_set, norm_cols, window_len)
    # model output is next price normalised to 10th previous closing price
    LSTM_training_outputs = ModelUtils.buildLstmOutput(training_set, 'eth_Close', window_len)
    
    LSTM_test_inputs = ModelUtils.buildLstmInput(test_set, norm_cols, window_len)
    # model output is next price normalised to 10th previous closing price
    LSTM_test_outputs = ModelUtils.buildLstmOutput(test_set, 'eth_Close', window_len)
    
    print("\nNumber Of Input Training's sequences: {}".format(len(LSTM_training_inputs)))
    print("\nNumber Of Output Training's sequences: {}".format(len(LSTM_training_outputs)))
    print("\nNumber Of Input Test's sequences: {}".format(len(LSTM_test_inputs)))
    print("\nNumber Of Output Test's sequences: {}".format(len(LSTM_test_outputs)))

    # I find it easier to work with numpy arrays rather than pandas dataframes
    # especially as we now only have numerical data
    LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
    LSTM_training_inputs = np.array(LSTM_training_inputs)
    
    LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
    LSTM_test_inputs = np.array(LSTM_test_inputs)
    
    #####################################
    # Modeling
    #####################################
    
    # initialise model architecture
    eth_model = ModelUtils.build_model(LSTM_training_inputs, output_size=1, neurons_lv1=num_of_neurons_lv1, neurons_lv2=num_of_neurons_lv2)
    # train model on data
    eth_history = eth_model.fit(LSTM_training_inputs, LSTM_training_outputs,
                                epochs=eth_epochs, batch_size=eth_batch_size, verbose=2, shuffle=True,
                                validation_split=0.2,
                                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min'),
                                           keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)])
    
    # We've just built an LSTM model to predict tomorrow's Ethereum closing price.
    scores = eth_model.evaluate(LSTM_test_inputs, LSTM_test_outputs, verbose=1, batch_size=eth_batch_size)
    print('\nMSE: {}'.format(scores[1]))   
    print('\nMAE: {}'.format(scores[2])) 
    print('\nR^2: {}'.format(scores[3])) 
       
    # Plot Error  
    figErr, ax1 = plt.subplots(1, 1)
    ax1.plot(eth_history.epoch, eth_history.history['loss'])
    ax1.set_title('Training Error')
    if eth_model.loss == 'mae':
        ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    # just in case you decided to change the model loss calculation
    else:
        ax1.set_ylabel('Model Loss', fontsize=12)
    ax1.set_xlabel('# Epochs', fontsize=12)
    plt.show()  
    figErr.savefig("../Output/eth_error.png")

    #####################################
    # EVALUATE ON TEST DATA
    #####################################
    
    # Plot Performance
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    ax1.set_xticks([datetime.date(2017, i + 1, 1) for i in range(12)])
    ax1.set_xticklabels([datetime.date(2017, i + 1, 1).strftime('%b %d %Y')  for i in range(12)])
    ax1.plot(model_data[model_data['Date'] >= split_date]['Date'][window_len:].astype(datetime.datetime),
             test_set['eth_Close'][window_len:], label='Actual')
    ax1.plot(model_data[model_data['Date'] >= split_date]['Date'][window_len:].astype(datetime.datetime),
             ((np.transpose(eth_model.predict(LSTM_test_inputs)) + 1) * test_set['eth_Close'].values[:-window_len])[0],
             label='Predicted')
    ax1.annotate('MAE: %.4f' % np.mean(np.abs((np.transpose(eth_model.predict(LSTM_test_inputs)) + 1) - \
                (test_set['eth_Close'].values[window_len:]) / (test_set['eth_Close'].values[:-window_len]))),
                 xy=(0.75, 0.9), xycoords='axes fraction',
                xytext=(0.75, 0.9), textcoords='axes fraction')
    ax1.set_title('Test Set: Single Timepoint Prediction', fontsize=13)
    ax1.set_ylabel('Ethereum Price ($)', fontsize=12)
    ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
    plt.show()    
    fig.savefig("../Output/eth_performanceTraining.png")
