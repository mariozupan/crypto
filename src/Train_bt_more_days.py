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

def buildLstmOutputMoreDays(dataset, target, pred_range):
    """
    # model output is next pred_range prices normalised to 10th previous closing price
    """  
    LSTM_outputs = []
    for i in range(window_len, len(dataset[target]) - pred_range):
        LSTM_outputs.append((dataset[target][i:i + pred_range].values / 
                                      dataset[target].values[i - window_len]) - 1)
    LSTM_outputs = np.array(LSTM_outputs)
    return LSTM_outputs

if __name__ == '__main__':   
    
    ########################
    # Configuration
    ######################## 
    # random seed for reproducibility
    np.random.seed(202)
    model_path = "../Output/bt_moreDay_model.h5"
    
    start_date = '20131227'
    # end_date=time.strftime("%Y%m%d")
    end_date = '20171127'
    split_date = '2017-09-25'
    
    # Our LSTM model will use previous data to predict the next day's closing price of bitcoin. 
    # We must decide how many previous days it will have access to
    window_len = 20   
    bt_epochs = 100
    bt_batch_size = 32
    num_of_neurons_lv1 = 50    
    num_of_neurons_lv2 = 25  
    activ_func="linear"
    dropout=0.5 
    loss="mean_squared_error" 
    optimizer="adam" 
    # we'll try to predict the closing price for the next N days 
    # change this value if you want to make longer/shorter prediction
    pred_range = 5
    
    ###################################
    # Getting the Bitcoin
    ###################################
    
    bt_img = urllib.request.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")    
    image_file = io.BytesIO(bt_img.read())
    bt_im = Image.open(image_file)
    width_bt_im , height_bt_im = bt_im.size
    bt_im = bt_im.resize((int(bt_im.size[0] * 0.8), int(bt_im.size[1] * 0.8)), Image.ANTIALIAS)
    
    ################
    # Data Ingestion
    ################
    
    bt_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=" + start_date + "&end=" + end_date)[0]
    # convert the date string to the correct date format
    bt_market_info = bt_market_info.assign(Date=pd.to_datetime(bt_market_info['Date']))
    # look at the first few rows
    print("ETH")
    print(bt_market_info.head())
    print('\nshape: {}'.format(bt_market_info.shape)) 
    print("\n")
    PlotUtils.plotCoinTrend(bt_market_info, bt_im,"Bitcoin")
    
    # Feature Eng
    print("Feature ENG")
    bt_market_info.columns = [bt_market_info.columns[0]] + ['bt_' + i for i in bt_market_info.columns[1:]]
    for coins in ['bt_']: 
        kwargs = { coins + 'day_diff': lambda x: (x[coins + 'Close'] - x[coins + 'Open']) / x[coins + 'Open']}
        bt_market_info = bt_market_info.assign(**kwargs)
    print('\nshape: {}'.format(bt_market_info.shape))
    print(bt_market_info.head())
    print("\n")
    PlotUtils.plotCoinTrainingTest(bt_market_info, split_date, bt_im,"bt_Close","Ethereum")
    
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
    print("Model Data")
    print('\nshape: {}'.format(model_data.shape))
    print(model_data.head())
    print("\n")
    
    # create Training and Test set    
    training_set, test_set = model_data[model_data['Date'] < split_date], model_data[model_data['Date'] >= split_date]
    # we don't need the date columns anymore
    training_set = training_set.drop('Date', 1)
    test_set = test_set.drop('Date', 1)
    
    norm_cols = [coin + metric for coin in ['bt_'] for metric in ['Close', 'Volume', 'Market Cap']]
    
    LSTM_training_inputs = ModelUtils.buildLstmInput(training_set, norm_cols, window_len)
    # model output is next price normalised to 10th previous closing price
    LSTM_training_outputs = buildLstmOutputMoreDays(training_set, 'bt_Close', pred_range)
    
    LSTM_test_inputs = ModelUtils.buildLstmInput(test_set, norm_cols, window_len)
    # model output is next price normalised to 10th previous closing price
    LSTM_test_outputs = buildLstmOutputMoreDays(test_set, 'bt_Close', pred_range)
    
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
    bt_model = ModelUtils.build_model(LSTM_training_inputs, output_size=pred_range, neurons_lv1=num_of_neurons_lv1, neurons_lv2=num_of_neurons_lv2,
                                      activ_func=activ_func, dropout=dropout, loss=loss, optimizer=optimizer)
    # train model on data
    bt_history = bt_model.fit(LSTM_training_inputs[:-pred_range], LSTM_training_outputs,
                                epochs=bt_epochs, batch_size=bt_batch_size, verbose=2, shuffle=True,
                                validation_split=0.2,
                                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min'),
                                           keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)])
    
    scores = bt_model.evaluate(LSTM_test_inputs[:-pred_range], LSTM_test_outputs, verbose=1, batch_size=bt_batch_size)
    print('\nMSE: {}'.format(scores[1]))   
    print('\nMAE: {}'.format(scores[2])) 
    print('\nR^2: {}'.format(scores[3])) 
       
    # Plot Error  
    figErr, ax1 = plt.subplots(1, 1)
    ax1.plot(bt_history.epoch, bt_history.history['loss'])
    ax1.set_title('Training Error')
    if bt_model.loss == 'mae':
        ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    # just in case you decided to change the model loss calculation
    else:
        ax1.set_ylabel('Model Loss', fontsize=12)
    ax1.set_xlabel('# Epochs', fontsize=12)
    plt.show()  
    figErr.savefig("../Output/bt_moreDays_error.png")

    #####################################
    # EVALUATE ON TEST DATA
    #####################################
     
    # little bit of reformatting the predictions to closing prices
    bt_pred_prices = ((bt_model.predict(LSTM_test_inputs)[:-pred_range][::pred_range]+1)*
                       test_set['bt_Close'].values[:-(window_len + pred_range)][::pred_range].reshape(int(np.ceil((len(LSTM_test_inputs)-pred_range)/float(pred_range))),1))
    bt_pred_prices = ((bt_model.predict(LSTM_test_inputs)[:-pred_range][::pred_range]+1)*
                       test_set['bt_Close'].values[:-(window_len + pred_range)][::pred_range].reshape(int(np.ceil((len(LSTM_test_inputs)-pred_range)/float(pred_range))),1))
    
    pred_colors = ["#FF69B4", "#5D6D7E", "#F4D03F","#A569BD","#45B39D"]
    fig, ax1 = plt.subplots(1,1, figsize=(10, 10))
    ax1.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
    ax1.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime),
             test_set['bt_Close'][window_len:], label='Actual')
    for i, (bt_pred, bt_pred) in enumerate(zip(bt_pred_prices, bt_pred_prices)):
        # Only adding lines to the legend once
        if i < pred_range:
            ax1.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime)[i*pred_range:i*pred_range+pred_range],
                     bt_pred, color=pred_colors[i%5], label="Predicted")
        else: 
            ax1.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime)[i*pred_range:i*pred_range+pred_range],
                     bt_pred, color=pred_colors[i%5])
    ax1.set_title('Test Set: '+str(pred_range)+' Timepoint Predictions',fontsize=13)
    ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
    ax1.set_xticklabels('')
    ax1.legend(bbox_to_anchor=(0.13, 1), loc=2, borderaxespad=0., prop={'size': 12})
    fig.tight_layout()
    plt.show()
    fig.savefig("../Output/bt_PredictionMoreDays.png")
