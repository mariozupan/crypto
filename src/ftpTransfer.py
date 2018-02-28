



import ftplib

import urllib 



import time, threading

def transfer():

	session = ftplib.FTP('pascal.hr','pascalpo','Yxw140b*')
	file = open('../Output/cijenaBTC.html','rb')                  # file to send
	session.storbinary('STOR httpdocs\cijenaBTC.cshtml', file)     # send the file

	file = open('../Output/BitcoinTrend.png','rb')                  # file to send
	session.storbinary('STOR httpdocs\BTCoutput\BitcoinTrend.png', file)     # send the file

	file = open('../Output/bt_TrainTest.png','rb')                  # file to send
	session.storbinary('STOR httpdocs\BTCoutput\BitcoinTrainTest.png', file)     # send the file

	file = open('../Output/bt_performanceTraining.png','rb')                  # file to send
	session.storbinary('STOR httpdocs\BTCoutput\BTCperformanceTraining.png', file)     # send the file

	file = open('../Output/bt_PredictionMoreDays.png','rb')                  # file to send
	session.storbinary('STOR httpdocs\BTCoutput\BTCPredictionMoreDays.png', file)     # send the file

	file = open('../Output/bt_moreDays_error.png','rb')                  # file to send
	session.storbinary('STOR httpdocs\BTC_output\BTCmoreDays_error.png', file)     # send the file

	file.close()                                    # close file and FTP
	session.quit()




#start every 24r hours
try:
    while True:
        transfer()
        print('periodicno snimanje: WAIT!!!')
        time.sleep(120) #216000 je 24 sata
except KeyboardInterrupt:
    print('Manual break by user')