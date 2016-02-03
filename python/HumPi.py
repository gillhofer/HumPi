#!/usr/bin/python

from __future__ import with_statement
from __future__ import print_function

import numpy as np
import alsaaudio
import numexpr
import threading
import signal
import sys
import time
import ntplib
import argparse
import requests

#import matplotlib.pyplot as plt

from scipy.optimize import leastsq
from numpy import sin, pi

MEASUREMENT_TIMEFRAME = 1 #s
BUFFERMAXSIZE = 120 #s
LOG_SIZE = 100 #measurements

AUDIO_FORMAT = alsaaudio.PCM_FORMAT_FLOAT_LE
CHANNELS = 1
RATE = 24000
FRAMESIZE = 1024
numexpr.set_num_threads(3)

INITIAL_SIGNAL_AMPLITUDE = 0.2

SANITY_MAX_FREQUENCYCHANGE = 0.03 #Hz/s
SANITY_UPPER_BOUND = 50.4 #Hz
SANITY_LOWER_BOUND = 49.6 #Hz

NTP_TIMESYNC_INTERVAL = 1*60*60#s

parser = argparse.ArgumentParser()
parser.add_argument("device", help="The device to use. Try some (1-10), or get one by using the 'findYourALSADevice.py script'.",  type=int)
parser.add_argument("--store", help="The file in which measurments get stored", type=str)
parser.add_argument("--sendserver", help="The server URL submitting to: e.g. \"http://192.168.3.1:8080\"", type=str)
parser.add_argument("--meterid", help="The name for the meter to use", type=str)
parser.add_argument("--apikey", help="The API-Key to use", type=str)
parser.add_argument("--silent", help="Don't show measurments as output of HumPi. Only Errors / Exceptions are shown.", type=int)

args = parser.parse_args()
devices = alsaaudio.pcms(alsaaudio.PCM_CAPTURE)
AUDIO_DEVICE_STRING = devices[args.device-1]
print("Using Audio Device", AUDIO_DEVICE_STRING)
if args.sendserver:
	if not args.meterid:	
		print("Please also provide a meter name by specifying the --meterid option")
		sys.exit(0)		
	if not args.apikey:
		print("Please also provide an API-Key by specifying the --apikey option.")
		sys.exit(0)		
	SERVER_URL = args.sendserver + '/api/submit/' + args.meterid
	API_KEY = args.apikey
	print("Sending to netzsinus using the URL", SERVER_URL, "with given API-Key.")
else:
	print("I don't send any data")
if args.store:
	MEASUREMENTS_FILE = args.store
	print("Storing measurments into", MEASUREMENTS_FILE, "by appending to it.")
else:
	print("I don't store any data")


class RingBuffer():
    def __init__(self, maxSize):
    	self.data = np.zeros(maxSize, dtype='f')
        self.index = 0
        self.lock = threading.Lock()

    def extend(self, stream):
       [length, string] = stream
       if length > 0:
           x_index = np.arange(self.index,self.index + length) % self.data.size
           with self.lock :
              self.data[x_index] = np.fromstring(string, dtype='f')
              self.index = x_index[-1] + 1

    def get(self, length):
       with self.lock:
          idx = np.arange(self.index-length, self.index) % self.data.size
          return self.data[idx]

# According to Wikipedia, NTP is capable of synchronizing clocks over the web with an error of 1ms. This should be sufficient.  
class Log():
	def __init__(self):
		self.syncWithNTP()		
		self.data = np.zeros([LOG_SIZE,2],dtype='d')
		self.index = 0
		if args.sendserver:		
			self.session = requests.Session()
			self.session.headers.update({
				'Content-Type': 'application/json',
				'X-API-KEY': API_KEY})
    

	def store(self,frequency, calculationTime):
		measurmentTime = time.time() + self.offset - calculationTime
		self.data[self.index] =  [measurmentTime, frequency]
		printToConsole(time.ctime(self.data[self.index,0]) + ", " +str(self.data[self.index,1]) + ", " + str(calculationTime),0)
		if args.sendserver:
			payload = {
				"Value": frequency,
				"Timestamp": measurmentTime}
			try: 
				r= self.session.post(SERVER_URL, json=payload)
				if not r.status_code == 200:
					printToConsole("Got HTTP status code" + r.status_code, 10)
			except Exception as e:
				printToConsole(str(e), 20)
			self.index += 1
		if self.index==LOG_SIZE:
			self.saveToDisk()
		if time.time() - self.lastSync > NTP_TIMESYNC_INTERVAL:
			self.syncWithNTP()	
		
	
	def saveToDisk(self):
		if args.store:		
			printToConsole("========= Storing logfile =========",4)
			with open(MEASUREMENTS_FILE, 'a') as f:
				np.savetxt(f, self.data[:self.index-1],delimiter=",")
		self.data = np.zeros([LOG_SIZE,2],dtype='d')
		self.index =0

	def syncWithNTP(self):
		c = ntplib.NTPClient()
		response = c.request('europe.pool.ntp.org', version=3)
		self.offset = response.offset - FRAMESIZE/RATE
		printToConsole("The clock is " + str(self.offset) +  " seconds wrong. Changing timestamps", 5)		
		self.lastSync = time.time()


class Capture_Hum (threading.Thread):
    def __init__(self, threadID, name, buffer, stopSignal):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.buffer= buffer
        self.stopSignal = stopSignal

    def run(self):
        recorder=alsaaudio.PCM(alsaaudio.PCM_CAPTURE,
                       alsaaudio.PCM_NORMAL, 
                       AUDIO_DEVICE_STRING)
        recorder.setchannels(CHANNELS)
        recorder.setrate(RATE)
        recorder.setformat(AUDIO_FORMAT)
        recorder.setperiodsize(FRAMESIZE)
 

        print(self.name ,"* started recording")
        try:
            while (not self.stopSignal.is_set()):
                self.buffer.extend(recorder.read())
        except Exception as e:
            printToConsole(self.name + str(e), 20)
        print(self.name ,"* stopped recording")


class Analyze_Hum(threading.Thread):
    def __init__(self, threadID, name, buffer,log, stopSignal):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.buffer= buffer
        self.log = log
        self.stopSignal = stopSignal
    
    def run(self):
        def residuals(p,x,y):
            A, k, theta = p
            x = x
            y = y
            return numexpr.evaluate('y - A * sin(2 * pi * k * x + theta)')
        
        print(self.name ,"* Started measurements")
        a = INITIAL_SIGNAL_AMPLITUDE
        b = 50
        c = 0
        
        lastMeasurmentTime = 0
        x = np.linspace(0, 1, num=RATE*MEASUREMENT_TIMEFRAME, endpoint=False)
        y = self.buffer.get(RATE*MEASUREMENT_TIMEFRAME)
        plsq = leastsq(residuals, np.array([a,b,c]),args=(x,y))
        a = plsq[0][0]
        b = plsq[0][1]
        c = plsq[0][2]
	nrMeasurments = 0
	TIME_OFFSET = time.time()
		
        while (not self.stopSignal.is_set()):
            analyze_start = time.time()
            if (nrMeasurments > 200):
                nrMeasurments = 0
                TIME_OFFSET=analyze_start
            x = np.linspace(analyze_start-TIME_OFFSET-1, analyze_start-TIME_OFFSET, num=RATE*MEASUREMENT_TIMEFRAME, endpoint=False)
            y = self.buffer.get(RATE*MEASUREMENT_TIMEFRAME)
            plsq = leastsq(residuals, np.array([a,b,c]),args=(x,y))
            if plsq[0][1] < SANITY_LOWER_BOUND or plsq[0][1] > SANITY_UPPER_BOUND:
                printToConsole(plsq[0][1] + "looks fishy, trying again." , 5)
                plsq = leastsq(residuals, np.array([INITIAL_SIGNAL_AMPLITUDE,50,0]),args=(x,y))
            if plsq[0][1] < SANITY_LOWER_BOUND or plsq[0][1] > SANITY_UPPER_BOUND:
                printToConsole("Now got", plsq[0][1], ". Buffer data is corrupt, need new data", 5)
                time.sleep(MEASUREMENT_TIMEFRAME)
                printToConsole("Back up, continue measurments",5)	    
            else:
                frqChange = np.abs(plsq[0][1] - b)
                frqChangeTime = time.time() - lastMeasurmentTime
                #plt.plot(x,y, x,plsq[0][0] * sin(2 * pi * plsq[0][1] * x + plsq[0][2]))
	        #plt.show()
                if frqChange/frqChangeTime <  SANITY_MAX_FREQUENCYCHANGE:
                    a = plsq[0][0]
                    b = plsq[0][1]
                    c = plsq[0][2]
                    lastMeasurmentTime = time.time()
                    log.store(b,lastMeasurmentTime-analyze_start)
                else: 
                    printToConsole("Frequency Change too big " +  str(frqChange) + ", "+ str(frqChangeTime) +", " + str(frqChange / frqChangeTime) + "," + "Buffer is probably corrupt", 5)
                    time.sleep(MEASUREMENT_TIMEFRAME)
		nrMeasurments+=1
	    
def signal_handler(signal, frame):
	print(' --> Exiting HumPi')
	stopSignal.set()
	time.sleep(0.5)
	log.saveToDisk()
	time.sleep(0.5)
	sys.exit(0)

def printToConsole(message, severity):
	if not args.silent >=severity:
		print(message) 


log = Log()
databuffer = RingBuffer(RATE*BUFFERMAXSIZE)
stopSignal = threading.Event()
signal.signal(signal.SIGINT, signal_handler)


capture = Capture_Hum(1,"Capture", databuffer, stopSignal)
capture.start()
time.sleep(MEASUREMENT_TIMEFRAME+0.05)
analyze = Analyze_Hum(2,"Analyze", databuffer,log, stopSignal)
analyze.start()
signal.pause()






