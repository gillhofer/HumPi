#!/usr/bin/python

from __future__ import with_statement
from __future__ import print_function

import numpy as np
import alsaaudio
import numexpr as ne
import threading
import signal
import sys
import array
import time
import ntplib

import argparse

import matplotlib.pyplot as plt

from scipy.optimize import leastsq
from numpy import sin, pi

MEASUREMENT_TIMEFRAME = 1 #second
BUFFERMAXSIZE = 120 #seconds
LOG_SIZE = 100 #measurements
MEASUREMENTS_FILE = "measurements.csv"
INFORMAT = alsaaudio.PCM_FORMAT_FLOAT_LE
CHANNELS = 1
RATE = 24000
FRAMESIZE = 512
ne.set_num_threads(3)

INITIAL_SIGNAL_AMPLITUDE = 0.2

SANITY_MAX_FREQUENCYCHANGE = 0.03 #Hz per Second
SANITY_UPPER_BOUND = 50.4
SANITY_LOWER_BOUND = 49.6

parser = argparse.ArgumentParser()
parser.add_argument("-d","--device", help="The device to use. Try some (1-10), or get one by using the 'findYourALSADevice.py script'.",  type=int)
args = parser.parse_args()
devices = alsaaudio.pcms(alsaaudio.PCM_CAPTURE)
AUDIO_DEVICE_STRING = devices[args.device-1]
print("Using Audio Device", AUDIO_DEVICE_STRING)

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
		self.offset = self.getoffset()
		print("The clock is ", self.offset, "seconds wrong. Changing timestamps")
		self.data = np.zeros([LOG_SIZE,2],dtype='d')
		self.index =0
        
    
	def getoffset(self):
		c = ntplib.NTPClient()
		response = c.request('europe.pool.ntp.org', version=3)
		return response.offset

	def store(self,frequency, calculationTime):
		measurmentTime = time.time() + self.offset - calculationTime
		self.data[self.index] =  [measurmentTime, frequency]
		print(time.ctime(self.data[self.index,0]), self.data[self.index,1], calculationTime)
		self.index += 1
		if self.index==LOG_SIZE:
			# send it to Netzsinus
			# for now save it to disk.
			self.saveToDisk()
		

	def saveToDisk(self):
		print("========= Storing logfile ========= ")
		with open(MEASUREMENTS_FILE, 'a') as f:
			np.savetxt(f, self.data[:self.index-1],delimiter=",")
		self.data = np.zeros([LOG_SIZE,2],dtype='d')
		self.index =0


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
        recorder.setformat(INFORMAT)
        recorder.setperiodsize(FRAMESIZE)
 

        print(self.name ,"* started recording")
        try:
            while (not self.stopSignal.is_set()):
                self.buffer.extend(recorder.read())
        except Exception as e:
            print(self.name ,str(e))
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
            err = ne.evaluate('y - A * sin(2 * pi * k * x + theta)')
            #err = y - A * sin(2 * pi * k * x + theta)
            return err
        
        print(self.name ,"* Started measurements")
        x = np.divide(np.arange(RATE*MEASUREMENT_TIMEFRAME),np.array(RATE,dtype=float))
        a = INITIAL_SIGNAL_AMPLITUDE
        b = 50
        c = 0
        
        lastMeasurmentTime = 0
        y = self.buffer.get(RATE*MEASUREMENT_TIMEFRAME)
        plsq = leastsq(residuals, np.array([a,b,c]),args=(x,y))
        a = plsq[0][0]
        b = plsq[0][1]
        c = plsq[0][2]
		
        while (not self.stopSignal.is_set()):
            analyze_start = time.time()
            y = self.buffer.get(RATE*MEASUREMENT_TIMEFRAME)
            plsq = leastsq(residuals, np.array([a,b,c]),args=(x,y))
            if plsq[0][1] < SANITY_LOWER_BOUND or plsq[0][1] > SANITY_UPPER_BOUND:
                print("Trying again", plsq[0][1], "looks fishy")
                plsq = leastsq(residuals, np.array([INITIAL_SIGNAL_AMPLITUDE,50,0]),args=(x,y))
            if plsq[0][1] < SANITY_LOWER_BOUND or plsq[0][1] > SANITY_UPPER_BOUND:
                print("Now got", plsq[0][1], "Buffer data is Corrupt, need new data")
                time.sleep(MEASUREMENT_TIMEFRAME)
                print("Back up, continue measuring")	    
            else:
                frqChange = np.abs(plsq[0][1] - b)
                frqChangeTime = time.time() - lastMeasurmentTime
                if frqChange/frqChangeTime <  SANITY_MAX_FREQUENCYCHANGE:
                    a = plsq[0][0]
                    b = plsq[0][1]
                    c = plsq[0][2]
                    lastMeasurmentTime = time.time()
                    log.store(b,lastMeasurmentTime-analyze_start)
                else: 
                    print("Frequency Change too big", frqChange, frqChangeTime, frqChange / frqChangeTime, "Buffer is probably corrupt" )
                    time.sleep(MEASUREMENT_TIMEFRAME)
            #plt.plot(x,y, x,plsq[0][0] * sin(2 * pi * plsq[0][1] * x + plsq[0][2]))
	    #plt.show()
	    
def signal_handler(signal, frame):
	print(' --> Exiting HumPi')
	stopSignal.set()
	time.sleep(0.5)
	log.saveToDisk()
	time.sleep(0.5)
	sys.exit(0)


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



