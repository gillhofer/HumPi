#!/usr/bin/python

from __future__ import with_statement

import numpy as np
import alsaaudio
import numexpr as ne
import threading
import signal
import sys
import array
import time
import ntplib

from scipy.optimize import leastsq
from numpy import sin, pi
#from scipy.io import wavfile



MEASUREMENT_TIMEFRAME = 1 #second
BUFFERMAXSIZE = 10 #seconds
LOG_SIZE = 100 #measurments
MEASUREMENTS_FILE = "measurments.csv"
INFORMAT = alsaaudio.PCM_FORMAT_FLOAT_LE
INPUT_CHANNEL=2
CHANNELS = 1
RATE = 24000
FRAMESIZE = 512
ne.set_num_threads(3)

SANITY_MAX_FREQUENCYCHANGE = 0.03
SANITY_UPPER_BOUND = 50.4
SANITY_LOWER_BOUND = 49.6


# A multithreading compatible buffer. Tuned for maximum write_in performance

#According to 
#https://stackoverflow.com/questions/7133885/fastest-way-to-grow-a-numpy-numeric-array
# appending to python arrays is way faster than appending to numpy arrays.

class Buffer():
    def __init__(self, minSize, maxSize):
        self.data = array.array('f')
        self.lock = threading.Lock()
        self.minSize = minSize
        self.maxSize = maxSize

    def extend(self,stream):
        [length, string] = stream
        if length > 0:
            with self.lock:
                self.data.fromstring(string)
        
    def get(self, length):        
        with self.lock:
            bufferSize = self.data.buffer_info()[1]
            if bufferSize >= self.maxSize:
                #shrink buffer
                newdata = array.array('f')
                iterator = (self.data[x] for x in range(bufferSize - self.minSize, bufferSize))
                newdata.extend(iterator)
                self.data = newdata
                bufferSize = self.minSize
        iterator = (self.data[x] for x in range(bufferSize-length, bufferSize))
        return np.fromiter(iterator, dtype='f')


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
		currTime = time.time() +self.offset- calculationTime - MEASUREMENT_TIMEFRAME/2
		self.data[self.index] =  [currTime, frequency]
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
                       u'sysdefault:CARD=Device')
        recorder.setchannels(CHANNELS)
        recorder.setrate(RATE)
        recorder.setformat(INFORMAT)
        recorder.setperiodsize(FRAMESIZE)
 

        print(self.name ,"* started recording")
        try:
            while (not self.stopSignal.is_set()):
            #for i in range(0, int(RATE / FRAMESIZE * self.seconds)):
                self.buffer.extend(recorder.read())
        except Exception,e:
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
        a = 0.2
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
		plsq = leastsq(residuals, np.array([0.2,50,0]),args=(x,y))
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
	    
def signal_handler(signal, frame):
        print(' --> Exiting HumPi')
	stopSignal.set()
	time.sleep(0.5)
	log.saveToDisk()
	time.sleep(0.5)
        sys.exit(0)


log = Log()
databuffer = Buffer(RATE*MEASUREMENT_TIMEFRAME, RATE*BUFFERMAXSIZE)
stopSignal = threading.Event()
signal.signal(signal.SIGINT, signal_handler)

capture = Capture_Hum(1,"Capture", databuffer, stopSignal)
capture.start()
time.sleep(MEASUREMENT_TIMEFRAME+0.05)
analyze = Analyze_Hum(2,"Analyze", databuffer,log, stopSignal)
analyze.start()
signal.pause()



