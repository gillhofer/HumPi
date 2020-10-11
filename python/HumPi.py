#!/usr/bin/python

import argparse
import signal
import sys
import threading
import time
from datetime import datetime
from urllib.parse import quote_plus

import alsaaudio
import ntplib
import numexpr
import numpy as np
from pymongo import MongoClient
from scipy.optimize import leastsq

MEASUREMENT_TIMEFRAME = 1  # s
BUFFERMAXSIZE = 120  # s
LOG_SIZE = 100  # measurements

AUDIO_FORMAT = alsaaudio.PCM_FORMAT_FLOAT_LE
CHANNELS = 1
RATE = 24000
FRAMESIZE = 1024
numexpr.set_num_threads(3)

INITIAL_SIGNAL_AMPLITUDE = 0.2

SANITY_MAX_FREQUENCYCHANGE = 0.03  # Hz/s
SANITY_UPPER_BOUND = 50.4  # Hz
SANITY_LOWER_BOUND = 49.6  # Hz

NTP_TIMESYNC_INTERVAL = 1 * 60 * 60  # s

parser = argparse.ArgumentParser()
parser.add_argument("device",
                    help="The device to use. Try some (1-10), or get one by using the 'findYourALSADevice.py script'.",
                    type=int)
parser.add_argument("--store", help="The file in which measurments get stored", type=str)
# parser.add_argument("--sendserver", help="The server URL submitting to: e.g. \"http://192.168.3.1:8080\"", type=str)
# parser.add_argument("--meterid", help="The name for the meter to use", type=str)
# parser.add_argument("--apikey", help="The API-Key to use", type=str)
parser.add_argument("--silent", help="Don't show measurments as output of HumPi. Only Errors / Exceptions are shown.",
                    type=int)
parser.add_argument("--serverurl", help="Path of the mongodb server")
parser.add_argument("--serveruser", help="mongodb username")
parser.add_argument("--serverpassword", help="mongodb password")

args = parser.parse_args()
devices = alsaaudio.pcms(alsaaudio.PCM_CAPTURE)
AUDIO_DEVICE_STRING = devices[args.device - 1]
print("Using Audio Device", AUDIO_DEVICE_STRING)
if args.serverurl:
    uri = "mongodb://%s:%s@%s" % (
        quote_plus(args.serveruser), quote_plus(args.serverpassword), quote_plus("192.168.3.1:27017"))
    client = MongoClient(uri)
    db = client.gridfrequency.raw
else:
    print("I don't send any data")
if args.store:
    MEASUREMENTS_FILE = args.store
    print("Storing measurments into", MEASUREMENTS_FILE, "by appending to it.")
else:
    print("I don't store any data")

if not args.silent:
    args.silent = 0


class RingBuffer:
    def __init__(self, maxSize):
        self.data = np.zeros(maxSize, dtype='f')
        self.index = 0
        self.lock = threading.Lock()

    def extend(self, stream):
        [length, string] = stream
        if length > 0:
            x_index = np.arange(self.index, self.index + length) % self.data.size
            with self.lock:
                self.data[x_index] = np.frombuffer(string, dtype='f')
                self.index = x_index[-1] + 1

    def get(self, length):
        with self.lock:
            idx = np.arange(self.index - length, self.index) % self.data.size
            return self.data[idx]


# According to Wikipedia, NTP is capable of synchronizing clocks over the web with an error of 1ms. This should be sufficient.
class Log():
    def __init__(self):
        self.syncWithNTP()
        self.data = []
        self.index = 0
        self.last_stored_date = datetime.now()

    def store(self, frequency, timestamp, calculationTime):
        measurmentTime = timestamp + self.offset
        measurmentTime_ = datetime.utcfromtimestamp(measurmentTime)
        if len(self.data) > 0 and measurmentTime_.minute != self.data[-1][0].minute:
            self.store_to_db(self.data)
            self.data = []
        self.data.append([measurmentTime_, frequency, calculationTime])
        printToConsole(
            repr(measurmentTime_) + ", " + str(self.data[-1][1]) + ", " + str(calculationTime), 0)
        if time.time() - self.lastSync > NTP_TIMESYNC_INTERVAL:
            self.syncWithNTP()
        self.index += 1

    def store_to_db(self, data):
        data_dict = {"n_samples": len(data)}
        data_dict["first"] = data[0][0]
        data_dict["last"] = data[-1][0]
        data_dict["rate"] = RATE
        data_dict["framesize"] = FRAMESIZE
        data_dict["measurement_timeframe"] = MEASUREMENT_TIMEFRAME
        data_dict.update({"ts": [d[0] for d in data], "frequ": [d[1] for d in data], "calc_time": [d[2] for d in data]})
        db.insert_one(data_dict)
        print("Stored to database")

    def saveToDisk(self):
        if args.store:
            printToConsole("========= Storing logfile =========", 4)
            with open(MEASUREMENTS_FILE, 'a') as f:
                np.savetxt(f, self.data[:self.index - 1], delimiter=",")
        self.data = np.zeros([LOG_SIZE, 2], dtype='d')
        self.index = 0

    def syncWithNTP(self):
        c = ntplib.NTPClient()
        try:
            response = c.request('europe.pool.ntp.org', version=3)
            self.offset = response.offset - FRAMESIZE / RATE
            printToConsole("The clock is " + str(self.offset) + " seconds wrong. Changing timestamps", 5)
            self.lastSync = time.time()
        except Exception as e:
            printToConsole(str(e), 20)


class Capture_Hum(threading.Thread):
    def __init__(self, threadID, name, buffer, stopSignal):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.buffer = buffer
        self.stopSignal = stopSignal

    def run(self):
        recorder = alsaaudio.PCM(alsaaudio.PCM_CAPTURE,
                                 alsaaudio.PCM_NORMAL,
                                 AUDIO_DEVICE_STRING)
        recorder.setchannels(CHANNELS)
        recorder.setrate(RATE)
        recorder.setformat(AUDIO_FORMAT)
        recorder.setperiodsize(FRAMESIZE)

        print(self.name, "* started recording")
        try:
            while (not self.stopSignal.is_set()):
                self.buffer.extend(recorder.read())
        except Exception as e:
            printToConsole(self.name + str(e), 20)
        print(self.name, "* stopped recording")


class Analyze_Hum(threading.Thread):
    def __init__(self, threadID, name, buffer, log, stopSignal):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.buffer = buffer
        self.log = log
        self.stopSignal = stopSignal

    def run(self):

        def residuals(p, x, y):
            A, k, theta = p
            x = x
            y = y
            return y - A * np.sin(2 * np.pi * k * x + theta)

        print(self.name, "* Started measurements")
        a = INITIAL_SIGNAL_AMPLITUDE
        b = 50
        c = 0

        lastMeasurmentTime = 0
        x = np.linspace(0, 1, num=RATE * MEASUREMENT_TIMEFRAME, endpoint=False)
        y = self.buffer.get(RATE * MEASUREMENT_TIMEFRAME)
        plsq = leastsq(residuals, np.array([a, b, c]), args=(x, y))
        a = plsq[0][0]
        b = plsq[0][1]
        c = plsq[0][2]
        nrMeasurments = 0
        TIME_OFFSET = time.time()

        while (not self.stopSignal.is_set()):
            time.sleep(time.time() % 0.5)
            analyze_start = time.time()
            if (nrMeasurments > 200):
                nrMeasurments = 0
                TIME_OFFSET = analyze_start
            x = np.linspace(analyze_start - TIME_OFFSET - 1, analyze_start - TIME_OFFSET,
                            num=RATE * MEASUREMENT_TIMEFRAME, endpoint=False)
            y = self.buffer.get(RATE * MEASUREMENT_TIMEFRAME)
            plsq = leastsq(residuals, np.array([a, b, c]), args=(x, y))
            # if plsq[0][1] < SANITY_LOWER_BOUND or plsq[0][1] > SANITY_UPPER_BOUND:
            #    printToConsole(str(plsq[0][1]) + "looks fishy, trying again.", 5)
            #    plsq = leastsq(residuals, np.array([INITIAL_SIGNAL_AMPLITUDE, 50, 0]), args=(x, y))
            # if plsq[0][1] < SANITY_LOWER_BOUND or plsq[0][1] > SANITY_UPPER_BOUND:
            #    printToConsole("Now got " + str(plsq[0][1]) + ". Buffer data is corrupt, need new data", 5)
            #    time.sleep(MEASUREMENT_TIMEFRAME)
            #    printToConsole("Back up, continue measurments", 5)
            # else:
            frqChange = np.abs(plsq[0][1] - b)
            frqChangeTime = time.time() - lastMeasurmentTime
            # plt.plot(x,y, x,plsq[0][0] * sin(2 * pi * plsq[0][1] * x + plsq[0][2]))
            # plt.show()
            if frqChange / frqChangeTime < SANITY_MAX_FREQUENCYCHANGE:
                a = plsq[0][0]
                b = plsq[0][1]
                c = plsq[0][2]
                lastMeasurmentTime = time.time()
                log.store(b, analyze_start, lastMeasurmentTime - analyze_start)
            else:
                printToConsole(
                    "Frequency Change too big " + str(frqChange) + ", " + str(frqChangeTime) + ", " + str(
                        frqChange / frqChangeTime) + "," + "Buffer is probably corrupt", 5)
                time.sleep(MEASUREMENT_TIMEFRAME)
            nrMeasurments += 1


def signal_handler(signal, frame):
    print(' --> Exiting HumPi')
    stopSignal.set()
    time.sleep(0.5)
    log.saveToDisk()
    time.sleep(0.5)
    sys.exit(0)


def printToConsole(message, severity):
    if args.silent >= severity:
        print(message)


log = Log()
databuffer = RingBuffer(RATE * BUFFERMAXSIZE)
stopSignal = threading.Event()
signal.signal(signal.SIGINT, signal_handler)

capture = Capture_Hum(1, "Capture", databuffer, stopSignal)
capture.start()
time.sleep(MEASUREMENT_TIMEFRAME + 0.05)
analyze = Analyze_Hum(2, "Analyze", databuffer, log, stopSignal)
analyze.start()
signal.pause()
