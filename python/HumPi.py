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
import numpy as np
from pymongo import MongoClient
from pymongo import UpdateOne
from scipy.optimize import leastsq

MEASUREMENT_TIMEFRAME = 1  # s
BUFFERMAXSIZE = 120  # s
LOG_SIZE = 100  # measurements
MAX_DB_DOCUMENT_LENGTH = 256
DOCUMENT_WRITE_INTERVAL = 32  # measurements

AUDIO_FORMAT = format = alsaaudio.PCM_FORMAT_FLOAT_LE
CHANNELS = 2
RATE = 44100
FRAMESIZE = 1024

INITIAL_SIGNAL_AMPLITUDE = 0.2

SANITY_MAX_FREQUENCYCHANGE = 0.03  # Hz/s
SANITY_UPPER_BOUND = 50.4  # Hz
SANITY_LOWER_BOUND = 49.6  # Hz

NTP_TIMESYNC_INTERVAL = 1 * 60 * 60  # s

parser = argparse.ArgumentParser()
parser.add_argument("device",
                    help="The device to use. Get one by using the 'findYourALSADevice.py script'.",
                    type=str)
parser.add_argument("--store", help="The file in which measurments get stored", type=str)
parser.add_argument("--silent", help="Don't show measurments as output of HumPi. Only Errors / Exceptions are shown.",
                    type=int)
parser.add_argument("--serverurl", help="Path of the mongodb server")
parser.add_argument("--serveruser", help="mongodb username")
parser.add_argument("--serverpassword", help="mongodb password")

args = parser.parse_args()
devices = alsaaudio.pcms(alsaaudio.PCM_CAPTURE)
AUDIO_DEVICE_STRING = args.device
print("Using Audio Device", AUDIO_DEVICE_STRING)
if args.serverurl:
    uri = "mongodb://%s:%s@%s" % (
        quote_plus(args.serveruser), quote_plus(args.serverpassword), quote_plus(args.serverurl))
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
                self.data[x_index] = np.frombuffer(string, dtype="f")[::CHANNELS]
                self.index = x_index[-1] + 1

    def get(self, length):
        with self.lock:
            idx = np.arange(self.index - length, self.index) % self.data.size
            return self.data[idx]


# According to Wikipedia, NTP is capable of synchronizing clocks over the web with an error of 1ms. This should be sufficient.
class Log:
    def __init__(self):
        self.sync_with_ntp()
        self.data = []
        self.index = 0
        self.last_stored_date = datetime.now()
        self.offset = None
        self.last_sync = None

    def store(self, frequency, timestamp, calculation_time):
        measurement_time = timestamp + self.offset
        measurement_time_ = datetime.utcfromtimestamp(measurement_time)
        if len(self.data) >= DOCUMENT_WRITE_INTERVAL:
            self.store_to_db(self.data)
            self.data = []
        self.data.append([measurement_time_, frequency, calculation_time])
        print_to_console(
            repr(measurement_time_) + ", " + str(self.data[-1][1]) + ", " + str(calculation_time), 0)
        if time.time() - self.last_sync > NTP_TIMESYNC_INTERVAL:
            self.sync_with_ntp()
        self.index += 1

    def store_to_db(self, data):
        updates = [UpdateOne({"rate": RATE, "n_samples": {"$lt": MAX_DB_DOCUMENT_LENGTH},
                              "measurement_timeframe": MEASUREMENT_TIMEFRAME},
                             {"$push": {"data": {"ts": d[0], "freq": d[1], "calc_time": d[2]}},
                              "$min": {"first": d[0], "min_freq": d[1], "min_calc_time": d[2]},
                              "$max": {"last": d[0], "max_freq": d[1], "max_calc_time": d[2]},
                              "$inc": {"n_samples": 1}},
                             upsert=True
                             ) for d in data]
        db.bulk_write(updates, ordered=True)
        print("Stored to database")

    def save_to_disk(self):
        if args.store:
            print_to_console("========= Storing logfile =========", 4)
            with open(MEASUREMENTS_FILE, 'a') as f:
                np.savetxt(f, self.data[:self.index - 1], delimiter=",")
        self.data = np.zeros([LOG_SIZE, 2], dtype='d')
        self.index = 0

    def sync_with_ntp(self):
        c = ntplib.NTPClient()
        try:
            response = c.request('europe.pool.ntp.org', version=3)
            self.offset = response.offset - FRAMESIZE / RATE
            print_to_console("The clock is " + str(self.offset) + " seconds wrong. Changing timestamps", 5)
            self.last_sync = time.time()
        except Exception as e:
            print_to_console(str(e), 20)


class CaptureHum(threading.Thread):
    def __init__(self, threadID, name, buffer, stopSignal):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.buffer = buffer
        self.stopSignal = stopSignal

    def run(self):
        try:
            recorder = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, AUDIO_DEVICE_STRING)
        except:
            signal_handler()
        recorder.setchannels(CHANNELS)
        recorder.setrate(RATE)
        recorder.setformat(AUDIO_FORMAT)
        recorder.setperiodsize(FRAMESIZE)

        print(self.name, "* started recording")
        try:
            while (not self.stopSignal.is_set()):
                self.buffer.extend(recorder.read())
        except Exception as e:
            print_to_console(self.name + repr(e), 20)
        print(self.name, "* stopped recording")


class AnalyzeHum(threading.Thread):
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

        x = np.linspace(0, 1, num=RATE * MEASUREMENT_TIMEFRAME, endpoint=False)
        a, b, c = self.fit_sine(a, b, c, residuals, x)
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
            b = self.fit_sine(a, b, c, residuals, x)
            lastMeasurmentTime = time.time()
            log.store(b, analyze_start, lastMeasurmentTime - analyze_start)
            nrMeasurments += 1

    def fit_sine(self, a, b, c, residuals, x):
        y = self.buffer.get(RATE * MEASUREMENT_TIMEFRAME)
        plsq = leastsq(residuals, np.array([a, b, c]), args=(x, y))
        a = plsq[0][0]
        b = plsq[0][1]
        c = plsq[0][2]
        return a, b, c


def signal_handler(signal, frame):
    print(' --> Exiting HumPi')
    stopSignal.set()
    time.sleep(0.5)
    log.save_to_disk()
    time.sleep(0.5)
    sys.exit(0)


def print_to_console(message, severity):
    if args.silent >= severity:
        print(message)


log = Log()
databuffer = RingBuffer(RATE * BUFFERMAXSIZE)
stopSignal = threading.Event()
signal.signal(signal.SIGINT, signal_handler)

capture = CaptureHum(1, "Capture", databuffer, stopSignal)
capture.start()
time.sleep(MEASUREMENT_TIMEFRAME + 0.05)
analyze = AnalyzeHum(2, "Analyze", databuffer, log, stopSignal)
analyze.start()
signal.pause()
