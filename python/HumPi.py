import argparse
import csv
import pickle
import signal
import sys
import threading
import time
from datetime import datetime
from urllib.parse import quote_plus

import alsaaudio
import bson
import ntplib
import numpy as np
from pymongo import MongoClient
from pymongo import UpdateOne
from scipy.optimize import leastsq

MEASUREMENT_TIMEFRAME = 1  # s
BUFFERMAXSIZE = 500  # s
LOG_SIZE = 100  # measurements
MAX_DB_DOCUMENT_LENGTH = 256
MAX_DB_AUDIO_MINUTES = 60
DOCUMENT_WRITE_INTERVAL = 32  # measurements
COMPRESSED_AUDIO_SIZE = 4096 * 2
AUDIO_STORE_INTERVAL = 15

AUDIO_FORMAT = format = alsaaudio.PCM_FORMAT_FLOAT_LE
CHANNELS = 2
RATE = 192000
FRAMESIZE = 1024  # my hardware seeems to ignore this value

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
    audiodb = client.gridfrequency.audio
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
        self.read_length = []
        self.last_timestamp = None

    def extend(self, stream):
        [length, string] = stream
        self.read_length.append(length)
        if length > 0:
            x_index = np.arange(self.index, self.index + length) % self.data.size
            data = np.frombuffer(string, dtype="f")[::CHANNELS]
            with self.lock:
                self.last_timestamp = time.time()
                self.data[x_index] = data
                self.index = x_index[-1] + 1

    def get(self, length, with_time_stamp=False):
        with self.lock:
            index = self.index
        idx = np.arange(index - length, index) % self.data.size
        if not with_time_stamp:
            return self.data[idx]
        else:
            return self.data[idx], self.last_timestamp


# According to Wikipedia, NTP is capable of synchronizing clocks over the web with an error of 1ms. This should be sufficient.
class Log:
    def __init__(self):
        self.data = []
        self.last_stored_date = datetime.now()
        self.offset = None
        self.last_sync = None
        self.sync_with_ntp()

    def store_audio_to_db(self):
        y, timestamp = databuffer.get(RATE * AUDIO_STORE_INTERVAL)
        start_timestamp = timestamp - 1 / RATE * AUDIO_STORE_INTERVAL
        p = np.fft.rfft(y, COMPRESSED_AUDIO_SIZE * 2)
        update = UpdateOne({"array_length": int(RATE * AUDIO_STORE_INTERVAL),
                            "compressed_length": COMPRESSED_AUDIO_SIZE,
                            "audio_minutes": {"$lt": MAX_DB_AUDIO_MINUTES}},
                           {"$min": {"first": start_timestamp},
                            "$max": {"last": start_timestamp},
                            "$inc": {"audio_minutes": AUDIO_STORE_INTERVAL},
                            "$push": {"rfft-binary": bson.Binary(pickle.dumps(p[:COMPRESSED_AUDIO_SIZE])),
                                      "timestamp": start_timestamp},
                            }, upsert=True)
        audiodb.write(update)
        if time.time() - databuffer.last_timestamp > AUDIO_STORE_INTERVAL:
            self.store_audio_to_db()
        print("Stored audio to database")

    def store(self, frequency, timestamp, calculation_time):
        measurement_time = timestamp + self.offset
        measurement_time_ = datetime.utcfromtimestamp(measurement_time)
        if len(self.data) >= DOCUMENT_WRITE_INTERVAL:
            if args.serverurl:
                self.store_to_db(self.data)
            if args.store:
                self.save_to_disk()
            self.data = []

        self.data.append([measurement_time_, frequency, calculation_time])
        print_to_console(
            repr(measurement_time_) + ", " + str(self.data[-1][1]) + ", " + str(calculation_time), 0)
        if time.time() - self.last_sync > NTP_TIMESYNC_INTERVAL:
            self.sync_with_ntp()

    def store_to_db(self, data: list):
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
                csv_writer = csv.writer(f, delimiter=',')
                for d in self.data:
                    csv_writer.writerow(",".join(d))

    def sync_with_ntp(self):
        c = ntplib.NTPClient()
        try:
            response = c.request('europe.pool.ntp.org', version=3)
            self.offset = response.offset - 940 / RATE
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
            signal_handler(None, None)
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

        x = np.linspace(start=c, stop=1 + c, num=RATE) % (np.pi * 4)
        # x = np.linspace(0, 1, num=RATE * MEASUREMENT_TIMEFRAME, endpoint=False)
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
            a, b, c = self.fit_sine(a, b, c, residuals, x)
            lastMeasurmentTime = time.time()
            log.store(b, analyze_start, lastMeasurmentTime - analyze_start)
            nrMeasurments += 1

    def fit_sine(self, a, b, c, residuals, x):
        change = SANITY_MAX_FREQUENCYCHANGE + 1
        while change > SANITY_MAX_FREQUENCYCHANGE:
            y = self.buffer.get(int(RATE * MEASUREMENT_TIMEFRAME))
            plsq = leastsq(residuals, np.array([a, b, c]), args=(x, y))
            change = abs(plsq[0][1] - b)
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
