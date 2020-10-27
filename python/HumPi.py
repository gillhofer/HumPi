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
    db = client.gridfrequency.measurment
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
        self.time = np.zeros(maxSize, dtype=np.float64)
        self.index = 0
        self.lock = threading.Lock()
        self.last_timestamp = None
        self.offset = None
        self.sync_with_ntp()

    def extend(self, stream):
        [length, string] = stream
        if length > 0:
            x_index = np.arange(self.index, self.index + length) % self.data.size
            curr_time = time.time() + self.offset
            t, step = np.linspace(start=curr_time - length * RATE, stop=curr_time, num=length, retstep=True)
            data = np.frombuffer(string, dtype="f")[::CHANNELS]
            with self.lock:
                self.data[x_index] = data
                self.time[x_index] = t
                self.index = x_index[-1] + 1

    def get(self, length, with_time_stamp=False):
        with self.lock:
            index = self.index
        idx = np.arange(index - length, index) % self.data.size
        if not with_time_stamp:
            return self.data[idx]
        else:
            return self.data[idx], self.time[idx][-1]

    def sync_with_ntp(self):
        c = ntplib.NTPClient()
        try:
            response = c.request('europe.pool.ntp.org', version=3)
            with self.lock:
                self.offset = response.offset
            print_to_console("The clock is " + str(self.offset) + " seconds wrong. Changing timestamps", 5)
        except Exception as e:
            print_to_console(str(e), 20)


class Log:
    def __init__(self, sync_timestamp_fn):
        self.sync_with_ntp = sync_timestamp_fn
        self.data = []
        self.last_stored_date = datetime.now()
        self.offset = None
        self.last_sync = time.time()

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

    def store(self, frequency, timestamp, last_period_start):
        measurement_time_ = datetime.utcfromtimestamp(timestamp)
        last_period_start = datetime.utcfromtimestamp(last_period_start)
        if len(self.data) >= DOCUMENT_WRITE_INTERVAL:
            if args.serverurl:
                self.store_to_db(self.data)
            if args.store:
                self.save_to_disk()
            self.data = []

        self.data.append([measurement_time_, frequency, last_period_start])
        print_to_console(f"{repr(measurement_time_)}, {repr(last_period_start)}, {str(self.data[-1][1])}", 0)
        if time.time() - self.last_sync > NTP_TIMESYNC_INTERVAL:
            self.sync_with_ntp()
            self.last_sync = time.time()

    def store_to_db(self, data: list):
        updates = [UpdateOne({"rate": RATE, "n_samples": {"$lt": MAX_DB_DOCUMENT_LENGTH},
                              "measurement_timeframe": MEASUREMENT_TIMEFRAME},
                             {"$push": {"data": {"t": d[0], "f": d[1], "period_start": d[2]}},
                              "$min": {"first": d[0], "min_freq": d[1]},
                              "$max": {"last": d[0], "max_freq": d[1]},
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



class CaptureHum(threading.Thread):
    def __init__(self, threadID, name, buffer, stopSignal):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.buffer = buffer
        self.stopSignal = stopSignal
        self.start_time = time.time()

    def run(self):
        try:
            recorder = alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NONBLOCK,
                                     device=AUDIO_DEVICE_STRING, channels=CHANNELS, format=AUDIO_FORMAT,
                                     rate=RATE, periodsize=FRAMESIZE)
            self.recorder = recorder
        except:
            signal_handler(None, None)
        print(self.name, "* started recording")
        try:
            while (not self.stopSignal.is_set()):
                self.buffer.extend(recorder.read())
                time.sleep(0.001)
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
        self.start_time = time.time()

    def run(self):

        print(self.name, "* Started measurements")
        a = INITIAL_SIGNAL_AMPLITUDE
        b = 50 * 2 * np.pi
        c = 0
        n_measurments = 0
        while (not self.stopSignal.is_set()):
            time.sleep(time.time() % 0.5)
            a, b, c, ts, period_start = self.fit_sine(a, b, c)
            log.store(b / (np.pi * 2), ts, period_start)
            n_measurments += 1

    def fit_sine(self, a, b, c):
        def residuals(p, data, response):
            return response - p[0] * np.sin(p[1] * data + p[2])

        now = time.time()-BUFFERMAXSIZE
        change = SANITY_MAX_FREQUENCYCHANGE + 1
        while change > SANITY_MAX_FREQUENCYCHANGE:
            y, last_data_timestamp = self.buffer.get(int(RATE * MEASUREMENT_TIMEFRAME), with_time_stamp=True)
            x = np.linspace(start=last_data_timestamp - now - MEASUREMENT_TIMEFRAME,
                            stop=last_data_timestamp - now, num=len(y))
            fitted_result = leastsq(residuals, np.array([a, b, c]), args=(x, y))
            change = abs(fitted_result[0][1] - b)
        a, b, c = fitted_result[0]
        if a < 0:
            a = -a
            c -= np.pi / b
        x_desired = int(last_data_timestamp * 2) / 2 - now
        k = int((b * x_desired + c) / (2 * np.pi))
        x_raising = (2 * k * np.pi - c) / b
        last_period_start = x_raising + now
        return a, b, c, last_data_timestamp, last_period_start


def signal_handler(signal, frame):
    print(' --> Exiting HumPi')
    stopSignal.set()
    time.sleep(0.5)
    log.save_to_disk()
    time.sleep(0.5)
    sys.exit(0)


def print_to_console(message, severity):
    print(message)


databuffer = RingBuffer(RATE * BUFFERMAXSIZE)
log = Log(sync_timestamp_fn=databuffer.sync_with_ntp)
stopSignal = threading.Event()
signal.signal(signal.SIGINT, signal_handler)

capture = CaptureHum(1, "Capture", databuffer, stopSignal)
capture.start()
time.sleep(MEASUREMENT_TIMEFRAME + 0.5)
analyze = AnalyzeHum(2, "Analyze", databuffer, log, stopSignal)
analyze.start()
signal.pause()