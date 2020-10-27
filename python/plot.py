import argparse
from datetime import datetime
from datetime import timedelta
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
from pandas import json_normalize
from pymongo import MongoClient

parser = argparse.ArgumentParser()

parser.add_argument("--serverurl", help="Path of the mongodb server")
parser.add_argument("--serveruser", help="mongodb username")
parser.add_argument("--serverpassword", help="mongodb password")

args = parser.parse_args()
uri = "mongodb://%s:%s@%s" % (
    quote_plus(args.serveruser), quote_plus(args.serverpassword), quote_plus(args.serverurl))
client = MongoClient(uri)
db = client.gridfrequency.measuerment

stop = datetime.fromisoformat("2020-10-21T00:23:50.676+00:00").replace(tzinfo=None)
start = datetime.fromisoformat("2020-10-21T00:23:37.856+00:00").replace(tzinfo=None)

stop = datetime.utcnow()
start = datetime.utcnow() - timedelta(minutes=30)


def get_data(start, stop):
    cursor = db.aggregate(pipeline=[
        {"$match": {"first": {"$lt": stop}, "last": {"$gt": start}}},
        {"$match": {"data.t": {"$lt": stop}, "data.t": {"$gt": start}}},
        {"$unwind": "$data"},
        {"$group": {"_id": "$data.t", "t": {"$first": "$data.t"}, "f": {"$first": "$data.f"},
                    "period_start": {"$first": "$data.period_start"}}},
    ])
    data = json_normalize(cursor).drop(["_id"], axis=1)
    data = data.set_index("t").sort_index()
    return data


data = get_data(start=start, stop=stop)
data["f"].plot()
plt.ylim([49.8, 50.2])
plt.gca().axhline(y=50, color="grey", alpha=0.5)
plt.show()
pass
