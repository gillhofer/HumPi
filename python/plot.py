import argparse
from datetime import datetime
from datetime import timedelta
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient

parser = argparse.ArgumentParser()

parser.add_argument("--serverurl", help="Path of the mongodb server")
parser.add_argument("--serveruser", help="mongodb username")
parser.add_argument("--serverpassword", help="mongodb password")

args = parser.parse_args()
uri = "mongodb://%s:%s@%s" % (
    quote_plus(args.serveruser), quote_plus(args.serverpassword), quote_plus(args.serurl))
client = MongoClient(uri)
db = client.gridfrequency.raw

stop = datetime.utcnow()
start = datetime.utcnow() - timedelta(minutes=30)


def get_data(start, stop):
    cursor = db.find({"first": {"$lt": stop}, "last": {"$gt": start}}, {"data": 1})
    data = pd.concat([pd.DataFrame(d["data"]) for d in cursor])
    data = data.set_index("ts").sort_index()[start:stop]
    return data


data = get_data(start=start, stop=stop)
data["freq"].plot()
plt.ylim([49.8, 50.2])
plt.gca().axhline(y=50, color="grey", alpha=0.5)
plt.show()
pass
