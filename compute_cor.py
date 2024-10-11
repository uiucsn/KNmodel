import json
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def scrape_data():

    import requests

    f = open("api_call.txt", "w")

    # Correct JSON data without serializing it
    payload = {
    "intervalMs": 1 * 60 * 1000, # 1 minute intervals
    "maxDataPoints": 600000,
    "timeRange": {
        "from": "1710308738515",
        "to": "1728540788437"
    }
    }

    # Make the request
    x = requests.post(
        'https://online.igwn.org/grafana/api/public/dashboards/1a0efabe65384a7287abfcc1996e4c4d/panels/2/query',
        json=payload
    )

    # Check the response
    print(x.status_code)
    json.dump(x.json(), f)

#scrape_data()

f = open("api_call.txt", "r")
data = json.load(f)


# LIGO O4 start and end date
Leg_1_start = datetime.datetime(2024, 4, 10)
Leg_1_end = datetime.datetime(2024, 7, 10)

Leg_2_start = datetime.datetime(2024, 8, 25)
Leg_2_end = datetime.datetime(2024, 10, 10)


frames = H1_times = data["results"]["A"]["frames"]

times = {}
distances = {}

# If the instruments are observing
observing_bool = {}

for frame in frames:

    instrument = frame['schema']['name']

    t_unix = frame["data"]['values'][0]
    t_human = np.array([datetime.datetime.fromtimestamp(t/1000.0) for t in t_unix])

    d = np.array(frame["data"]['values'][1])

    times[instrument] = t_human
    distances[instrument] = d


# Only pick indices with dates after O4 start
for key in times:

    t = times[key]
    d = distances[key]

    idx = np.where(((np.array(t) > Leg_1_start) & (np.array(t) < Leg_1_end)) | ((np.array(t) > Leg_2_start) & (np.array(t) < Leg_2_end)))[0]

    # Only pick indices with dates after O4 start
    times[key] = t[idx]
    distances[key] = d[idx]

    observing_bool[key] = np.where(distances[key]!=None,1, 0)

data = []
columns = []
for key in times:
    columns.append(key)
    data.append(observing_bool[key].tolist())

data = np.array(data).T
df = pd.DataFrame(data, columns=columns)

#print(f"Start:{LVK_O4_start} End:{LVK_O4_end}")
print("H1 L1 Uptime correlation:", np.sum(observing_bool["H1"]*observing_bool["L1"])/len(observing_bool['H1']))
print("H1 V1 Uptime correlation:", np.sum(observing_bool["H1"]*observing_bool["V1"])/len(observing_bool['V1']))
print("V1 L1 Uptime correlation:", np.sum(observing_bool["V1"]*observing_bool["L1"])/len(observing_bool['L1']))

for key in times:
    
    up_bool = observing_bool[key]
    print(f"{key} Uptime", np.sum(up_bool)/len(up_bool))

for key in times:

    t = times[key]
    d = distances[key]
    b = observing_bool[key]

    plt.scatter(t, d, marker='.', label=key)

plt.legend()

plt.show()





