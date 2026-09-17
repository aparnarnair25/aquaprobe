import csv
import random
import math
from datetime import datetime, timedelta

OUTPUT_FILE = "data/raw/aquaprobe_raw.csv"
NUM_ROWS = 5000

random.seed(42)

headers = [
    "lat", "lon", "date",
    "sst", "sss", "ssh", "wind_u", "wind_v",
    "t_0", "t_10", "t_30", "t_50", "t_100", "t_200",
    "split"
]

start_date = datetime(2020, 1, 1)

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)

    for i in range(NUM_ROWS):

        lat = random.uniform(5, 20)
        lon = random.uniform(65, 90)

        date = start_date + timedelta(days=random.randint(0, 2000))

        sst = random.uniform(24, 31)
        sss = random.uniform(32, 37)
        ssh = random.uniform(-1.0, 1.0)
        wind_u = random.uniform(-10, 10)
        wind_v = random.uniform(-10, 10)

        # Generate depth temperatures
        t0 = sst
        t10 = t0 - random.uniform(0.2, 1.0)
        t30 = t10 - random.uniform(0.2, 1.0)
        t50 = t30 - random.uniform(0.1, 0.8)
        t100 = t50 - random.uniform(0.1, 0.7)
        t200 = t100 - random.uniform(0.1, 0.6)

        split_value = random.random()

        if split_value < 0.7:
            split = "train"
        elif split_value < 0.85:
            split = "validation"
        else:
            split = "test"

        writer.writerow([
            round(lat, 4),
            round(lon, 4),
            date.strftime("%Y-%m-%d"),
            round(sst, 3),
            round(sss, 3),
            round(ssh, 3),
            round(wind_u, 3),
            round(wind_v, 3),
            round(t0, 3),
            round(t10, 3),
            round(t30, 3),
            round(t50, 3),
            round(t100, 3),
            round(t200, 3),
            split
        ])

print(f"Created {NUM_ROWS} rows in {OUTPUT_FILE}")