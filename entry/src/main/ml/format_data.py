import pandas as pd
import re
from datetime import date
import numpy as np


def parse_raw_logs(filename):
    data_records = []

    pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3}).*?(Real_Acc_[XYZ]|Real_Gyr_[XYZ]|Real_HR):\s*([-\d\.]+)")

    try:
        with open(filename, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    time_str = match.group(1)
                    sensor_type = match.group(2)
                    value = float(match.group(3))
                    data_records.append({'time': time_str, 'type': sensor_type, 'val': value})
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return []

    if not data_records:
        return []

    current_date = date.today()
    date_string = current_date.strftime("%Y-%m-%d")

    # 1. Convert to DataFrame
    df = pd.DataFrame(data_records)
    df['timestamp'] = pd.to_datetime(date_string + ' ' +  df['time'])
    df = df.set_index('timestamp')

    return df

def generate_feature_space(df, label_code):
    grouped = df.groupby(pd.Grouper(freq='1S'))

    features = []

    for time_bucket, group in grouped:
        if group.empty: continue

        z_data = group[group['type'] == 'Real_Acc_Z']['val']
        x_data = group[group['type'] == 'Real_Acc_X']['val']
        y_data = group[group['type'] == 'Real_Acc_Y']['val']
        g_data = group[group['type'] == 'Real_Gyr_X']['val'] 
        hr_data = group[group['type'] == 'Real_HR']['val']


        if len(z_data) < 5: 
            continue

        mean_x = x_data.mean() if not x_data.empty else 0
        mean_y = y_data.mean() if not y_data.empty else 0
        tilt_error = np.sqrt(mean_x**2 + mean_y**2)

        wobble_score = g_data.std() if not g_data.empty else 0

        std_dev_z = z_data.std()

        recoil_proxy = z_data.max()

        rescuer_hr = hr_data.mean() if not hr_data.empty else 0

        features.append([std_dev_z, recoil_proxy, rescuer_hr, tilt_error, wobble_score, label_code])

    return features

def generate_features_from_file(filename, label_code):
    df = parse_raw_logs(filename)
    return generate_feature_space(df, label_code)

print(generate_features_from_file("test2.txt", 2))


