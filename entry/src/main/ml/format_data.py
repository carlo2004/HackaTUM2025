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
        return pd.DataFrame([])

    if not data_records:
        return pd.DataFrame([])

    current_date = date.today()
    date_string = current_date.strftime("%Y-%m-%d")

    # 1. Convert to DataFrame
    df = pd.DataFrame(data_records)
    df['timestamp'] = pd.to_datetime(date_string + ' ' +  df['time'])
    df = df.set_index('timestamp')

    print(df)
    return df

def calculate_kurtosis(data):
    """
    Calculates Fisher Kurtosis manually.
    Formula: (Mean((x - mean)^4) / StdDev^4) - 3
    """
    n = len(data)
    if n < 4: return 0.0 # Not enough data to be significant
    
    # 1. Mean
    mean = np.mean(data)
    
    # 2. Differences
    diffs = data - mean
    
    # 3. Standard Deviation (Sigma)
    # Note: Pandas .std() uses n-1 (sample), Numpy .std() uses n (population). 
    # For signal processing, population (n) is usually fine and consistent.
    sigma = np.std(data)
    
    if sigma == 0: return 0.0 # Flatline = No shape
    
    # 4. Fourth Central Moment (Average of diffs to the 4th power)
    # This measures "tail heaviness" or "peakiness"
    moment4 = np.mean(diffs ** 4)
    
    # 5. Calculate Kurtosis
    kurt = (moment4 / (sigma ** 4)) - 3.0
    
    return kurt

def generate_feature_space(df, label_code):
    grouped = df.groupby(pd.Grouper(freq='1.5S'))

    features = []

    for time_bucket, group in grouped:
        if group.empty: continue

        z_data = group[group['type'] == 'Real_Acc_Z']['val']
        x_data = group[group['type'] == 'Real_Acc_X']['val']
        y_data = group[group['type'] == 'Real_Acc_Y']['val']
        g_data = group[group['type'] == 'Real_Gyr_X']['val'] 
        g2_data = group[group['type'] == 'Real_Gyr_Y']['val'] 
        g3_data = group[group['type'] == 'Real_Gyr_Z']['val']
        hr_data = group[group['type'] == 'Real_HR']['val']


        if len(z_data) < 5: 
            continue

        mean_x = x_data.mean() if not x_data.empty else 0
        mean_y = y_data.mean() if not y_data.empty else 0
        tilt_error = np.sqrt(mean_x**2 + mean_y**2)

        wobble_score_x = g_data.std() if not g_data.empty else 0
        wobble_score_y = g2_data.std() if not g2_data.empty else 0
        wobble_score_z = g2_data.std() if not g2_data.empty else 0

        std_dev_z = z_data.std()

        recoil_proxy_z = z_data.max() 
        recoil_proxy_x = 0

        std_x = x_data.std()
        std_y = y_data.std()
        std_z = z_data.std()
        recoil_proxy_y = x_data.max()
        recoil_proxy = 0

        rescuer_hr = hr_data.mean() if not hr_data.empty else 0

        features.append([std_dev_z, recoil_proxy_z, recoil_proxy_x, recoil_proxy_y, recoil_proxy, rescuer_hr, tilt_error, wobble_score_x, wobble_score_y, wobble_score_z, label_code])

    return features

def generate_features_from_file(filename, label_code):
    df = parse_raw_logs(filename)
    return generate_feature_space(df, label_code)



