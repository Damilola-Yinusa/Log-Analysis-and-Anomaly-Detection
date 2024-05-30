import pandas as pd
import numpy as np
import re
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import socket
import select
import threading

# Configure seaborn for better visualization
sns.set(style="whitegrid")

# Step 1: Simulate Log Data (for testing)
def generate_synthetic_logs(num_logs=1000):
    import random
    import datetime
    
    logs = []
    status_codes = [200, 301, 404, 500]
    for _ in range(num_logs):
        ip = ".".join(map(str, (random.randint(0, 255) for _ in range(4))))
        status = random.choice(status_codes)
        date = datetime.datetime.now() - datetime.timedelta(seconds=random.randint(0, 100000))
        logs.append(f"{ip} - - [{date.strftime('%d/%b/%Y:%H:%M:%S %z')}] \"GET /index.html HTTP/1.1\" {status} -")
    return logs

# Step 2: Log Parsing
def parse_logs(logs):
    log_entries = []
    log_pattern = re.compile(r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<datetime>.*?)\] "(?P<request>.*?)" (?P<status>\d+) -')
    
    for log in logs:
        match = log_pattern.match(log)
        if match:
            log_entries.append(match.groupdict())
    
    df = pd.DataFrame(log_entries)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%b/%Y:%H:%M:%S %z')
    df['status'] = df['status'].astype(int)
    return df

# Step 3: Feature Extraction
def extract_features(df):
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    return df[['ip', 'hour', 'day_of_week', 'status']]

# Step 4: Anomaly Detection
def detect_anomalies(df):
    features = df[['hour', 'day_of_week', 'status']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    model = IsolationForest(contamination=0.01)
    df['anomaly'] = model.fit_predict(features_scaled)
    return df[df['anomaly'] == -1]

# Real-time log monitoring
def real_time_log_monitoring():
    # Real-time log monitoring code (use socket, threading)
    server_address = ('localhost', 514)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(server_address)
    
    def handle_logs():
        while True:
            data, address = sock.recvfrom(4096)
            log = data.decode('utf-8')
            parsed_log = parse_logs([log])
            features = extract_features(parsed_log)
            anomalies = detect_anomalies(features)
            if not anomalies.empty:
                print(f"Anomaly detected: {anomalies}")
                # Send alert to admin (email, SMS, etc.)
    
    thread = threading.Thread(target=handle_logs)
    thread.daemon = True
    thread.start()

# Visualization
def visualize_data(df):
    plt.figure(figsize=(14, 7))
    sns.histplot(df['hour'], bins=24, kde=True)
    plt.title('Distribution of Log Entries by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(14, 7))
    sns.countplot(x='day_of_week', data=df)
    plt.title('Distribution of Log Entries by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Frequency')
    plt.show()

def main():
    # Uncomment to use synthetic log generation for testing
    # logs = generate_synthetic_logs()
    
    # Uncomment to load logs from a CSV file
    # df = pd.read_csv('logs.csv')
    
    # Uncomment to parse real-time logs
    # real_time_log_monitoring()
    
    # For testing purposes with synthetic logs
    logs = generate_synthetic_logs()
    parsed_logs = parse_logs(logs)
    features = extract_features(parsed_logs)
    anomalies = detect_anomalies(features)
    
    # Visualize the data
    visualize_data(parsed_logs)
    
    print("Detected Anomalies:")
    print(anomalies)

if __name__ == "__main__":
    main()
