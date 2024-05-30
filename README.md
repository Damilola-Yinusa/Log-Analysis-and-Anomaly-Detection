# Log Analysis and Anomaly Detection

## Project Description

This project implements a robust log analysis and anomaly detection system designed to detect potential security threats in real-time. The system can process log data from various sources, including CSV files and real-time network connections, and it uses machine learning techniques to identify unusual patterns that may indicate security breaches.

Key features include:
- Log parsing with regular expressions
- Feature extraction for anomaly detection
- Anomaly detection using Isolation Forest
- Real-time log monitoring
- Data visualization for analysis

## Key Concepts

- Log parsing
- Anomaly detection
- Regular expressions
- Machine learning

## Libraries Used

- pandas
- numpy
- scikit-learn
- re
- matplotlib
- seaborn
- socket
- threading

## Setup Instructions

1. **Clone the Repository**
    ```bash
    git clone https://github.com/Damilola-Yinusa/log-analysis-anomaly-detection.git
    cd log-analysis-anomaly-detection
    ```

2. **Install Dependencies**
    Make sure you have Python 3.6 or later installed. Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

## Usage

### 1. Simulated Logs

To test the system with synthetic log data, uncomment the following lines in the `main` function:
```python
# logs = generate_synthetic_logs()




2. CSV Logs
To load log data from a CSV file, uncomment the following lines in the main function and provide the path to your CSV file:

```python
# df = pd.read_csv('logs.csv')


3. Real-time Log Monitoring
To enable real-time log monitoring, uncomment the following lines in the main function:

```python

# real_time_log_monitoring()


4. Visualization
The script includes functions for visualizing the distribution of log entries by hour and day of the week. These visualizations help in understanding the data and identifying patterns.

5. Running the Script
Execute the script:

```bash
python log_analysis.py


Example Output
After running the script, you should see visualizations and a printed list of detected anomalies. Real-time monitoring will print anomalies as they are detected.

Customization
Anomaly Detection Sensitivity: Adjust the contamination parameter in the Isolation Forest model to control the sensitivity of anomaly detection.
Alerting Mechanism: Customize the real-time alerting mechanism to send notifications via email, SMS, etc.
Contribution
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

License
This project is licensed under the MIT License - see the LICENSE file for details.
