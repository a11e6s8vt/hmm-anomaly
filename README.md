# hmm-anomaly

### Client Commands

1. Upload training data

```
./target/debug/client upload-training cpu_utilization ../dataset/training/CPUUtilization.csv
```

2. Train Model

```
./target/debug/client train-model cpu_model 3 cpu_utilization CPUUtilization
```

3. Find Anomalies

```
./target/debug/client find-anomalies cpu_model ../dataset/testing/CPUUtilization.csv 7.0
```
