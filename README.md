# hmm-anomaly

### Client Commands

1. Upload training data

```
# CPUUtilization
./target/debug/client upload-training cpu_utilization ../dataset/training/CPUUtilization.csv

# CommitLatency
./target/debug/client upload-training commit_latency ../dataset/training/CommitLatency.csv

# DiskQueueDepth
./target/debug/client upload-training disk_queue_depth ../dataset/training/DiskQueueDepth.csv
```

2. Train Model

```
# CPUUtilization
./target/debug/client train-model cpu_model 3 cpu_utilization CPUUtilization

# CommitLatency
./target/debug/client train-model commit_latency 3 commit_latency CommitLatency

# DiskQueueDepth
./target/debug/client train-model disk_queue_model 3 disk_queue_depth DiskQueueDepth
```

3. Find Anomalies

```
# CPUUtilization
./target/debug/client find-anomalies cpu_model ../dataset/testing/CPUUtilization.csv 8.0

# CommitLatency
./target/debug/client find-anomalies commit_latency ../dataset/testing/CommitLatency.csv 100.00

# DiskQueueDepth
./target/debug/client find-anomalies disk_queue_model ../dataset/testing/DiskQueueDepth.csv 1.0
```
