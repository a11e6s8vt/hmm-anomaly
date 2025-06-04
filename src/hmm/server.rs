use crate::plot_anomalies;
use crate::{AnalyticsEngine, Hmm};
use anyhow::anyhow;
use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use ndarray::Array2;
use polars::df;
use polars::{prelude::*, series::IsSorted};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::Bayesian;

#[derive(Debug, Serialize, Deserialize)]
pub enum ServerCommand {
    UploadTraining {
        name: String,
        data: DataFrame,
    },
    ListTraining,
    DescTraining {
        name: String,
    },
    TrainModel {
        model_name: String,
        n_states: usize,
        training_data: String,
        field: String,
    },
    FindAnomalies {
        model_name: String,
        data: DataFrame,
        threshold: f64,
    },
    CorrelateAnomalies {
        model_names: Vec<String>,
        data: Vec<DataFrame>,
    },
    // GetModelInfo {
    //     name: String,
    // },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ServerResponse {
    Success(String),
    Error(String),
    TrainingList(Vec<String>),
    DescTraining(Vec<(String, String)>),
    AnomalyResults(Vec<(String, f64, f64)>), // (index, anomaly_score)
    CorrelationResults(Vec<(String, Vec<usize>)>), // (model_name, anomaly_indices)
    ModelInfo(String),
}

pub struct HmmServer {
    models: Arc<Mutex<HashMap<String, Hmm>>>,
    training_data: Arc<Mutex<HashMap<String, DataFrame>>>,
}

impl HmmServer {
    fn new() -> Self {
        HmmServer {
            models: Arc::new(Mutex::new(HashMap::new())),
            training_data: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn handle_command(&self, command: ServerCommand) -> anyhow::Result<ServerResponse> {
        match command {
            ServerCommand::UploadTraining { name, data } => {
                let mut training_data = self.training_data.lock().await;
                training_data.insert(name.clone(), data);
                Ok(ServerResponse::Success(format!(
                    "Training data '{}' uploaded",
                    name
                )))
            }
            ServerCommand::ListTraining => {
                let training_data = self.training_data.lock().await;
                let l = training_data
                    .keys()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>();
                Ok(ServerResponse::TrainingList(l))
            }
            ServerCommand::DescTraining { name } => {
                let training_data = self.training_data.lock().await;
                if let Some(data) = training_data.get(&name) {
                    let s = data
                        .schema()
                        .iter()
                        .map(|(n, t)| (n.to_string(), t.to_string()))
                        .collect::<Vec<(String, String)>>();
                    Ok(ServerResponse::DescTraining(s))
                } else {
                    Ok(ServerResponse::Error(format!(
                        "Training data, {} not found!",
                        name
                    )))
                }
            }
            ServerCommand::TrainModel {
                model_name,
                n_states,
                training_data,
                field,
            } => {
                let training_data_lock = self.training_data.lock().await;
                if let Some(data) = training_data_lock.get(&training_data) {
                    // Convert to ndarray format
                    // TODO: Change unwrap()
                    let observations = data.column(&field).unwrap().f64().unwrap();
                    let mean = observations
                        .mean()
                        .ok_or(anyhow!("Calculating Sample mean failed!"))?;
                    // ddof=0 provides a maximum likelihood estimate
                    let ddof = 0;
                    let std = observations
                        .std(ddof)
                        .ok_or(anyhow!("Calculating std failed"))?;
                    let variance = std * std;
                    let shape = (mean * mean) / variance;
                    let scale = variance / mean;
                    let observations: Vec<f64> = observations.into_no_null_iter().collect();
                    let observations =
                        Array2::from_shape_vec((observations.len(), 1), observations)?;

                    // TODO: compute it from number of fields in training_data
                    let n_features: usize = 1;

                    // TODO: Make it a command line input parameter
                    // No of iterations in the gibbs_sampling
                    const NUM_ITER: usize = 1000;
                    const BURN_IN: usize = 500;

                    // Initialise HMM
                    let mut hmm = Hmm::new(n_features, n_states, mean, variance, shape, scale);

                    // Train HMM - Gibbs Sampling
                    match hmm.learn_gibbs_sampling(&observations, NUM_ITER, BURN_IN) {
                        Ok(_) => {}
                        Err(e) => eprintln!("{:?}", e),
                    };

                    // Store model
                    let mut models = self.models.lock().await;
                    models.insert(model_name.clone(), hmm);

                    Ok(ServerResponse::Success(format!(
                        "Model '{}' trained with {} states",
                        model_name, n_states
                    )))
                } else {
                    Ok(ServerResponse::Error(format!(
                        "No training data found for '{}'",
                        model_name
                    )))
                }
            }
            ServerCommand::FindAnomalies {
                model_name,
                data,
                threshold,
            } => {
                let models = self.models.lock().await;
                if let Some(model) = models.get(&model_name) {
                    // Find anomalies
                    let scores = match model.anomaly_scores(&data, threshold) {
                        Ok(scores) => scores,
                        Err(e) => {
                            eprintln!("err: {:?}", e);
                            return Ok(ServerResponse::Error(format!(
                                "Anomaly scores calculations failed using model '{}'",
                                model_name
                            )));
                        }
                    };
                    plot_anomalies(&scores, threshold)?;
                    let anomalies: Vec<_> = scores
                        .iter()
                        .enumerate()
                        .filter(|(_, (_, _, score))| score > &threshold)
                        .map(|(_, (time, obs, score))| (time.to_string(), *obs, *score))
                        .collect();

                    Ok(ServerResponse::AnomalyResults(anomalies))
                } else {
                    Ok(ServerResponse::Error(format!(
                        "Anomaly scores calculations failed using model '{}'",
                        model_name
                    )))
                }
            }
            ServerCommand::CorrelateAnomalies { model_names, data } => {
                let mut results = Vec::new();

                align_events_windows(data).await?;
                Ok(ServerResponse::CorrelationResults(results))
            } // ServerCommand::GetModelInfo { name } => {
              //     let models = self.models.lock().await;
              //     if let Some(model) = models.get(&name) {
              //         ServerResponse::ModelInfo(format!("{:?}", model))
              //     } else {
              //         ServerResponse::Error(format!("Model '{}' not found", name))
              //     }
              // }
        }
    }
}

impl Default for HmmServer {
    fn default() -> Self {
        Self::new()
    }
}

async fn align_events_windows(data: Vec<DataFrame>) -> anyhow::Result<()> {
    let mut data_lazy = Vec::new();
    for (i, d) in data.iter().enumerate() {
        let mut d = d.clone();
        data_lazy.push(d.lazy());
    }

    let time_options = StrptimeOptions {
        format: Some("%Y/%m/%d %H:%M:%S".into()),
        ..Default::default()
    };

    // // 2. Sort by Time column
    // combined_df = combined_df.sort(
    //     ["Time"],
    //     SortMultipleOptions::new().with_order_descending(true),
    // );
    //
    let mut combined = data_lazy[0].clone();
    for lz in data_lazy.iter().skip(1) {
        combined = combined.join(
            lz.clone(),
            [col("Time")],
            [col("Time")],
            JoinArgs::new(JoinType::Inner),
        );
    }
    let combined = combined
        .sort(
            ["Time"],
            SortMultipleOptions {
                descending: vec![false],
                nulls_last: vec![false],
                ..Default::default()
            },
        )
        .collect()?;
    println!("{:?}", combined);
    // let aggregated = combined_df
    //     .lazy()
    //     .group_by_dynamic(col("Time"), [], window_options)
    //     .agg([
    //         // Mean calculations for all numeric columns
    //         col("CPUUtilization").mean().alias("CPUUtilization_mean"),
    //         col("CPUUtilization").std(0).alias("CPUUtilization_std"),
    //         col("DiskQueueDepth").mean().alias("DiskQueueDepth_mean"),
    //         col("DiskQueueDepth").std(0).alias("DiskQueueDepth_std"),
    //         col("CommitLatency").mean().alias("CommitLatency_mean"),
    //         col("CommitLatency").std(0).alias("CommitLatency_std"),
    //         // Add variance (std squared)
    //         (col("CPUUtilization").std(0) * col("CPUUtilization").std(0))
    //             .alias("CPUUtilization_var"),
    //         (col("DiskQueueDepth").std(0) * col("DiskQueueDepth").std(0))
    //             .alias("DiskQueueDepth_var"),
    //         (col("CommitLatency").std(0) * col("Value1").std(0)).alias("CommitLatency_var"),
    //     ])
    //     .collect()?;
    Ok(())
}
