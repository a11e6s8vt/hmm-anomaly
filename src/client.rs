use clap::{Parser, Subcommand};
use csv::Reader;
use hmm_anomaly::{is_csv_by_content, read_input_data, ServerCommand, ServerResponse};
use serde_json;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

#[derive(Parser)]
#[command(name = "hmm-client")]
#[command(about = "HMM Anomaly Detection Client", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(long, default_value = "127.0.0.1:8080")]
    server: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Upload training data from CSV file
    UploadTraining {
        /// Name for this training dataset
        name: String,

        /// Path to CSV file
        file: PathBuf,
    },
    /*
    /// Train an HMM model
    TrainModel {
        /// Name of the model
        name: String,

        /// Number of hidden states
        states: usize,

        /// Name of training data to use
        training_data: String,
    },

    /// Find anomalies in data
    FindAnomalies {
        /// Model name to use
        model: String,

        /// Path to CSV file with data to analyze
        file: PathBuf,

        /// Anomaly threshold
        threshold: f64,
    },

    /// Correlate anomalies across multiple models
    CorrelateAnomalies {
        /// Comma-separated list of model names
        models: String,

        /// Path to CSV file with data to analyze
        file: PathBuf,
    },

    /// Get information about a model
    ModelInfo {
        /// Model name
        name: String,
    },
    */
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Connect to server
    let mut stream = TcpStream::connect(&cli.server).await?;

    // Process command
    match cli.command {
        Commands::UploadTraining { name, file } => {
            // Read CSV file

            if !file.is_dir() && is_csv_by_content(&file)? {
                let data = read_input_data(&file).await?;
                println!("{:?}", data);
                let cmd = ServerCommand::UploadTraining { name, data };
                send_command(&mut stream, cmd).await?;
            } else {
                eprintln!("Only CSV files are supported");
            }
        } /*
          Commands::TrainModel {
              name,
              states,
              training_data,
          } => {
              let cmd = ServerCommand::TrainModel {
                  name,
                  states,
                  training_data,
              };
              send_command(&mut stream, cmd).await?;
          }
          Commands::FindAnomalies {
              model,
              file,
              threshold,
          } => {
              let data = read_csv_file(&file)?;
              let cmd = ServerCommand::FindAnomalies {
                  model_name: model,
                  data,
                  threshold,
              };
              send_command(&mut stream, cmd).await?;
          }
          Commands::CorrelateAnomalies { models, file } => {
              let data = read_csv_file(&file)?;
              let model_names = models.split(',').map(|s| s.trim().to_string()).collect();
              let cmd = ServerCommand::CorrelateAnomalies { model_names, data };
              send_command(&mut stream, cmd).await?;
          }
          Commands::ModelInfo { name } => {
              let cmd = ServerCommand::GetModelInfo { name };
              send_command(&mut stream, cmd).await?;
          }*/
    }

    Ok(())
}

async fn send_command(stream: &mut TcpStream, command: ServerCommand) -> anyhow::Result<()> {
    // Serialize command
    let cmd_json = serde_json::to_string(&command)?;

    // Send length prefix
    let len = cmd_json.len() as u32;
    stream.write_all(&len.to_be_bytes()).await?;

    // Send command
    stream.write_all(cmd_json.as_bytes()).await?;

    // Read response length
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf);

    // Read response
    let mut resp_buf = vec![0u8; len as usize];
    stream.read_exact(&mut resp_buf).await?;
    let response: ServerResponse = serde_json::from_slice(&resp_buf)?;

    // Handle response
    match response {
        ServerResponse::Success(msg) => println!("Success: {}", msg),
        ServerResponse::Error(err) => eprintln!("Error: {}", err),
        ServerResponse::AnomalyResults(anomalies) => {
            println!("Anomalies found at indices:");
            for (idx, score) in anomalies {
                println!("  {}: {:.4}", idx, score);
            }
        }
        ServerResponse::CorrelationResults(results) => {
            println!("Correlated anomalies:");
            for (model, anomalies) in results {
                println!("  {}: {:?}", model, anomalies);
            }
        }
        ServerResponse::ModelInfo(info) => {
            println!("Model info:\n{}", info);
        }
    }

    Ok(())
}
