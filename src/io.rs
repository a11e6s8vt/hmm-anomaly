use anyhow::Result;
use clap::{command, CommandFactory, Parser};
use csv::Reader;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[derive(Debug, Deserialize)]
pub struct CliInputs {
    #[arg(short, long)]
    training_data: Option<PathBuf>,

    #[arg(short, long)]
    input_data: Option<PathBuf>,
}

impl CliInputs {
    pub fn read_cli() -> Result<(PathBuf, PathBuf)> {
        let cli_inputs = CliInputs::parse();
        let training_data_file = cli_inputs.training_data.as_deref().unwrap_or_else(|| {
            let mut cmd = CliInputs::command();
            cmd.error(
                clap::error::ErrorKind::MissingRequiredArgument,
                "--training-data is required!",
            )
            .exit()
        });

        let input_data_file = cli_inputs.input_data.as_deref().unwrap_or_else(|| {
            let mut cmd = CliInputs::command();
            cmd.error(
                clap::error::ErrorKind::MissingRequiredArgument,
                "--input-data is required!",
            )
            .exit()
        });
        Ok((
            training_data_file.to_path_buf(),
            input_data_file.to_path_buf(),
        ))
    }
}

pub trait CsvReader<T> {
    fn read_csv(&mut self, path: PathBuf) -> Result<()>;
}

// cpu utilization is a time series data with a single f64 field
#[derive(Debug, Deserialize)]
pub struct CpuUtilizationEntry {
    pub time: String,
    pub utilization: f64,
}

#[derive(Default)]
pub struct TrainingData<T> {
    pub records: Vec<T>,
}

impl Default for TrainingData<CpuUtilizationEntry> {
    fn default() -> Self {
        Self {
            records: Vec::new(),
        }
    }
}

impl CsvReader<CpuUtilizationEntry> for TrainingData<CpuUtilizationEntry> {
    fn read_csv(&mut self, path: PathBuf) -> Result<()> {
        let mut csv_reader = Reader::from_path(path)?;
        for result in csv_reader.deserialize() {
            let record = result?;
            self.records.push(record);
        }
        Ok(())
    }
}
