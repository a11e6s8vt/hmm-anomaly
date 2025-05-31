use anyhow::{anyhow, Result};
use chrono::DateTime;
use chrono::Utc;
use clap::{command, CommandFactory, Parser};
use csv::Reader;
use plotters::prelude::*;
use polars::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path as StdPath, PathBuf};
use walkdir::WalkDir;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[derive(Debug, Deserialize)]
pub struct CliInputs {
    #[arg(long)]
    training_data_path: Option<PathBuf>,

    #[arg(long)]
    test_data_path: Option<PathBuf>,
}

impl CliInputs {
    pub fn read_cli() -> Result<(PathBuf, PathBuf)> {
        let cli_inputs = CliInputs::parse();
        let training_data_path = cli_inputs.training_data_path.as_deref().unwrap_or_else(|| {
            let mut cmd = CliInputs::command();
            cmd.error(
                clap::error::ErrorKind::MissingRequiredArgument,
                "--training-data-path is required!",
            )
            .exit()
        });

        let test_data_path = cli_inputs.test_data_path.as_deref().unwrap_or_else(|| {
            let mut cmd = CliInputs::command();
            cmd.error(
                clap::error::ErrorKind::MissingRequiredArgument,
                "--test-data-path is required!",
            )
            .exit()
        });
        Ok((
            training_data_path.to_path_buf(),
            test_data_path.to_path_buf(),
        ))
    }
}

pub trait MetricReader<T> {
    fn read_csv(&mut self, path: PathBuf) -> Result<()>;
}

// cpu utilization is a time series data with a single f64 field
#[derive(Debug, Deserialize)]
pub struct MetricEntry {
    pub time: String,
    pub data: f64,
}

#[derive(Default)]
pub struct Metric<T> {
    pub name: String,
    pub records: Vec<T>,
}

impl Metric<MetricEntry> {
    fn new(name: String) -> Self {
        Self {
            name,
            records: Vec::new(),
        }
    }
}

impl MetricReader<MetricEntry> for Metric<MetricEntry> {
    fn read_csv(&mut self, path: PathBuf) -> Result<()> {
        let mut csv_reader = Reader::from_path(path)?;
        for result in csv_reader.deserialize() {
            let record = result?;
            self.records.push(record);
        }
        Ok(())
    }
}

pub fn is_csv_by_content<P: AsRef<StdPath>>(path: P) -> anyhow::Result<bool> {
    let file = File::open(path)?;
    let reader = BufReader::new(&file);

    // Check first few lines for comma-separated values
    for line in reader.lines().take(5).filter_map(|l| l.ok()) {
        if line.split(',').count() < 2 {
            return Ok(false); // Not CSV (fewer than 2 columns)
        }
    }
    Ok(true)
}

pub async fn read_input_data(path: &StdPath) -> anyhow::Result<DataFrame> {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path.to_path_buf()))?
        .finish()?;

    Ok(df)
}

pub fn plot_anomalies(scores: Vec<(String, f64, f64)>, score_threshold: f64) -> anyhow::Result<()> {
    let home = match std::env::var("HOME") {
        Ok(path) => path,
        Err(e) => return Err(anyhow!("Getting path information failed!")),
    };

    let exe_path = std::env::current_exe().expect("Failed to get executable path");
    let program_name = exe_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown");
    let output_path = std::path::Path::new(&home).join(program_name);
    if !output_path.exists() {
        std::fs::create_dir_all(&output_path)?;
    }
    let output_path = output_path.join("anomaly.png");

    let root = BitMapBackend::new(&output_path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let format = "%Y/%m/%d %H:%M:%S";

    let t = scores.first().unwrap().clone().0.to_string();
    let start_time: DateTime<Utc> =
        match DateTime::parse_from_str(&format!("{} +0000", t), &format!("{} %z", format)) {
            Ok(t) => t.with_timezone(&Utc),
            Err(e) => {
                eprint!("{:?}", e);
                return Err(anyhow!("Failed to parse time: {}", e));
            }
        };

    let end_time: DateTime<Utc> = DateTime::parse_from_str(
        &format!("{} +0000", scores.last().unwrap().clone().0),
        &format!("{} %z", format),
    )?
    .with_timezone(&Utc);

    let mut ctx = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("CPU Utilization with Anomalies", ("sans-serif", 40))
        .build_cartesian_2d(start_time..end_time, 0.0..100.0)
        .unwrap();

    ctx.configure_mesh()
        .x_labels(40)
        .x_label_formatter(&|dt| format!("{}", dt.format("%m-%d %H")))
        .x_label_style(
            ("sans-serif", 15)
                .into_font()
                .transform(FontTransform::Rotate270),
        )
        .y_desc("CPU Usage")
        .x_desc("Time")
        .draw()
        .unwrap();

    let mut graph_points: Vec<_> = Vec::new();
    for (t, val, score) in scores.iter() {
        let time: DateTime<Utc> =
            DateTime::parse_from_str(&format!("{} +0000", t), &format!("{} %z", format))?
                .with_timezone(&Utc);
        graph_points.push((time, *val));
    }
    ctx.draw_series(LineSeries::new(graph_points, &BLUE))
        .unwrap();

    let mut anomaly_points: Vec<(DateTime<Utc>, f64)> = Vec::new();

    for (t, val, score) in scores.iter() {
        if score >= &score_threshold {
            let time: DateTime<Utc> =
                DateTime::parse_from_str(&format!("{} +0000", t), &format!("{} %z", format))?
                    .with_timezone(&Utc);
            anomaly_points.push((time, *val));
        } else {
            continue;
        }
    }

    ctx.draw_series(
        anomaly_points
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 1.2, RED.filled())),
    )?;

    Ok(())
}
