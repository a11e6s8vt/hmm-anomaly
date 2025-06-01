use anyhow::Result;
use hmm_anomaly::{HmmServer, ServerCommand};
use std::sync::Arc;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

#[tokio::main]
async fn main() -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    let server = Arc::new(HmmServer::default());

    println!("Server listening on 127.0.0.1:8080");

    loop {
        let (socket, _) = listener.accept().await?;
        let server = server.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_connection(socket, server).await {
                eprintln!("Error handling connection: {}", e);
            }
        });
    }
}

async fn handle_connection(mut stream: TcpStream, server: Arc<HmmServer>) -> anyhow::Result<()> {
    // Read command length
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf);

    // Read command
    let mut cmd_buf = vec![0u8; len as usize];
    stream.read_exact(&mut cmd_buf).await?;
    let command: ServerCommand = serde_json::from_slice(&cmd_buf)?;

    // Process command
    let response = server.handle_command(command).await?;

    // Send response
    let resp_json = serde_json::to_string(&response)?;
    let len = resp_json.len() as u32;
    stream.write_all(&len.to_be_bytes()).await?;
    stream.write_all(resp_json.as_bytes()).await?;

    Ok(())
}
