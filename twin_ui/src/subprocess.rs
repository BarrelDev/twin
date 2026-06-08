use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use serde::Deserialize;

// --- JSON response types ---

#[derive(Debug, Deserialize)]
pub struct QueryResult {
    pub text: String,
    pub source: String,
    pub score: f32,
    pub heading_path: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct RagResponse {
    pub answer: String,
    pub sources: Vec<RagSource>,
    pub usage: RagUsage,
}

#[derive(Debug, Deserialize)]
pub struct RagSource {
    pub file: String,
    pub heading: String,
}

#[derive(Debug, Deserialize)]
pub struct RagUsage {
    pub tokens: u64,
    pub cost: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct IngestResult {
    pub doc_id: String,
    pub chunks_added: u32,
    pub skipped: bool,
}

/// One NDJSON line emitted by `twin agent --json`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    ToolCall { tool: String, input: String },
    ToolResult { tool: String, output: String },
    Answer { text: String },
    Log { message: String },
}

#[derive(Debug, Deserialize)]
pub struct UsageRecord {
    pub date: String,
    pub provider: String,
    pub calls: u32,
    pub tokens: u64,
    pub cost: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct ConfigInfo {
    pub provider: String,
    pub model: String,
    pub keys_present: Vec<String>,
}

// --- Command enum ---

pub enum CMD {
    Ingest(String),
    Query(String),
    RAG(String, u8),
    /// Agent streams NDJSON events over the sender; JoinHandle signals completion.
    Agent(String, u8, mpsc::Sender<AgentEvent>),
    Usage,
    Watch(String),
    ConfigList,
    ConfigListModels,
}

pub type CmdOut = anyhow::Result<String>;

/// Spawn a background thread for a CLI command. Non-blocking for the UI thread.
pub fn call_cli(cmd: CMD) -> std::thread::JoinHandle<CmdOut> {
    std::thread::spawn(move || match cmd {
        CMD::Ingest(path) => ingest_cmd(&path),
        CMD::Query(query) => query_cmd(&query),
        CMD::RAG(query, k) => rag_cmd(&query, k),
        CMD::Agent(task, k, tx) => agent_cmd(&task, k, tx),
        CMD::Usage => usage_cmd(),
        CMD::Watch(path) => watch_cmd(&path),
        CMD::ConfigList => config_list_cmd(),
        CMD::ConfigListModels => config_list_models_cmd(),
    })
}

fn ingest_cmd(s: &str) -> CmdOut {
    let raw = Command::new("twin").args(["ingest", s, "--json"]).output()?;
    anyhow::ensure!(raw.status.success(), "{}", String::from_utf8_lossy(&raw.stderr));
    Ok(String::from_utf8(raw.stdout)?)
}

fn query_cmd(s: &str) -> CmdOut {
    let raw = Command::new("twin").args(["query", s, "--json"]).output()?;
    anyhow::ensure!(raw.status.success(), "{}", String::from_utf8_lossy(&raw.stderr));
    Ok(String::from_utf8(raw.stdout)?)
}

fn rag_cmd(s: &str, k: u8) -> CmdOut {
    let raw = Command::new("twin")
        .args(["rag", s, "-k", &k.to_string(), "--json"])
        .output()?;
    anyhow::ensure!(raw.status.success(), "{}", String::from_utf8_lossy(&raw.stderr));
    Ok(String::from_utf8(raw.stdout)?)
}

/// Streams NDJSON lines from `twin agent` over the channel as they arrive.
fn agent_cmd(s: &str, k: u8, tx: mpsc::Sender<AgentEvent>) -> CmdOut {
    let mut child = Command::new("twin")
        .args(["agent", s, "-k", &k.to_string(), "--json"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    if let Some(stdout) = child.stdout.take() {
        for line in BufReader::new(stdout).lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            if let Ok(event) = serde_json::from_str::<AgentEvent>(&line) {
                if tx.send(event).is_err() {
                    break; // UI dropped the receiver — abort cleanly
                }
            }
        }
    }

    let status = child.wait()?;
    anyhow::ensure!(status.success(), "agent command failed");
    Ok(String::new())
}

fn usage_cmd() -> CmdOut {
    let raw = Command::new("twin").args(["usage", "--json"]).output()?;
    anyhow::ensure!(raw.status.success(), "{}", String::from_utf8_lossy(&raw.stderr));
    Ok(String::from_utf8(raw.stdout)?)
}

fn watch_cmd(s: &str) -> CmdOut {
    let raw = Command::new("twin").args(["watch", s, "--json"]).output()?;
    anyhow::ensure!(raw.status.success(), "{}", String::from_utf8_lossy(&raw.stderr));
    Ok(String::from_utf8(raw.stdout)?)
}

fn config_list_cmd() -> CmdOut {
    let raw = Command::new("twin").args(["config", "list", "--json"]).output()?;
    anyhow::ensure!(raw.status.success(), "{}", String::from_utf8_lossy(&raw.stderr));
    Ok(String::from_utf8(raw.stdout)?)
}

fn config_list_models_cmd() -> CmdOut {
    let raw = Command::new("twin").args(["config", "list-models", "--json"]).output()?;
    anyhow::ensure!(raw.status.success(), "{}", String::from_utf8_lossy(&raw.stderr));
    Ok(String::from_utf8(raw.stdout)?)
}
