use tokio::task;
use std::process::Command;

pub enum CMD {
    Ingest(String),
    Query(String),
    RAG(String, u8),
    Agent(String, u8),
    Usage,
    Watch(String),
    ConfigList,
    ConfigListModels,
}

fn ingest_cmd (s: String) -> String {
    let raw = Command::new("twin")
                        .arg(format!("ingest {} --json", (s)))
                        .output().expect("{}");
    let stdout = str::from_utf8(&raw.stdout);
    match stdout {
        Ok(s) => s.to_owned(),
        Err(_) => "{}".to_owned(),
    }
}

// Massive stub function rn
// Just placed some nonesense for typecheck
// need to implement calls to twin cli
pub fn call_cli(cmd: CMD) -> tokio::task::JoinHandle<String> {
    match cmd {
        CMD::Ingest(s) => task::spawn(async { ingest_cmd(s) }),
        CMD::Query(s) => task::spawn(async {"".to_owned()}),
        CMD::RAG(s, i) => task::spawn(async {"".to_owned()}),
        CMD::Agent(s, i) => task::spawn(async {"".to_owned()}),
        CMD::Usage => task::spawn(async {"".to_owned()}),
        CMD::Watch(s) => task::spawn(async {"".to_owned()}),
        CMD::ConfigList => task::spawn(async {"".to_owned()}),
        CMD::ConfigListModels => task::spawn(async {"".to_owned()})
    }
}