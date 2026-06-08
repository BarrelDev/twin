use tokio::task;
use std::{any, process::Command};

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

pub type CmdOut = anyhow::Result<String>;

pub fn call_cli(cmd: CMD) -> tokio::task::JoinHandle<CmdOut> {
    match cmd {
        CMD::Ingest(path) => task::spawn_blocking(move || {
                ingest_cmd(&path)
            }),
        CMD::Query(query) => task::spawn_blocking(move || {
            query_cmd(&query)
        }),
        CMD::RAG(query, i) => task::spawn_blocking(move || {
            rag_cmd(&query, i)
        }),
        CMD::Agent(query, i) => task::spawn_blocking(move || {
            agent_cmd(&query, i)
        }),
        CMD::Usage => task::spawn_blocking(move || {
            usage_cmd()
        }),
        CMD::Watch(path) => task::spawn_blocking(move || {
            watch_cmd(&path)
        }),
        CMD::ConfigList => task::spawn_blocking(move || {
            config_list_cmd()
        }),
        CMD::ConfigListModels => task::spawn_blocking(move || {
            config_list_models_cmd()
        })
    }
}

fn ingest_cmd (s: &str) -> CmdOut {
    let raw = Command::new("twin")
                        .args(&["ingest", s, "--json"])
                        .output()?;

    anyhow::ensure!(raw.status.success(),
        "{}",
        String::from_utf8_lossy(&raw.stderr)
    );

    Ok(String::from_utf8(raw.stdout)?)
}

fn query_cmd (s: &str) -> CmdOut {
    let raw = Command::new("twin")
                                .args(&["query", s, "--json"])
                                .output()?;

    anyhow::ensure!(raw.status.success(),
        "{}",
        String::from_utf8_lossy(&raw.stderr)
    );

    Ok(String::from_utf8(raw.stdout)?)
}

fn rag_cmd (s: &str, i: u8) -> CmdOut {
    let raw = Command::new("twin")
                                .args(&["rag", s, "-k", &i.to_string(), "--json"])
                                .output()?;

    anyhow::ensure!(raw.status.success(),
        "{}",
        String::from_utf8_lossy(&raw.stderr)
    );

    Ok(String::from_utf8(raw.stdout)?)
}

fn agent_cmd (s: &str, i: u8) -> CmdOut {
    let raw = Command::new("twin")
                                .args(&["agent", s, "-k", &i.to_string(), "--json"])
                                .output()?;

    anyhow::ensure!(raw.status.success(),
        "{}",
        String::from_utf8_lossy(&raw.stderr)
    );

    Ok(String::from_utf8(raw.stdout)?)
}

fn usage_cmd() -> CmdOut {
    let raw = Command::new("twin")
                                .args(&["usage", "--json"])
                                .output()?;

    anyhow::ensure!(raw.status.success(),
        "{}",
        String::from_utf8_lossy(&raw.stderr)
    );

    Ok(String::from_utf8(raw.stdout)?)
}

fn watch_cmd(s: &str) -> CmdOut {
    let raw = Command::new("twin")
                                .args(&["watch", s, "--json"])
                                .output()?;

    anyhow::ensure!(raw.status.success(),
        "{}",
        String::from_utf8_lossy(&raw.stderr)
    );

    Ok(String::from_utf8(raw.stdout)?)
}

fn config_list_cmd() -> CmdOut {
    let raw = Command::new("twin")
                                .args(&["config", "list", "--json"])
                                .output()?;

    anyhow::ensure!(raw.status.success(),
        "{}",
        String::from_utf8_lossy(&raw.stderr)
    );

    Ok(String::from_utf8(raw.stdout)?)
}

fn config_list_models_cmd() -> CmdOut {
    let raw = Command::new("twin")
                                .args(&["config", "list-models", "--json"])
                                .output()?;

    anyhow::ensure!(raw.status.success(),
        "{}",
        String::from_utf8_lossy(&raw.stderr)
    );

    Ok(String::from_utf8(raw.stdout)?)
}
