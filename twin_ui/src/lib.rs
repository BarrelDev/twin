use eframe::egui;

mod app;
pub use app::TwinUIApp;

mod subprocess;
pub use subprocess::{
    call_cli, AgentEvent, CMD, CmdOut, ConfigInfo, IngestResult, QueryResult, RagResponse,
    RagSource, RagUsage, UsageRecord,
};