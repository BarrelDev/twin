use serde::{Deserialize, Serialize};

const MAX_EVENTS: usize = 50;

/// One entry in the in-process session event log.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEvent {
    QueryAsked { query: String, timestamp: String },
    RagResponse { answer: String, sources: Vec<String>, timestamp: String },
    AgentTask { task: String, tool_calls: Vec<String>, answer: String, timestamp: String },
    ToolResult { tool: String, input: String, output: String, timestamp: String },
}

/// In-process session event log. Never persisted directly — feeds distillation at shutdown.
#[derive(Debug, Default)]
pub struct SessionMemory {
    events: Vec<SessionEvent>,
}

impl SessionMemory {
    /// Append an event, evicting the oldest when the 50-event cap is reached.
    pub fn push(&mut self, event: SessionEvent) {
        if self.events.len() >= MAX_EVENTS {
            self.events.remove(0);
        }
        self.events.push(event);
    }

    /// Last `n` events for agent context assembly (returns fewer if log is short).
    pub fn recent(&self, n: usize) -> &[SessionEvent] {
        let start = self.events.len().saturating_sub(n);
        &self.events[start..]
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}
