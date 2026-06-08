use crate::app::DashboardState;

/// Bottom-bar dashboard: KB stats, session usage, active provider, vault watcher.
pub fn show(state: &DashboardState, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        kb_stats(state, ui);
        ui.separator();
        session_usage(state, ui);
        ui.separator();
        active_provider(state, ui);
        ui.separator();
        vault_watcher(state, ui);
    });
}

fn kb_stats(state: &DashboardState, ui: &mut egui::Ui) {
    let last = state
        .last_ingest
        .as_deref()
        .map(|t| format!(" | Last: {t}"))
        .unwrap_or_default();
    ui.label(format!("Docs: {} | Chunks: {}{}", state.total_docs, state.total_chunks, last));
}

fn session_usage(state: &DashboardState, ui: &mut egui::Ui) {
    let cost = state
        .session_cost
        .map(|c| format!(" | ${c:.4}"))
        .unwrap_or_default();
    ui.label(format!(
        "Calls: {} | Tokens: {}{}",
        state.session_calls, state.session_tokens, cost
    ));
}

fn active_provider(state: &DashboardState, ui: &mut egui::Ui) {
    ui.label(format!("{} / {}", state.provider, state.model));
}

fn vault_watcher(state: &DashboardState, ui: &mut egui::Ui) {
    let status = if state.watcher_running { "● Running" } else { "○ Stopped" };
    ui.label(status);
    if let Some(event) = &state.last_watcher_event {
        ui.label(event.as_str());
    }
}
