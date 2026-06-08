use crate::app::TwinUIApp;

/// Settings panel: provider, model, and API key configuration.
/// Shown as an overlay window triggered by the top-bar gear icon.
pub fn show(_app: &mut TwinUIApp, _ui: &mut egui::Ui) {
    // TODO: egui::Window::new("Settings").show(ctx, |ui| { ... })
    //   → provider selector (dropdown from ConfigInfo.keys_present)
    //   → model selector (populated from twin config list-models --json)
    //   → API key input (password field, never shown in plain text)
    //   → Save button → call_cli(CMD::ConfigList) to refresh dashboard.provider/model
}
