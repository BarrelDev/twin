use crate::app::TwinUIApp;

/// Right agent panel: task input, step-by-step reasoning trace, activity indicator.
/// Hidden when app.right_panel_open is false.
pub fn show(app: &mut TwinUIApp, ui: &mut egui::Ui) {
    if !app.right_panel_open {
        return;
    }
    egui::Panel::right("agent_panel")
        .default_size(280.0)
        .show_inside(ui, |_ui| {
            // TODO: agent_input text field + Run button
            //   → on run: call_cli(CMD::Agent(task, k, tx)), store rx in app.rx
            // TODO: poll app.rx each frame, push AgentEvents into app.agent_trace
            // TODO: scrollable trace log (agent_trace), monospace font for tool calls
            // TODO: animated activity indicator while agent is running
        });
}
