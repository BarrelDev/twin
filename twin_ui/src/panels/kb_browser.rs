use crate::app::TwinUIApp;

/// Left KB browser panel: document list, search bar, graph view toggle.
/// Hidden when app.left_panel_open is false.
pub fn show(app: &mut TwinUIApp, ui: &mut egui::Ui) {
    if !app.left_panel_open {
        return;
    }
    egui::Panel::left("kb_browser")
        .default_size(240.0)
        .show_inside(ui, |_ui| {
            // TODO: search bar (text_edit_singleline)
            // TODO: scrollable doc list (kb_docs), click sets selected_doc
            // TODO: graph view toggle — renders graph::KnowledgeGraph via egui Painter
            _ui.text_edit_singleline(&mut app.agent_input);
        });
}
