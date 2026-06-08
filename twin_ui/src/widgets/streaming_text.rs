/// Renders streaming text token-by-token. Call each frame with the accumulated buffer.
pub fn show(text: &str, ui: &mut egui::Ui) {
    ui.label(text);
}
