/// Inline citation chip. Returns true if clicked (caller should open KB browser to that doc).
pub fn show(source: &str, ui: &mut egui::Ui) -> bool {
    ui.small_button(format!("[{source}]")).clicked()
}
