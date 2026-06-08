use crate::app::TwinUIApp;

/// Central chat panel: message history, streaming response, source chips, input bar.
/// Renders into the remaining ui area after all side panels — must be called last.
pub fn show(_app: &mut TwinUIApp, ui: &mut egui::Ui) {
    // TODO: scrollable message history (chat_messages)
    // TODO: streaming_text::show for in-progress response (streaming_buffer)
    // TODO: source_chip::show for inline citations
    // TODO: input bar — text_edit_multiline(chat_input) + Send button
    // TODO: mode toggle (RAG vs Query)
    ui.text_edit_multiline(&mut _app.chat_input);
}
