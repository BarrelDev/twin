use std::sync::mpsc::Receiver;
use serde::{Deserialize, Serialize};

use crate::AgentEvent;

/// Live state for the four bottom-bar dashboard widgets.
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct DashboardState {
    // KB Stats widget
    pub total_docs: u32,
    pub total_chunks: u32,
    pub last_ingest: Option<String>,

    // Session Usage widget
    pub session_calls: u32,
    pub session_tokens: u64,
    pub session_cost: Option<f64>,

    // Active Provider widget
    pub provider: String,
    pub model: String,

    // Vault Watcher widget
    pub watcher_running: bool,
    pub last_watcher_event: Option<String>,
}

pub struct TwinUIApp {
    chat_messages: Vec<String>,
    chat_input: String,
    streaming_buffer: String,
    rx: Option<Receiver<AgentEvent>>,
    left_panel_open: bool,
    right_panel_open: bool,
    kb_docs: Vec<Option<String>>,
    selected_doc: Option<String>,
    agent_trace: Vec<String>,
    agent_input: String,
    dashboard: DashboardState,
}

impl Default for TwinUIApp {
    fn default() -> Self {
        Self {
            chat_messages: Vec::new(),
            chat_input: String::new(),
            streaming_buffer: String::new(),
            rx: None,
            left_panel_open: false,
            right_panel_open: false,
            kb_docs: Vec::new(),
            selected_doc: None,
            agent_trace: Vec::new(),
            agent_input: String::new(),
            dashboard: DashboardState::default(),
        }
    }
}

impl TwinUIApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_global_style.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        Default::default()
    }
}

impl eframe::App for TwinUIApp {
   fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
       egui::Panel::top("top_panel").show_inside(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Quit").clicked() {
                        ui.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.add_space(16.0);
                egui::widgets::global_theme_preference_buttons(ui);
            });
       });

       egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("twin_ui");
            ui.horizontal(|ui| {
                ui.label("Write something: ");
                ui.text_edit_singleline(&mut self.chat_input);
            });

            // ui.add(egui::Slider::new(&mut self.value, 0.0..=10.0).text("value"));
            // if ui.button("Increment").clicked() {
            //     self.value += 1.0;
            // }

            ui.separator();

            ui.add(egui::github_link_file!(
                "https://github.com/BarrelDev/twin/blob/master",
                "Source code."
            ));

            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                powered_by_egui_and_eframe(ui);
                egui::warn_if_debug_build(ui);
            });
       });
   }
}

fn powered_by_egui_and_eframe(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.label("Powered by ");
        ui.hyperlink_to("egui", "https://github.com/emilk/egui");
        ui.label(" and ");
        ui.hyperlink_to(
            "eframe",
            "https://github.com/emilk/egui/tree/master/crates/eframe",
        );
        ui.label(".");
    });
}