use std::sync::mpsc::Receiver;
use serde::{Deserialize, Serialize};
use egui::Image;

use crate::AgentEvent;

// Visual design constants — defined here, referenced everywhere (never hardcoded elsewhere).
pub const BG_PRIMARY:   egui::Color32 = egui::Color32::from_rgb(10, 10, 15);
pub const BG_SECONDARY: egui::Color32 = egui::Color32::from_rgb(18, 18, 28);
pub const ACCENT:       egui::Color32 = egui::Color32::from_rgb(61, 90, 254);
pub const TEXT_PRIMARY: egui::Color32 = egui::Color32::from_rgb(220, 220, 235);
pub const TEXT_DIM:     egui::Color32 = egui::Color32::from_rgb(120, 120, 145);
pub const SUCCESS:      egui::Color32 = egui::Color32::from_rgb(40, 200, 100);
pub const WARNING:      egui::Color32 = egui::Color32::from_rgb(255, 180, 0);

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
    pub chat_messages: Vec<String>,
    pub chat_input: String,
    pub streaming_buffer: String,
    pub rx: Option<Receiver<AgentEvent>>,
    pub left_panel_open: bool,
    pub right_panel_open: bool,
    pub kb_docs: Vec<Option<String>>,
    pub selected_doc: Option<String>,
    pub agent_trace: Vec<String>,
    pub agent_input: String,
    pub dashboard: DashboardState,
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
        egui_extras::install_image_loaders(&cc.egui_ctx);
        Default::default()
    }
}

impl eframe::App for TwinUIApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // ui.visuals_mut().code_bg_color = BG_PRIMARY;
        // show_inside() order is enforced by egui: top/bottom → sides → remaining central area.
        egui::Panel::top("top_bar").show_inside(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                let img = Image::new(egui::include_image!("../assets/favicon-512x512.png"));
                if ui.button(img).clicked() {
                    self.left_panel_open = !self.left_panel_open;
                }
                ui.menu_button("Menu", |ui| {
                    egui::widgets::global_theme_preference_buttons(ui);
                    if ui.button("Quit").clicked() {
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });
        });

        egui::Panel::bottom("bottom_bar").show_inside(ui, |ui| {
            crate::widgets::dashboard::show(&self.dashboard, ui);
        });

        crate::panels::kb_browser::show(self, ui);
        crate::panels::agent::show(self, ui);
        crate::panels::chat::show(self, ui); // remaining central area — must be last
    }
}