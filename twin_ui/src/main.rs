#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() -> eframe::Result {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 300.0])
            .with_min_inner_size([300.0, 200.0])
            .with_icon(
                eframe::icon_data::from_png_bytes(
                    &include_bytes!("../assets/favicon-512x512.png")[..],
                )
                .expect("Failed to load icon")
            ),
        ..Default::default()
    };
    eframe::run_native(
        "twin_ui", 
        native_options, 
        Box::new(|cc| Ok(Box::new(twin_ui::TwinUIApp::new(cc)))),
    )
}