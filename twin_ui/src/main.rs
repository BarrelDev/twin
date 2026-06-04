fn main() {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native("twin_ui", native_options, Box::new(|cc| Ok(Box::new(twin_ui::TwinUIApp::new(cc)))));
}