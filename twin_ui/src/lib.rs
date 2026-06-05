use eframe::egui;

mod app;
pub use app::TwinUIApp;

mod subprocess;
pub use subprocess::CMD;
pub use subprocess::call_cli;