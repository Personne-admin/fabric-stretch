//! Headless PNG renderer. Run `cargo run --release --example press_snapshot -- --help`
//! for options; image recipes are in `docs/press_view.md`.

fn main() {
    if let Err(error) = fabric_viewer::press_snapshot::run(std::env::args().skip(1)) {
        eprintln!("press_snapshot: {error}");
        std::process::exit(2);
    }
}
