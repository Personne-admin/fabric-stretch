//! Renders the committed demo project through the public API of the headless snapshot, using
//! the CPU renderer (no GPU adapter needed), and checks the picture.

use std::path::PathBuf;

use fabric_viewer::press_render::RgbaImage;
use fabric_viewer::press_snapshot::{render, SnapshotOptions};

fn demo_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/demo")
}

fn distinct_colours(image: &RgbaImage) -> usize {
    let mut seen = std::collections::HashSet::new();
    for pixel in image.pixels.as_chunks::<4>().0 {
        seen.insert(*pixel);
        if seen.len() > 64 {
            break;
        }
    }
    seen.len()
}

#[test]
fn demo_renders_a_non_uniform_image_of_the_requested_size() {
    let mut options = SnapshotOptions::new(demo_dir());
    options.size = [320, 200];
    let (image, status) = render(&options).expect("the demo project renders");
    assert_eq!((image.width, image.height), (320, 200));
    assert!(!status.is_empty());
    assert!(
        distinct_colours(&image) > 16,
        "the image is (nearly) uniform"
    );
}

#[test]
fn demo_relight_differs_from_the_colour_texture() {
    let mut plain = SnapshotOptions::new(demo_dir());
    plain.size = [240, 150];
    let mut relit = plain.clone();
    relit.relight = true;
    let (a, _) = render(&plain).expect("plain render");
    let (b, _) = render(&relit).expect("relit render");
    assert_eq!((a.width, a.height), (b.width, b.height));
    assert!(distinct_colours(&b) > 16);
    assert_ne!(a.pixels, b.pixels);
}
