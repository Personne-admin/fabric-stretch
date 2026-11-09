//! Headless rendering of [`crate::press_sim`] scenes to PNG, on CPU or offscreen GPU.

use crate::lighting::Light;
use crate::material_package::{discover, MaterialPackage, PressBranch};
use crate::press_geometry::ViewMode;
use crate::press_render::{draw_overlay, render_cloth, RgbaImage, SceneTextures, TextureCache};
use crate::press_sim::{build_scene, Control, SceneRequest};
use crate::press_tab::{status_line, ControlMode};
use egui::{Pos2, Rect, Vec2};
use std::path::PathBuf;

pub const USAGE: &str = "\
usage: press_snapshot <project> [options]

Renders the Press View of a project (the folder holding material_package_v3*/ or material_package*/)
to a PNG, without a window.

  --out FILE            output PNG (default press_snapshot.png)
  --size WxH            image size in pixels (default 1200x700)
  --set v3|v2           package set (default: the newest the project has)
  --orientation NAME    orientation, e.g. warp or weft (default: the first discovered)
  --branch NAME         loading (default) or release
  --force N             drive by the strip tension in newtons (through the force fit)
  --indentation PCT     drive by the bar indentation in percent of the clamp span (default 12)
  --bar FRACTION        bar position as a fraction of the span (default 0.5)
  --texture tiled|untiled   texture variant (default tiled, or the one the project has)
  --relight             shade the photometric layers with a light instead of the colour textures
  --light-az DEG        light azimuth on the screen pad, counter-clockwise from screen right (default 135)
  --light-el DEG        light elevation above the sheet, 90 = straight on (default 45)
  --intensity X         diffuse intensity (default 1)
  --ambient X           ambient fill (default 0.1)
  --view oblique|top    projection (default oblique)
  --gpu                 render with the GPU shaders on an offscreen target (needs an adapter)
  -h, --help            this text
";

/// Parsed command line.
#[derive(Debug, Clone)]
pub struct SnapshotOptions {
    pub project: PathBuf,
    pub out: PathBuf,
    pub size: [u32; 2],
    pub set: Option<String>,
    pub orientation: Option<String>,
    pub branch: PressBranch,
    pub control: ControlMode,
    /// Tension in N (force mode) or percent of the span (indentation mode).
    pub amount: f32,
    pub beta: f32,
    pub tiled: bool,
    pub relight: bool,
    pub light: Light,
    pub view: ViewMode,
    pub gpu: bool,
}

impl SnapshotOptions {
    /// Options with the defaults of the command line.
    pub fn new(project: impl Into<PathBuf>) -> Self {
        Self {
            project: project.into(),
            out: "press_snapshot.png".into(),
            size: [1200, 700],
            set: None,
            orientation: None,
            branch: PressBranch::Loading,
            control: ControlMode::Indentation,
            amount: 12.0,
            beta: 0.5,
            tiled: true,
            relight: false,
            light: Light::default(),
            view: ViewMode::Oblique,
            gpu: false,
        }
    }

    /// Parses the arguments after the program name. `Ok(None)` means help was requested.
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Self>, String> {
        let mut args = args.into_iter();
        let mut options: Option<Self> = None;
        let number = |name: &str, value: Option<String>| -> Result<f32, String> {
            let value = value.ok_or_else(|| format!("{name} needs a value"))?;
            value
                .parse::<f32>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| format!("{name}: {value:?} is not a number"))
        };
        while let Some(arg) = args.next() {
            if arg == "-h" || arg == "--help" {
                return Ok(None);
            }
            if !arg.starts_with("--") {
                let o = options.get_or_insert_with(|| Self::new(PathBuf::new()));
                if !o.project.as_os_str().is_empty() {
                    return Err(format!("unexpected argument {arg:?}"));
                }
                o.project = arg.into();
                continue;
            }
            let o = options.get_or_insert_with(|| Self::new(PathBuf::new()));
            match arg.as_str() {
                "--out" => o.out = args.next().ok_or("--out needs a file")?.into(),
                "--size" => {
                    let value = args.next().ok_or("--size needs WxH")?;
                    let (w, h) = value
                        .split_once(['x', 'X'])
                        .ok_or_else(|| format!("--size: {value:?} is not WxH"))?;
                    let parse = |s: &str| s.parse::<u32>().ok().filter(|v| (16..=8192).contains(v));
                    o.size = [
                        parse(w).ok_or_else(|| format!("--size: bad width in {value:?}"))?,
                        parse(h).ok_or_else(|| format!("--size: bad height in {value:?}"))?,
                    ];
                }
                "--set" => o.set = Some(args.next().ok_or("--set needs v3 or v2")?),
                "--orientation" => {
                    o.orientation = Some(args.next().ok_or("--orientation needs a name")?)
                }
                "--branch" => {
                    let value = args.next().ok_or("--branch needs loading or release")?;
                    o.branch = match value.as_str() {
                        "loading" => PressBranch::Loading,
                        "release" => PressBranch::Release,
                        _ => return Err(format!("--branch: {value:?} is not loading or release")),
                    };
                }
                "--force" => {
                    o.control = ControlMode::Force;
                    o.amount = number("--force", args.next())?;
                }
                "--indentation" => {
                    o.control = ControlMode::Indentation;
                    o.amount = number("--indentation", args.next())?;
                }
                "--bar" => o.beta = number("--bar", args.next())?,
                "--texture" => {
                    let value = args.next().ok_or("--texture needs tiled or untiled")?;
                    o.tiled = match value.as_str() {
                        "tiled" => true,
                        "untiled" => false,
                        _ => return Err(format!("--texture: {value:?} is not tiled or untiled")),
                    };
                }
                "--relight" => o.relight = true,
                "--light-az" => o.light.azimuth_deg = number("--light-az", args.next())?,
                "--light-el" => o.light.elevation_deg = number("--light-el", args.next())?,
                "--intensity" => o.light.intensity = number("--intensity", args.next())?,
                "--ambient" => o.light.ambient = number("--ambient", args.next())?,
                "--view" => {
                    let value = args.next().ok_or("--view needs oblique or top")?;
                    o.view = match value.as_str() {
                        "oblique" => ViewMode::Oblique,
                        "top" => ViewMode::Top,
                        _ => return Err(format!("--view: {value:?} is not oblique or top")),
                    };
                }
                "--gpu" => o.gpu = true,
                other => return Err(format!("unknown option {other}")),
            }
        }
        match options {
            Some(o) if !o.project.as_os_str().is_empty() => Ok(Some(o)),
            _ => Err("missing <project>".into()),
        }
    }
}

/// Background colour behind the sheet (the dark panel colour of the viewer).
const BACKGROUND: [u8; 4] = [27, 27, 27, 255];

/// Renders the snapshot; returns the image and the status line of the scene.
pub fn render(options: &SnapshotOptions) -> Result<(RgbaImage, String), String> {
    let found = discover(&options.project);
    let sets = &found.sets;
    let set = match &options.set {
        Some(id) => sets
            .iter()
            .find(|s| s.id == id.as_str())
            .ok_or_else(|| format!("{}: no package set {id:?}", options.project.display()))?,
        None => sets.first().ok_or_else(|| {
            let mut message = format!(
                "{}: no material package (looked for material_package_v3 and material_package)",
                options.project.display()
            );
            for problem in &found.problems {
                message.push_str(&format!("\n  skipped {problem}"));
            }
            message
        })?,
    };
    let dirs = match &options.orientation {
        Some(name) => set.orientation(name).ok_or_else(|| {
            let known: Vec<_> = set.orientations.iter().map(|o| o.name.as_str()).collect();
            format!(
                "no orientation {name:?} in package set {} (have: {})",
                set.id,
                known.join(", ")
            )
        })?,
        None => set
            .orientations
            .first()
            .ok_or("the package set has no orientation")?,
    };
    let variant = dirs.choose(options.tiled);
    let dir = variant
        .dir
        .clone()
        .ok_or("the orientation has no package")?;
    let package = MaterialPackage::load(&dir)?;

    let size = options.size;
    let canvas = Rect::from_min_size(Pos2::ZERO, Vec2::new(size[0] as f32, size[1] as f32));
    let control = match options.control {
        ControlMode::Force => Control::Force(options.amount),
        ControlMode::Indentation => Control::Indentation(options.amount / 100.0),
    };
    if options.control == ControlMode::Force && package.force_fit(options.branch).is_none() {
        return Err(format!(
            "the {} branch has no force fit; use --indentation",
            options.branch.name()
        ));
    }
    let request = SceneRequest {
        branch: options.branch,
        control,
        beta: options.beta.clamp(
            crate::press_sim::BETA_RANGE.0,
            crate::press_sim::BETA_RANGE.1,
        ),
        previous_lambda: 1.0,
        view: options.view,
        relight: options.relight.then_some(options.light),
    };
    let scene = build_scene(&package, &request, canvas)?;
    let mut cache = TextureCache::new(8);
    let textures = SceneTextures::load(&scene.cloth.files, &mut cache)?;
    let cloth = if options.gpu {
        render_gpu(&scene.cloth, &textures, size)?
    } else {
        render_cloth(&scene.cloth, &textures, size, Pos2::ZERO, 1.0)
    };
    let mut image = RgbaImage::new(size[0], size[1]);
    image.fill(BACKGROUND);
    image.composite(&cloth);
    draw_overlay(&mut image, &scene.overlay, Pos2::ZERO, 1.0);
    let status = status_line(
        &scene,
        &package,
        set.label,
        &dirs.name,
        variant.tiled,
        options.relight,
        options.control,
    );
    Ok((image, status))
}

fn render_gpu(
    cloth: &crate::press_render::ClothScene,
    textures: &SceneTextures,
    size: [u32; 2],
) -> Result<RgbaImage, String> {
    use crate::press_render::MipChain;
    use std::sync::Arc;
    let ctx = crate::press_gpu::GpuContext::headless().ok_or("--gpu: no GPU adapter found")?;
    let upload =
        |chains: &[&Arc<MipChain>]| -> Result<Vec<Arc<crate::press_gpu::GpuTexture>>, String> {
            chains.iter().map(|c| ctx.upload(c).map(Arc::new)).collect()
        };
    let gpu_textures = match textures {
        SceneTextures::Rgb(c) => upload(&[&c[0], &c[1]])?,
        SceneTextures::Relight { albedo, normal } => {
            upload(&[&albedo[0], &albedo[1], &normal[0], &normal[1]])?
        }
    };
    ctx.render_offscreen(cloth, gpu_textures, size, Pos2::ZERO, 1.0)
}

/// The command line entry point: parses `args`, renders and writes the PNG.
pub fn run(args: impl IntoIterator<Item = String>) -> Result<(), String> {
    let Some(options) = SnapshotOptions::parse(args)? else {
        println!("{USAGE}");
        return Ok(());
    };
    let (image, status) = render(&options)?;
    image.save_png(&options.out)?;
    println!("{status}");
    println!("wrote {}", options.out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material_package::test_support::demo_project;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_command_line_is_parsed() {
        let o = SnapshotOptions::parse(args(&[
            "proj",
            "--orientation",
            "warp",
            "--branch",
            "release",
            "--force",
            "2.5",
            "--bar",
            "0.4",
            "--texture",
            "untiled",
            "--relight",
            "--light-az",
            "200",
            "--light-el",
            "30",
            "--view",
            "top",
            "--out",
            "x.png",
            "--size",
            "640x480",
            "--gpu",
            "--set",
            "v2",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(o.project, PathBuf::from("proj"));
        assert_eq!(
            (o.branch, o.control, o.amount, o.beta),
            (PressBranch::Release, ControlMode::Force, 2.5, 0.4)
        );
        assert!(!o.tiled && o.relight && o.gpu);
        assert_eq!((o.light.azimuth_deg, o.light.elevation_deg), (200.0, 30.0));
        assert_eq!(
            (o.view, o.size, o.set.as_deref()),
            (ViewMode::Top, [640, 480], Some("v2"))
        );
        // Options may precede the project; --indentation switches the control back.
        let o = SnapshotOptions::parse(args(&["--force", "1", "--indentation", "20", "proj"]))
            .unwrap()
            .unwrap();
        assert_eq!((o.control, o.amount), (ControlMode::Indentation, 20.0));
        assert!(SnapshotOptions::parse(args(&["--help"])).unwrap().is_none());
    }

    #[test]
    fn bad_command_lines_name_the_problem() {
        for (list, needle) in [
            (vec!["--out", "a.png"], "missing <project>"),
            (vec!["p", "--size", "12x"], "--size"),
            (vec!["p", "--branch", "up"], "--branch"),
            (vec!["p", "--force"], "needs a value"),
            (vec!["p", "--force", "abc"], "not a number"),
            (vec!["p", "--nonsense"], "unknown option"),
            (vec!["p", "q"], "unexpected argument"),
        ] {
            let error = SnapshotOptions::parse(args(&list)).unwrap_err();
            assert!(error.contains(needle), "{list:?}: {error}");
        }
    }

    #[test]
    fn the_snapshot_of_the_demo_is_drawn() {
        let out =
            std::env::temp_dir().join(format!("press_snapshot_smoke_{}.png", std::process::id()));
        let project = demo_project();
        run(args(&[
            project.to_str().unwrap(),
            "--indentation",
            "15",
            "--size",
            "480x300",
            "--out",
            out.to_str().unwrap(),
        ]))
        .unwrap();
        let png = image::open(&out).unwrap().to_rgba8();
        assert_eq!(png.dimensions(), (480, 300));
        assert_eq!(
            png.get_pixel(2, 2).0,
            BACKGROUND,
            "the corner is background"
        );
        let centre = png.get_pixel(200, 150).0;
        assert_ne!(
            centre, BACKGROUND,
            "the sheet covers the middle: {centre:?}"
        );
        std::fs::remove_file(&out).ok();
    }

    #[test]
    fn missing_things_are_reported() {
        let mut o = SnapshotOptions::new(std::env::temp_dir().join("press_snapshot_nothing_here"));
        assert!(render(&o).unwrap_err().contains("no material package"));
        o = SnapshotOptions::new(demo_project());
        o.orientation = Some("weft".into());
        assert!(render(&o).unwrap_err().contains("have: warp"));
        o.orientation = None;
        o.set = Some("v2".into());
        assert!(render(&o).unwrap_err().contains("no package set"));
    }

    #[test]
    fn relight_and_force_modes_render() {
        let mut o = SnapshotOptions::new(demo_project());
        o.size = [320, 200];
        o.relight = true;
        o.control = ControlMode::Force;
        o.amount = 1.5;
        let (image, status) = render(&o).unwrap();
        assert!(
            status.contains("relit") && status.contains("F 1.5"),
            "{status}"
        );
        assert_ne!(image.get(160, 100)[..3], BACKGROUND[..3]);
    }
}
