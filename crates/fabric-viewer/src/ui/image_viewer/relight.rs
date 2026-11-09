//! Image Viewer relight requests and uploads. Always use the frame's PS albedo and normals;
//! show a result only while its [`RelightSignature`] matches.

use std::path::PathBuf;

use super::ImageViewerState;
use crate::images::relight::RelightSignature;
use crate::images::{ImageCache, ImageStatus};
use crate::settings::RelightSettings;

/// The files a relight needs for the current frame.
#[derive(Debug, Clone, Default)]
pub(super) struct RelightSources {
    /// Albedo role file and the reflectance its byte 255 stands for.
    pub albedo: Option<(PathBuf, f32)>,
    pub normal: Option<PathBuf>,
    /// The run has PS layers at all (shapes the hint when a frame lacks them).
    pub run_has_ps: bool,
}

/// Why the relit preview is (not) being shown; drives the display texture and
/// an inline status line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum RelightState {
    /// The displayed relit texture matches the current signature.
    Active,
    /// A recompute is in flight; the previous texture (if any) keeps showing
    /// until it lands.
    Busy,
    Disabled,
    /// The frame has no PS albedo to shade; the reason is human-readable.
    NoAlbedo(String),
    /// The frame has no normal-map role to relight with.
    NoNormal,
    /// A source image failed to load; the reason is human-readable.
    SourceError(String),
    WaitingForDecode,
}

impl RelightState {
    pub fn note(&self) -> Option<String> {
        match self {
            RelightState::Busy => Some("relighting...".to_owned()),
            RelightState::NoAlbedo(reason) => Some(format!("relight: {reason}")),
            RelightState::NoNormal => Some("relight: no normal map role on this frame".to_owned()),
            RelightState::SourceError(err) => Some(format!("relight: {err}")),
            RelightState::WaitingForDecode => {
                Some("relight: waiting for image decode...".to_owned())
            }
            RelightState::Active | RelightState::Disabled => None,
        }
    }

    /// Whether the relit texture should be displayed in place of the image.
    pub fn shows_relit(&self, have_texture: bool) -> bool {
        match self {
            RelightState::Active => true,
            RelightState::Busy => have_texture,
            _ => false,
        }
    }
}

/// The relit image and the texture made from it.
pub struct RelightEntry {
    pub signature: RelightSignature,
    pub image: image::RgbaImage,
    pub texture: egui::TextureHandle,
    pub options: egui::TextureOptions,
}

impl std::fmt::Debug for RelightEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RelightEntry")
            .field("signature", &self.signature)
            .finish_non_exhaustive()
    }
}

fn upload(
    ctx: &egui::Context,
    image: &image::RgbaImage,
    options: egui::TextureOptions,
) -> egui::TextureHandle {
    let size = [image.width() as usize, image.height() as usize];
    let color = egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw());
    ctx.load_texture("fabsim-relight", color, options)
}

impl ImageViewerState {
    /// Upload the newest finished generation and repaint while work is pending.
    pub fn poll_relight(&mut self, ctx: &egui::Context, options: egui::TextureOptions) {
        while let Some(result) = self.relight_worker.try_recv_result() {
            // Require both generation and signature to match, including after a parameter revert.
            let is_latest = result.generation == self.relight_generation
                && self
                    .relight_requested
                    .as_ref()
                    .is_some_and(|s| *s == result.signature);
            if is_latest {
                let texture = upload(ctx, &result.image, options);
                self.relight = Some(RelightEntry {
                    signature: result.signature,
                    image: result.image,
                    texture,
                    options,
                });
                self.relight_pending = false;
            }
        }
        // Sampler changes need a texture upload, not new shading.
        if let Some(entry) = &mut self.relight {
            if entry.options != options {
                entry.texture = upload(ctx, &entry.image, options);
                entry.options = options;
            }
        }
        if self.relight_pending {
            ctx.request_repaint();
        }
    }

    fn drop_relight(&mut self) {
        self.relight = None;
        self.relight_pending = false;
        self.relight_requested = None;
    }

    /// Reuse matching output and invalidate superseded work. Return whether output is current.
    pub(super) fn invalidate_superseded_inflight(&mut self, signature: &RelightSignature) -> bool {
        if self
            .relight
            .as_ref()
            .is_some_and(|e| e.signature == *signature)
        {
            if self.relight_pending {
                self.relight_generation += 1;
                self.relight_pending = false;
                self.relight_requested = None;
            }
            true
        } else {
            false
        }
    }

    /// Relight cached pixels in the background when sources or lighting change.
    pub(super) fn request_relight(
        &mut self,
        image_cache: &mut ImageCache,
        relight: &RelightSettings,
        sources: &RelightSources,
    ) -> RelightState {
        if !relight.enabled {
            self.drop_relight();
            return RelightState::Disabled;
        }
        let Some((albedo_path, albedo_scale)) = &sources.albedo else {
            self.drop_relight();
            return RelightState::NoAlbedo(if sources.run_has_ps {
                "this frame has no PS albedo".to_owned()
            } else {
                "this run has no PS albedo / normal layers (run `fabsim ps <project>`)".to_owned()
            });
        };
        let Some(normal_path) = &sources.normal else {
            self.drop_relight();
            return RelightState::NoNormal;
        };
        for (path, label) in [(albedo_path, "albedo"), (normal_path, "normal")] {
            if let Some(ImageStatus::Error(err)) = image_cache.status(path) {
                self.drop_relight();
                return RelightState::SourceError(format!("{label}: {err}"));
            }
        }
        let signature = RelightSignature {
            albedo_key: albedo_path.clone(),
            normal_key: normal_path.clone(),
            albedo_scale: *albedo_scale,
            light: relight.light,
        };
        if self.invalidate_superseded_inflight(&signature) {
            return RelightState::Active;
        }
        if self.relight_pending && self.relight_requested.as_ref() == Some(&signature) {
            return RelightState::Busy;
        }
        let (Some(albedo), Some(normal)) = (
            image_cache.decoded(albedo_path),
            image_cache.decoded(normal_path),
        ) else {
            // Clear the previous frame's relight while the new sources decode.
            self.drop_relight();
            return RelightState::WaitingForDecode;
        };
        self.relight_generation += 1;
        if !self
            .relight_worker
            .submit(self.relight_generation, signature.clone(), albedo, normal)
        {
            // Report worker failure instead of repainting Busy forever.
            self.relight_pending = false;
            self.relight_requested = None;
            return RelightState::SourceError("relight worker unavailable".to_owned());
        }
        self.relight_pending = true;
        self.relight_requested = Some(signature);
        RelightState::Busy
    }
}
