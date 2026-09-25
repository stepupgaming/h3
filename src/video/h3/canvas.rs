//! Product canvas presets for MiniMax-H3 generate / continue / upscale.
//!
//! Owner ladder (16:9-ish, axes ×32):
//!    0.2 MP       →  608 × 352   (~0.214 MP)  ganloss Eros first pass
//!   480p / 0.4 MP →  864 × 480   (~0.415 MP)  native DiT default
//!   720p / 0.9 MP → 1280 × 736   (~0.942 MP)  native DiT quality
//!    1.0 MP       → 1344 × 768   (~1.032 MP)  ganloss Eros two-stage finish
//!  1080p / 2.0 MP → 1920 × 1088  (~2.089 MP)  deliverable / upscale target
//!
//! Open Base DiT area cap is ~1.032 MP, so **1080p is not a native sample
//! canvas** — use 720p (or 480p) for generate/continue, then
//! `h3 upscale --canvas 1080p`.

use anyhow::{bail, Result};
use clap::ValueEnum;
use serde::Serialize;

/// Fixed product sizes (width, height, nominal MP label, short name).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
pub(crate) enum H3CanvasPreset {
    /// 608×352 (~0.214 MP) — ganloss Eros first pass (video LByGCGzu67o).
    #[value(name = "0.2", alias = "mp02", alias = "p352")]
    Mp02,
    /// 864×480 (~0.415 MP) — fast preview / default.
    #[value(name = "480p", alias = "0.4", alias = "0.40", alias = "p480")]
    P480,
    /// 1280×736 (~0.942 MP) — quality native sample.
    #[value(name = "720p", alias = "0.9", alias = "0.90", alias = "p720")]
    P720,
    /// 1344×768 (~1.032 MP) — ganloss Eros two-stage finish (video 1344×768).
    #[value(name = "1.0", alias = "0.98", alias = "p1344", alias = "mp10")]
    Mp10,
    /// 1920×1088 (~2.089 MP) — upscale / deliverable target only.
    #[value(name = "1080p", alias = "2.0", alias = "2.00", alias = "p1080")]
    P1080,
}

impl H3CanvasPreset {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Mp02 => "0.2",
            Self::P480 => "480p",
            Self::P720 => "720p",
            Self::Mp10 => "1.0",
            Self::P1080 => "1080p",
        }
    }

    /// Exact pixel size (already ×32).
    pub(crate) fn size(self) -> (u32, u32) {
        match self {
            Self::Mp02 => (608, 352),
            Self::P480 => (864, 480),
            Self::P720 => (1280, 736),
            Self::Mp10 => (1344, 768),
            Self::P1080 => (1920, 1088),
        }
    }

    pub(crate) fn megapixels(self) -> f64 {
        let (w, h) = self.size();
        f64::from(w) * f64::from(h) / 1_000_000.0
    }

    /// Native DiT sample is legal on open Base (area ≤ ~1.032 MP).
    pub(crate) fn is_native_sample(self) -> bool {
        !matches!(self, Self::P1080)
    }

    pub(crate) fn label(self) -> String {
        let (w, h) = self.size();
        format!(
            "{} ({}×{}, ~{:.2} MP)",
            self.as_str(),
            w,
            h,
            self.megapixels()
        )
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(crate) struct ResolvedCanvas {
    pub width: u32,
    pub height: u32,
    pub preset: Option<H3CanvasPreset>,
    pub megapixels: f64,
}

impl ResolvedCanvas {
    pub(crate) fn label(&self) -> String {
        if let Some(p) = self.preset {
            return p.label();
        }
        format!(
            "{}×{} (~{:.2} MP)",
            self.width, self.height, self.megapixels
        )
    }
}

/// Resolve generate/continue canvas from optional `--canvas` and/or W×H.
///
/// Rules:
/// - `--canvas 480p|720p` → fixed size (overrides bare defaults)
/// - `--canvas 1080p` on **sample** paths → error (upscale-only)
/// - explicit `--width`+`--height` when no canvas
/// - default → 480p
/// - canvas + matching explicit size OK; canvas + conflicting size → error
pub(crate) fn resolve_sample_canvas(
    canvas: Option<H3CanvasPreset>,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<ResolvedCanvas> {
    if let Some(preset) = canvas {
        if !preset.is_native_sample() {
            let (tw, th) = preset.size();
            bail!(
                "--canvas {} ({}×{}, ~{:.2} MP) is above the open Base DiT area cap (~1.03 MP).\n\
                 Sample at --canvas 720p (or 480p), then upscale:\n\
                   h3 upscale --input out.mp4 --canvas 1080p -o out_1080p.mp4",
                preset.as_str(),
                tw,
                th,
                preset.megapixels()
            );
        }
        let (pw, ph) = preset.size();
        if let (Some(w), Some(h)) = (width, height) {
            if w != pw || h != ph {
                bail!(
                    "--canvas {} is {}×{}, but --width/--height are {}×{} — pick one",
                    preset.as_str(),
                    pw,
                    ph,
                    w,
                    h
                );
            }
        } else if width.is_some() ^ height.is_some() {
            bail!("when overriding a canvas preset, pass both --width and --height (or neither)");
        }
        return Ok(ResolvedCanvas {
            width: pw,
            height: ph,
            preset: Some(preset),
            megapixels: preset.megapixels(),
        });
    }

    match (width, height) {
        (None, None) => {
            let p = H3CanvasPreset::P480;
            let (w, h) = p.size();
            Ok(ResolvedCanvas {
                width: w,
                height: h,
                preset: Some(p),
                megapixels: p.megapixels(),
            })
        }
        (Some(w), Some(h)) => {
            validate_wh(w, h)?;
            Ok(ResolvedCanvas {
                width: w,
                height: h,
                preset: match_preset(w, h),
                megapixels: f64::from(w) * f64::from(h) / 1_000_000.0,
            })
        }
        _ => bail!("pass both --width and --height, or use --canvas 480p|720p"),
    }
}

/// Resolve upscale target size. 1080p is allowed (and preferred for final deliverables).
pub(crate) fn resolve_upscale_canvas(
    canvas: Option<H3CanvasPreset>,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<Option<ResolvedCanvas>> {
    if let Some(preset) = canvas {
        let (pw, ph) = preset.size();
        if let (Some(w), Some(h)) = (width, height) {
            if w != pw || h != ph {
                bail!(
                    "--canvas {} is {}×{}, but --width/--height are {}×{} — pick one",
                    preset.as_str(),
                    pw,
                    ph,
                    w,
                    h
                );
            }
        } else if width.is_some() ^ height.is_some() {
            bail!("when overriding a canvas preset, pass both --width and --height (or neither)");
        }
        return Ok(Some(ResolvedCanvas {
            width: pw,
            height: ph,
            preset: Some(preset),
            megapixels: preset.megapixels(),
        }));
    }
    match (width, height) {
        (None, None) => Ok(None),
        (Some(w), Some(h)) => {
            if w < 16 || h < 16 {
                bail!("upscale width/height must be >= 16");
            }
            Ok(Some(ResolvedCanvas {
                width: w,
                height: h,
                preset: match_preset(w, h),
                megapixels: f64::from(w) * f64::from(h) / 1_000_000.0,
            }))
        }
        (None, Some(h)) => {
            if h < 16 {
                bail!("upscale height must be >= 16");
            }
            // Height-only: script scales with aspect preserved; width filled later.
            Ok(Some(ResolvedCanvas {
                width: 0,
                height: h,
                preset: None,
                megapixels: 0.0,
            }))
        }
        (Some(w), None) => {
            if w < 16 {
                bail!("upscale width must be >= 16");
            }
            Ok(Some(ResolvedCanvas {
                width: w,
                height: 0,
                preset: None,
                megapixels: 0.0,
            }))
        }
    }
}

fn match_preset(w: u32, h: u32) -> Option<H3CanvasPreset> {
    for p in [
        H3CanvasPreset::Mp02,
        H3CanvasPreset::P480,
        H3CanvasPreset::P720,
        H3CanvasPreset::Mp10,
        H3CanvasPreset::P1080,
    ] {
        if p.size() == (w, h) {
            return Some(p);
        }
    }
    None
}

fn validate_wh(w: u32, h: u32) -> Result<()> {
    if w < 64 || h < 64 {
        bail!("width/height must be >= 64");
    }
    if w % 32 != 0 || h % 32 != 0 {
        bail!("width/height must be multiples of 32 (got {w}x{h})");
    }
    Ok(())
}

/// Open MiniMax-H3 Base area cap (~768×1344). Above this is upscale-only, not a DiT sample.
pub(crate) const OPEN_BASE_MP_CAP: f64 = 1.04;

pub(crate) fn assert_native_dit_area(width: u32, height: u32) -> Result<()> {
    let mp = f64::from(width) * f64::from(height) / 1_000_000.0;
    if mp > OPEN_BASE_MP_CAP {
        bail!(
            "{width}×{height} (~{mp:.2} MP) is above the open Base DiT area cap (~1.03 MP).\n\
             Latent refine is a second H3 sample at the new size, so stay on --canvas 720p or 480p.\n\
             For 1080p pixels use --backend latent-preview or --backend rtx / auto."
        );
    }
    Ok(())
}

/// Latent refine target. Default 720p. 1080p and oversize custom sizes fail closed.
pub(crate) fn resolve_latent_refine_canvas(
    canvas: Option<H3CanvasPreset>,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<ResolvedCanvas> {
    if matches!(canvas, Some(H3CanvasPreset::P1080)) {
        bail!(
            "--canvas 1080p is above the open Base DiT area cap (~1.03 MP).\n\
             Latent refine is a second H3 sample, so stay on --canvas 720p or 480p.\n\
             For 1080p pixels use --backend latent-preview or --backend rtx / auto."
        );
    }
    let canvas = match (canvas, width, height) {
        (None, None, None) => Some(H3CanvasPreset::P720),
        (c, _, _) => c,
    };
    let resolved = resolve_sample_canvas(canvas, width, height)?;
    assert_native_dit_area(resolved.width, resolved.height)?;
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_sizes_are_x32() {
        for p in [
            H3CanvasPreset::Mp02,
            H3CanvasPreset::P480,
            H3CanvasPreset::P720,
            H3CanvasPreset::Mp10,
            H3CanvasPreset::P1080,
        ] {
            let (w, h) = p.size();
            assert_eq!(w % 32, 0, "{p:?} w");
            assert_eq!(h % 32, 0, "{p:?} h");
        }
    }

    #[test]
    fn default_is_480p() {
        let c = resolve_sample_canvas(None, None, None).unwrap();
        assert_eq!((c.width, c.height), (864, 480));
        assert_eq!(c.preset, Some(H3CanvasPreset::P480));
    }

    #[test]
    fn canvas_720p() {
        let c = resolve_sample_canvas(Some(H3CanvasPreset::P720), None, None).unwrap();
        assert_eq!((c.width, c.height), (1280, 736));
    }

    #[test]
    fn canvas_1080p_rejected_for_sample() {
        assert!(resolve_sample_canvas(Some(H3CanvasPreset::P1080), None, None).is_err());
    }

    #[test]
    fn canvas_1080p_ok_for_upscale() {
        let c = resolve_upscale_canvas(Some(H3CanvasPreset::P1080), None, None)
            .unwrap()
            .unwrap();
        assert_eq!((c.width, c.height), (1920, 1088));
    }

    #[test]
    fn latent_refine_defaults_to_720p() {
        let c = resolve_latent_refine_canvas(None, None, None).unwrap();
        assert_eq!((c.width, c.height), (1280, 736));
        assert_eq!(c.preset, Some(H3CanvasPreset::P720));
    }

    #[test]
    fn latent_refine_rejects_1080p() {
        let err = resolve_latent_refine_canvas(Some(H3CanvasPreset::P1080), None, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("latent-preview"), "{err}");
    }

    #[test]
    fn latent_refine_rejects_oversize_custom() {
        assert!(resolve_latent_refine_canvas(None, Some(1920), Some(1088)).is_err());
    }

    #[test]
    fn ganloss_stage1_is_608x352() {
        let c = resolve_sample_canvas(Some(H3CanvasPreset::Mp02), None, None).unwrap();
        assert_eq!((c.width, c.height), (608, 352));
        assert!(c.megapixels < 0.25);
    }

    #[test]
    fn ganloss_stage2_1344x768_is_native_refine() {
        let c = resolve_latent_refine_canvas(Some(H3CanvasPreset::Mp10), None, None).unwrap();
        assert_eq!((c.width, c.height), (1344, 768));
        assert!(c.megapixels < OPEN_BASE_MP_CAP);
    }
}
