use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum OutputKind {
    File,
    NonEmptyFile,
    Json,
    Image,
    Video,
    Audio,
    Directory,
}

#[derive(Clone, Debug)]
pub struct OutputExpectation {
    path: PathBuf,
    kind: OutputKind,
    label: Option<String>,
    required: bool,
}

impl OutputExpectation {
    #[allow(dead_code)]
    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self::new(path, OutputKind::File)
    }

    #[allow(dead_code)]
    pub fn non_empty_file(path: impl Into<PathBuf>) -> Self {
        Self::new(path, OutputKind::NonEmptyFile)
    }

    pub fn json(path: impl Into<PathBuf>) -> Self {
        Self::new(path, OutputKind::Json)
    }

    pub fn image(path: impl Into<PathBuf>) -> Self {
        Self::new(path, OutputKind::Image)
    }

    pub fn video(path: impl Into<PathBuf>) -> Self {
        Self::new(path, OutputKind::Video)
    }

    pub fn audio(path: impl Into<PathBuf>) -> Self {
        Self::new(path, OutputKind::Audio)
    }

    #[allow(dead_code)]
    pub fn directory(path: impl Into<PathBuf>) -> Self {
        Self::new(path, OutputKind::Directory)
    }

    pub fn labeled(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    #[allow(dead_code)]
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    fn new(path: impl Into<PathBuf>, kind: OutputKind) -> Self {
        Self {
            path: path.into(),
            kind,
            label: None,
            required: true,
        }
    }
}

pub fn validate_outputs(outputs: &[OutputExpectation]) -> Result<()> {
    for output in outputs {
        validate_output(output)?;
    }
    Ok(())
}

pub fn validate_output(output: &OutputExpectation) -> Result<()> {
    if !output.path.exists() {
        if output.required {
            bail!(
                "{} was not written: {}",
                output.display_label(),
                output.path.display()
            );
        }
        return Ok(());
    }

    match output.kind {
        OutputKind::Directory => validate_directory(output),
        OutputKind::File => validate_file(output),
        OutputKind::NonEmptyFile => validate_non_empty_file(output),
        OutputKind::Json => validate_json_file(output),
        OutputKind::Image => validate_image_file(output),
        OutputKind::Video => validate_video_file(output),
        OutputKind::Audio => validate_audio_file(output),
    }
}

impl OutputExpectation {
    fn display_label(&self) -> &str {
        self.label.as_deref().unwrap_or(match self.kind {
            OutputKind::File => "output file",
            OutputKind::NonEmptyFile => "non-empty output file",
            OutputKind::Json => "JSON output",
            OutputKind::Image => "image output",
            OutputKind::Video => "video output",
            OutputKind::Audio => "audio output",
            OutputKind::Directory => "output directory",
        })
    }
}

fn validate_directory(output: &OutputExpectation) -> Result<()> {
    if !output.path.is_dir() {
        bail!(
            "{} is not a directory: {}",
            output.display_label(),
            output.path.display()
        );
    }
    Ok(())
}

fn validate_file(output: &OutputExpectation) -> Result<()> {
    if !output.path.is_file() {
        bail!(
            "{} is not a file: {}",
            output.display_label(),
            output.path.display()
        );
    }
    Ok(())
}

fn validate_non_empty_file(output: &OutputExpectation) -> Result<()> {
    validate_file(output)?;
    let size = output
        .path
        .metadata()
        .with_context(|| format!("Failed to inspect {}", output.path.display()))?
        .len();
    if size == 0 {
        bail!(
            "{} is empty: {}",
            output.display_label(),
            output.path.display()
        );
    }
    Ok(())
}

fn validate_json_file(output: &OutputExpectation) -> Result<()> {
    validate_non_empty_file(output)?;
    let bytes = std::fs::read(&output.path)
        .with_context(|| format!("Failed to read {}", output.path.display()))?;
    let _: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("{} is invalid JSON", output.path.display()))?;
    Ok(())
}

fn validate_image_file(output: &OutputExpectation) -> Result<()> {
    validate_non_empty_file(output)?;
    let bytes = read_prefix(&output.path, 16)?;
    if is_png(&bytes) || is_jpeg(&bytes) || is_webp(&bytes) || is_gif(&bytes) || is_bmp(&bytes) {
        return Ok(());
    }
    bail!(
        "{} does not look like PNG, JPEG, WebP, GIF, or BMP: {}",
        output.display_label(),
        output.path.display()
    );
}

fn validate_video_file(output: &OutputExpectation) -> Result<()> {
    validate_non_empty_file(output)?;
    let bytes = read_prefix(&output.path, 64)?;
    if is_mp4_like(&bytes) || is_ebml(&bytes) || is_avi(&bytes) {
        return Ok(());
    }
    bail!(
        "{} does not look like MP4/MOV, WebM/MKV, or AVI: {}",
        output.display_label(),
        output.path.display()
    );
}

fn validate_audio_file(output: &OutputExpectation) -> Result<()> {
    validate_non_empty_file(output)?;
    let bytes = read_prefix(&output.path, 64)?;
    if is_wav(&bytes) || is_flac(&bytes) || is_ogg(&bytes) || is_mp3(&bytes) || is_mp4_like(&bytes)
    {
        return Ok(());
    }
    bail!(
        "{} does not look like WAV, FLAC, Ogg, MP3, or M4A: {}",
        output.display_label(),
        output.path.display()
    );
}

fn read_prefix(path: &Path, max_bytes: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open {}", path.display()))?;
    let mut bytes = vec![0u8; max_bytes];
    let read = file
        .read(&mut bytes)
        .with_context(|| format!("Failed to read prefix of {}", path.display()))?;
    bytes.truncate(read);
    Ok(bytes)
}

fn is_png(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\x89PNG\r\n\x1a\n")
}

fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xff, 0xd8, 0xff])
}

fn is_webp(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP"
}

fn is_gif(bytes: &[u8]) -> bool {
    bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")
}

fn is_bmp(bytes: &[u8]) -> bool {
    bytes.starts_with(b"BM")
}

fn is_mp4_like(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[4..8] == b"ftyp"
}

fn is_ebml(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3])
}

fn is_avi(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"AVI "
}

fn is_wav(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WAVE"
}

fn is_flac(bytes: &[u8]) -> bool {
    bytes.starts_with(b"fLaC")
}

fn is_ogg(bytes: &[u8]) -> bool {
    bytes.starts_with(b"OggS")
}

fn is_mp3(bytes: &[u8]) -> bool {
    bytes.starts_with(b"ID3")
        || matches!(
            bytes,
            [0xff, 0xfb, ..] | [0xff, 0xf3, ..] | [0xff, 0xf2, ..]
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(name: &str, bytes: &[u8]) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        (dir, path)
    }

    #[test]
    fn validates_json_output() {
        let (_dir, path) = temp_file("result.json", br#"{"ok":true}"#);
        validate_output(&OutputExpectation::json(path)).unwrap();
    }

    #[test]
    fn rejects_invalid_json_output() {
        let (_dir, path) = temp_file("result.json", b"not json");
        assert!(validate_output(&OutputExpectation::json(path)).is_err());
    }

    #[test]
    fn validates_common_media_signatures() {
        let (_png_dir, png) = temp_file("image.png", b"\x89PNG\r\n\x1a\nrest");
        validate_output(&OutputExpectation::image(png)).unwrap();

        let (_mp4_dir, mp4) = temp_file("video.mp4", b"\0\0\0\x18ftypmp42rest");
        validate_output(&OutputExpectation::video(mp4)).unwrap();

        let (_wav_dir, wav) = temp_file("audio.wav", b"RIFF\x24\0\0\0WAVEfmt rest");
        validate_output(&OutputExpectation::audio(wav)).unwrap();
    }

    #[test]
    fn rejects_wrong_media_signature() {
        let (_dir, path) = temp_file("video.mp4", b"plain text");
        assert!(validate_output(&OutputExpectation::video(path)).is_err());
    }

    #[test]
    fn optional_missing_output_is_allowed() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.json");
        validate_output(&OutputExpectation::json(missing).optional()).unwrap();
    }

    #[test]
    fn required_missing_output_fails() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.json");
        assert!(validate_output(&OutputExpectation::json(missing)).is_err());
    }
}
