//! `h3` — MiniMax-H3 production video engine.
//!
//! Orchestrates the internalized MiniMax-H3 stack under `runtimes/minimax-h3`
//! (TE → DiT sample → VAE decode, multi-window continue, post-decode upscale).
//! Multi-GB checkpoints stay external. Production DiT is pruned int8 ConvRot.

mod args;
mod canvas;
mod continue_cmd;
mod doctor;
mod download;
mod edit;
mod face_refine;
mod generate;
mod install;
mod interpolate;
mod loop_cmd;
mod models;
pub(crate) mod paths;
mod scene_ir;
mod setup;
mod shortfilm;
mod refmod;
mod sprites;
mod upscale;

pub(crate) use args::{
    H3Args, H3Command, H3EditArgs, H3GenerateArgs, H3InterpolateArgs, H3LoopArgs, H3RefModArgs,
    H3ShortfilmArgs,
};

use crate::host::config::H3Config;
use anyhow::Result;

pub(crate) fn run_h3(args: H3Args, config: &H3Config) -> Result<()> {
    paths::apply_roots_from_config(config);
    match args.command {
        H3Command::Install(a) => install::run_install(a, config),
        H3Command::Setup(a) => setup::run_setup(a, config),
        H3Command::Download(a) => download::run_download(a),
        H3Command::Verify(a) => install::run_verify(a, config),
        H3Command::Doctor(a) => doctor::run_doctor(a, config),
        H3Command::Generate(a) => generate::run_generate(a, config),
        H3Command::Continue(a) => continue_cmd::run_continue(a, config),
        H3Command::Loop(a) => loop_cmd::run_loop(a, config),
        H3Command::Upscale(a) => upscale::run_upscale(a, config),
        H3Command::Interpolate(a) => interpolate::run_interpolate(a, config),
        H3Command::FaceRefine(a) => face_refine::run_face_refine(a, config),
        H3Command::Sprites(a) => sprites::run_sprites(a, config),
        H3Command::Shortfilm(a) => shortfilm::run_shortfilm(a, config),
        H3Command::Edit(a) => edit::run_edit(a, config),
        H3Command::Refmod(a) => refmod::run_refmod(a, config),
    }
}

pub(crate) fn run_refmod(args: H3RefModArgs, config: &H3Config) -> Result<()> {
    paths::apply_roots_from_config(config);
    refmod::run_refmod(args, config)
}

pub(crate) fn run_shortfilm(args: H3ShortfilmArgs, config: &H3Config) -> Result<()> {
    paths::apply_roots_from_config(config);
    shortfilm::run_shortfilm(args, config)
}

pub(crate) fn run_loop(args: H3LoopArgs, config: &H3Config) -> Result<()> {
    paths::apply_roots_from_config(config);
    loop_cmd::run_loop(args, config)
}

pub(crate) fn run_edit(args: H3EditArgs, config: &H3Config) -> Result<()> {
    paths::apply_roots_from_config(config);
    edit::run_edit(args, config)
}

pub(crate) fn run_interpolate(args: H3InterpolateArgs, config: &H3Config) -> Result<()> {
    paths::apply_roots_from_config(config);
    interpolate::run_interpolate(args, config)
}

/// Flat alias entrypoints: `gemmy video-h3` / `gemmy h3` → generate.
pub(crate) fn run_h3_generate(args: H3GenerateArgs, config: &H3Config) -> Result<()> {
    paths::apply_roots_from_config(config);
    generate::run_generate(args, config)
}
