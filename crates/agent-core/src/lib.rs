//! Deterministic local coding-agent core.
//!
//! This crate intentionally has no Tauri dependency. The application host
//! supplies adapters for llama.cpp, CodeGraph, filesystem reads, and tracing.

mod knowledge;
mod notebook;
mod pipeline;
mod prompt;
mod trace;
mod traits;
mod types;

pub use knowledge::*;
pub use notebook::*;
pub use pipeline::*;
pub use prompt::*;
pub use trace::*;
pub use traits::*;
pub use types::*;
