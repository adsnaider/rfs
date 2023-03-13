//! An ext2-like file system.
#![feature(inline_const)]
#![warn(missing_docs)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(missing_debug_implementations)]

pub mod blobstore;
pub mod block_device;
