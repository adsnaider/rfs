//! An ext2-like file system.
#![cfg_attr(not(test), no_std)]
#![feature(inline_const)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

pub mod blobstore;
pub mod block_device;
