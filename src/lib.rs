//! An ext2-like file system.
#![feature(inline_const)]
#![feature(exclusive_range_pattern)]
#![warn(missing_docs)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(missing_debug_implementations)]

pub mod blobstore;
pub mod block_device;
pub mod fstore;

#[cfg(test)]
mod tests {
    pub(crate) fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }
}
