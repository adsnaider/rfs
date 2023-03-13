//! Superblock data.
use core::fmt::Debug;

use super::inode::INodeBlock;
use crate::blobstore::BLOCK_SIZE;
use crate::block_device::BlockData;

/// The superblock is a special block located at the beginning of the block device.
///
/// The superblock is the first block in the filesystem and it stores information needed to
/// understand the structure of the filesystema t boot.
#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Superblock {
    /// Number of bitmap blocks.
    pub num_bitmap_blocks: u64,
    /// Number of blocks used for the Ilist.
    pub num_iblocks: u64,
    /// Number of blocks in the device.
    pub num_blocks: u64,
    /// Head of the free block list.
    pub free_list_head: u64,

    /// Padding to make fit in a block.
    _pad: [u64; BLOCK_SIZE as usize / 8 - 4],
}

impl Debug for Superblock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Superblock")
            .field("num_bitmap_blocks", &self.num_bitmap_blocks)
            .field("num_iblocks", &self.num_iblocks)
            .field("num_blocks", &self.num_blocks)
            .field("free_list_head", &self.free_list_head)
            .finish()
    }
}

const _SUPERBLOCK_SIZE_IS_BLOCK_SIZE: () = {
    use core::mem::size_of;
    assert!(size_of::<Superblock>() == BLOCK_SIZE as usize);
};
// SAFETY: All bit patterns for superblock are valid as it only contains numbers.
unsafe impl BlockData<BLOCK_SIZE> for Superblock {}

impl Superblock {
    /// Constructs a new superblock.
    pub fn new(num_bitmap_blocks: u64, num_iblocks: u64, num_blocks: u64) -> Self {
        Self {
            num_bitmap_blocks,
            num_iblocks,
            num_blocks,
            free_list_head: 1 + num_bitmap_blocks + num_iblocks,
            _pad: [0; 512 - 4],
        }
    }

    /// Returns the start of the bitmap blocks.
    pub const fn bitmap_head(&self) -> u64 {
        1
    }

    /// Returns the start of the ilist head.
    pub fn ilist_head(&self) -> u64 {
        1 + self.num_bitmap_blocks
    }

    /// Returns the start of the data blocks.
    pub fn data_block_head(&self) -> u64 {
        1 + self.num_bitmap_blocks + self.num_iblocks
    }

    pub fn num_inodes(&self) -> u64 {
        self.num_iblocks * INodeBlock::inodes_per_block() as u64
    }

    pub fn num_bitmap_blocks(&self) -> u64 {
        self.num_bitmap_blocks
    }
}
