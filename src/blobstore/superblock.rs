use core::fmt::Debug;

use crate::block_device::BlockData;

/// The superblock is a special block located at the beginning of the block device.
///
/// The superblock is the first block in the filesystem and it stores information needed to
/// understand the structure of the filesystema t boot.
#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Superblock {
    pub ilist_head: u32,
    pub num_iblocks: u32,
    pub num_blocks: u32,
    pub free_list_head: u32,

    _pad: [u32; 1024 - 4],
}

impl Debug for Superblock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Superblock")
            .field("ilist_head", &self.ilist_head)
            .field("num_iblocks", &self.num_iblocks)
            .field("num_blocks", &self.num_blocks)
            .field("free_list_head", &self.free_list_head)
            .finish()
    }
}

const _SUPERBLOCK_SIZE_IS_4096: () = {
    use core::mem::size_of;
    assert!(size_of::<Superblock>() == 4096);
};
unsafe impl BlockData<4096> for Superblock {}

impl Superblock {
    pub fn new(ilist_head: u32, num_iblocks: u32, num_blocks: u32, free_list_head: u32) -> Self {
        Self {
            ilist_head,
            num_iblocks,
            num_blocks,
            free_list_head,
            _pad: [0; 1024 - 4],
        }
    }
}
