use core::fmt::Debug;

use crate::block_device::BlockData;

/// An INode is a structure used to map files to blocks.
#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct INode {
    pub l0_blocks: [u64; 10],
    pub l1_block: u64,
    pub l2_block: u64,
    pub l3_block: u64,

    /// Metadata
    pub hard_links: u32,
    /// TODO: Permissions, file metadata, etc.
    _pad: [u32; 128 - 27],
}

const _INODE_SIZE_IS_512: () = {
    use core::mem::size_of;
    assert!(size_of::<INode>() == 512);
};

impl Debug for INode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("INode")
            .field("l0_blocks", &self.l0_blocks)
            .field("l1_block", &self.l1_block)
            .field("l2_block", &self.l2_block)
            .field("l3_block", &self.l3_block)
            .field("hard_links", &self.hard_links)
            .finish()
    }
}

/// A block used to store a collection of INodes.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct INodeBlock(pub [INode; INodeBlock::inodes_per_block()]);
// SAFETY: Representation of INode should be fully packed and all possible bit combinations should
// be valid.
unsafe impl BlockData<4096> for INodeBlock {}

impl INode {
    /// Constructs a new empty INode.
    pub const fn new() -> Self {
        Self {
            l0_blocks: [0; 10],
            l1_block: 0,
            l2_block: 0,
            l3_block: 0,
            hard_links: 0,
            _pad: [0; 128 - 27],
        }
    }
}

impl INodeBlock {
    pub const fn inodes_per_block() -> usize {
        4096 / 512
    }

    /// Constructs a new empty INode block.
    pub const fn new() -> Self {
        Self([INode::new(); Self::inodes_per_block()])
    }
}
