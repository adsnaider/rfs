//! INodes and metadata.

#![allow(dead_code)]

use core::fmt::Debug;
use core::mem::size_of;
use std::ops::{Index, IndexMut};
use std::time::{SystemTime, UNIX_EPOCH};

use super::BLOCK_SIZE;

const INODE_SIZE: usize = 128;
/// Number of direct blocks.
const NUM_DIRECT: usize = (INODE_SIZE - 3 * 8 - size_of::<Metadata>()) / size_of::<u64>();
/// Number of data blocks handled by indirect blocks.
const NUM_INDIRECT: usize = BLOCK_SIZE as usize / 8;
/// Number of data blocks handled by double indirect blocks.
const NUM_DOUBLE_INDIRECT: usize = (BLOCK_SIZE as usize / 8) * (BLOCK_SIZE as usize / 8);
/// Number of data blocks handled by triple indirect blocks.
const NUM_TRIPLE_INDIRECT: usize =
    (BLOCK_SIZE as usize / 8) * (BLOCK_SIZE as usize / 8) * (BLOCK_SIZE as usize / 8);

/// An INode is a structure used to map files to blocks.
#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct INode {
    /// Direct blocks.
    pub l0_blocks: [u64; NUM_DIRECT],
    /// Indirect block.
    pub l1_block: u64,
    /// Double Indirect block.
    pub l2_block: u64,
    /// Triple Indirect block.
    pub l3_block: u64,

    /// Metadata associated with the inode.
    pub metadata: Metadata,
}

const _INODE_SIZE_IS_CORRECT: () = {
    assert!(size_of::<INode>() == INODE_SIZE);
};

const _ENOUGH_DIRECT_BLOCKS: () = {
    assert!(NUM_DIRECT >= 5);
};

const START_INDIRECT: u64 = NUM_DIRECT as u64;
const START_DOUBLE_INDIRECT: u64 = START_INDIRECT + NUM_INDIRECT as u64;
const START_TRIPLE_INDIRECT: u64 = START_DOUBLE_INDIRECT + NUM_DOUBLE_INDIRECT as u64;

pub enum InodeBlockIndex {
    Direct(u64),
    Indirect(u64),
    DoubleIndirect(u64, u64),
    TripleIndirect(u64, u64, u64),
}

impl INode {
    /// Constructs a new INode with reasonable defaults.
    pub fn new(uid: u16, gid: u16, mode: Mode, kind: Kind) -> Self {
        Self {
            l0_blocks: [0; NUM_DIRECT],
            l1_block: 0,
            l2_block: 0,
            l3_block: 0,
            metadata: Metadata::new(uid, gid, mode, kind),
        }
    }

    /// Constructs an empty (essentially 0) inode.
    pub const fn empty() -> Self {
        Self {
            l0_blocks: [0; NUM_DIRECT],
            l1_block: 0,
            l2_block: 0,
            l3_block: 0,
            metadata: Metadata::empty(),
        }
    }

    /// The maximum file size representable with an INode.
    pub const fn max_capacity() -> u64 {
        NUM_DIRECT as u64 * BLOCK_SIZE
            + (BLOCK_SIZE / 8) * BLOCK_SIZE
            + BLOCK_SIZE * (BLOCK_SIZE / 8) * (BLOCK_SIZE / 8)
            + BLOCK_SIZE * (BLOCK_SIZE / 8) * (BLOCK_SIZE / 8) * (BLOCK_SIZE / 8)
    }

    /// Gets the index into the inode block.
    pub fn get_block_index(&self, index: u64) -> InodeBlockIndex {
        assert!(index < Self::max_capacity());
        match index {
            0..START_INDIRECT => self.get_direct_block(index),
            START_INDIRECT..START_DOUBLE_INDIRECT => self.get_indirect_block(index),
            START_DOUBLE_INDIRECT..START_TRIPLE_INDIRECT => self.get_double_indirect_block(index),
            START_TRIPLE_INDIRECT.. => self.get_triple_indirect_block(index),
        }
    }

    fn get_direct_block(&self, index: u64) -> InodeBlockIndex {
        assert!(index < START_INDIRECT);
        InodeBlockIndex::Direct(index)
    }

    fn get_indirect_block(&self, index: u64) -> InodeBlockIndex {
        assert!(index >= START_INDIRECT && index < START_DOUBLE_INDIRECT);
        let index = index - START_INDIRECT;

        InodeBlockIndex::Indirect(index)
    }

    fn get_double_indirect_block(&self, index: u64) -> InodeBlockIndex {
        assert!(index >= START_DOUBLE_INDIRECT && index < START_TRIPLE_INDIRECT);
        let index = index - START_DOUBLE_INDIRECT;

        let l1 = index / (BLOCK_SIZE / 8);
        let l0 = index % (BLOCK_SIZE / 8);
        InodeBlockIndex::DoubleIndirect(l1, l0)
    }

    fn get_triple_indirect_block(&self, index: u64) -> InodeBlockIndex {
        assert!(index >= START_TRIPLE_INDIRECT);

        let index = index - START_TRIPLE_INDIRECT;
        let l2 = index / ((BLOCK_SIZE / 8) * (BLOCK_SIZE / 8));
        let index = index % ((BLOCK_SIZE / 8) * (BLOCK_SIZE / 8));
        let l1 = index / (BLOCK_SIZE / 8);
        let l0 = index % (BLOCK_SIZE / 8);
        InodeBlockIndex::TripleIndirect(l2, l1, l0)
    }
}

/// The metadata that goes alongside inodes.
#[repr(C)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Metadata {
    /// Size of the blob in bytes.
    pub file_size_bytes: u64,
    /// Number of blocks stored with the file.
    pub num_blocks: u64,
    /// Last time the file was accessed.
    pub access_time: Timestamp,
    /// Last time the inode was modified.
    pub change_time: Timestamp,
    /// Time when the file was created.
    pub creation_time: Timestamp,
    /// Time when the file contents where last changed.
    pub modified_time: Timestamp,
    /// Number of hard links attached to this INode.
    pub hard_links: u32,
    /// Number of processes viewing this file.
    pub open_count: u32,
    /// User owner of the file.
    pub uid: u16,
    /// Group owner of the file.
    pub gid: u16,
    /// Permission bits.
    pub mode: Mode,
    /// Type of file.
    pub kind: Kind,
}

impl Metadata {
    /// Constructs metadata with zeroed out fields.
    pub const fn empty() -> Self {
        Self {
            hard_links: 0,
            file_size_bytes: 0,
            num_blocks: 0,
            open_count: 0,
            creation_time: Timestamp::epoch(),
            change_time: Timestamp::epoch(),
            access_time: Timestamp::epoch(),
            modified_time: Timestamp::epoch(),
            kind: Kind::Regular,
            mode: Mode::restrictive(),
            uid: 0,
            gid: 0,
        }
    }

    /// Constructs the metadata with the provided values.
    pub fn new(uid: u16, gid: u16, mode: Mode, kind: Kind) -> Self {
        let now = Timestamp::now();
        Self {
            hard_links: 0,
            file_size_bytes: 0,
            num_blocks: 0,
            open_count: 0,
            creation_time: now,
            change_time: now,
            access_time: now,
            modified_time: now,
            kind,
            mode,
            uid,
            gid,
        }
    }

    /// Increments the link count.
    pub fn increment_link(&mut self) -> Result<(), ()> {
        match self.hard_links.checked_add(1) {
            Some(links) => {
                self.hard_links = links;
                Ok(())
            }
            None => Err(()),
        }
    }

    /// Decrements the link count.
    pub fn decrement_link(&mut self) -> Result<(), ()> {
        match self.hard_links.checked_sub(1) {
            Some(links) => {
                self.hard_links = links;
                Ok(())
            }
            None => Err(()),
        }
    }

    /// Increments the open count.
    pub fn increment_open(&mut self) -> Result<(), ()> {
        match self.open_count.checked_add(1) {
            Some(count) => {
                self.open_count = count;
                Ok(())
            }
            None => Err(()),
        }
    }

    /// Decrements the open count.
    pub fn decrement_open(&mut self) -> Result<(), ()> {
        match self.open_count.checked_sub(1) {
            Some(count) => {
                self.open_count = count;
                Ok(())
            }
            None => Err(()),
        }
    }
}

/// Permission bits for files.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Mode(u16);

impl Mode {
    /// Construct the mode from the given bits.
    ///
    /// # Panics
    ///
    /// Only the lower 9 bits may be set.
    pub const fn from_bits(bits: u16) -> Self {
        assert!(bits >> 9 == 0, "Only the lower 9 bits in mode are used");
        Self(bits)
    }

    /// Get the permission bits.
    ///
    /// ```
    /// # use rfs::blobstore::Mode;
    /// let mode = Mode::from_bits(0o544);
    /// assert_eq!(mode.bits(), 0o544);
    /// ```
    pub const fn bits(&self) -> u16 {
        self.0
    }

    /// Construct the mode from the given bits.
    ///
    /// # Safety
    ///
    /// Only the lower 9 bits may be set.
    pub const unsafe fn from_bits_unchecked(bits: u16) -> Self {
        Self(bits)
    }

    /// Maximially restrictive mode.
    ///
    /// ```
    /// # use rfs::blobstore::Mode;
    /// let restricted = Mode::restrictive();
    /// assert_eq!(restricted, Mode::from_bits(0o000));
    /// ```
    pub const fn restrictive() -> Mode {
        Mode(0o000)
    }
}

impl std::ops::BitOr for Mode {
    type Output = Mode;

    fn bitor(self, rhs: Self) -> Self::Output {
        // SAFETY: Only the bottom 9 bits may be set in either self or rhs.
        unsafe { Mode::from_bits_unchecked(self.0 | rhs.0) }
    }
}

/// The type of file.
#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Kind {
    /// Regular file.
    Regular,
    /// Directory.
    Directory,
    /// Symbolic link.
    Symlink,
}

/// A timestamp similar to unix's measuring time as seconds since epoch.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Timestamp(u64);
impl Timestamp {
    /// The 0 timestamp.
    ///
    /// This time represents Jan 1, 1970 at 00:00.
    const fn epoch() -> Timestamp {
        Timestamp(0)
    }

    /// Returns a timestamp representing the current time provided by the system.
    pub fn now() -> Self {
        Self(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        )
    }
}

impl Debug for INode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("INode")
            .field("l0_blocks", &self.l0_blocks)
            .field("l1_block", &self.l1_block)
            .field("l2_block", &self.l2_block)
            .field("l3_block", &self.l3_block)
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// A block used to store a collection of INodes.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct INodeBlock(pub [INode; INodeBlock::inodes_per_block() as usize]);

impl Index<u64> for INodeBlock {
    type Output = INode;

    fn index(&self, index: u64) -> &Self::Output {
        &self.0[index as usize]
    }
}

impl IndexMut<u64> for INodeBlock {
    fn index_mut(&mut self, index: u64) -> &mut Self::Output {
        &mut self.0[index as usize]
    }
}

impl INodeBlock {
    /// Returns the number of inodes that fit in a block.
    pub const fn inodes_per_block() -> u64 {
        BLOCK_SIZE / INODE_SIZE as u64
    }

    /// Constructs a new empty INodeBlock.
    pub const fn empty() -> Self {
        Self([INode::empty(); Self::inodes_per_block() as usize])
    }
}
