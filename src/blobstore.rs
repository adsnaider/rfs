//! The blob storage layer of the filesystem.

use core::fmt::Debug;
use core::mem::MaybeUninit;

use self::inode::INodeBlock;
use self::superblock::Superblock;
use crate::block_device::{BlockData, BlockDevice};

mod inode;
mod superblock;

/// The blob store that abstracts over a collection of data (the blob).
///
/// In escense, the blob associates a unit of data into a logically contiguous store of bytes.
/// In reality, these bytes may be nothing but contiguous, but this layer abstracts over this.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Blobstore<D: BlockDevice<4096>> {
    /// Cached here, but stored in disk at block 0.
    superblock: Superblock,

    /// The actual storage device.
    device: D,
}

/// A block that just contains data (say for a blob).
#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq)]
struct DataBlock([u8; 4096]);
// SAFETY: Representation of data block is transparent.
unsafe impl BlockData<4096> for DataBlock {}

impl Debug for DataBlock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f)?;
        writeln!(f, "{: ^192}", "-------------------------------------- DataBlock ----------------------------------------------")?;
        for data in self.0.chunks(64) {
            for byte in data {
                write!(f, "{:02X} ", byte)?;
            }
            writeln!(f)?;
        }
        writeln!(f, "{: ^192}", "-----------------------------------------------------------------------------------------------\n")
    }
}

/// A block that contains addresses of free blocks as well as a chain to the next FreeNodeBlock.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct FreeNodeBlock([u32; 1024]);
// SAFETY: Representation of free node block is transparent.
unsafe impl BlockData<4096> for FreeNodeBlock {}

impl FreeNodeBlock {
    /// Create a FreeNodeBlock that is zeroed out.
    pub fn new() -> Self {
        Self([0; 1024])
    }

    /// Returns the next block in the free list.
    ///
    /// There's nothing special aobut this. It just returns the last element in the block.
    pub fn next_block(&self) -> u32 {
        self.0[1023]
    }
}

impl<D: BlockDevice<4096>> Blobstore<D> {
    /// Constructs the filesystem and returns the `Blobstore`.
    pub fn mkfs(mut device: D) -> Self {
        // First we make a superblock with the device's metadata.
        let superblock = {
            let num_blocks = device.num_blocks();

            // Eventually we actually just pad to utilize the entire block.
            let num_inodes = num_blocks / 10;
            let num_iblocks = (num_inodes - 1) / INodeBlock::inodes_per_block() + 1;

            let ilist_head = 1;
            let free_list_head = ilist_head + num_iblocks;

            Superblock::new(ilist_head, num_iblocks, device.num_blocks(), free_list_head)
        };
        device.write(0, &superblock);

        // Now the ilist which is simply all 0.
        for i in superblock.ilist_head..(superblock.ilist_head + superblock.num_iblocks) {
            let empty_inode = INodeBlock::new();
            device.write(i, &empty_inode);
        }

        // Now the free list. The way this works is that each node in the free list contains the
        // free blocks as a list all the way to the next node (which is represented as the last)
        // element in the INodeList.
        let mut bnum = superblock.free_list_head;
        while bnum != 0 {
            let mut node = FreeNodeBlock::new();
            for (i, elem) in node.0.iter_mut().enumerate() {
                let free_addr = bnum + i as u32 + 1;
                if free_addr >= superblock.num_blocks {
                    break;
                }
                *elem = free_addr;
            }
            device.write(bnum, &node);
            debug_assert!(node.next_block() == 0 || node.next_block() == bnum + 1024);
            bnum = node.next_block();
        }

        Self { superblock, device }
    }

    /// Opens the filesystem (without creating it again)
    pub fn openfs(device: D) -> Self {
        let mut superblock = MaybeUninit::uninit();
        device.read(0, &mut superblock);
        let superblock: Superblock = unsafe { superblock.assume_init() };
        // Though the superblock could logically be incorrect, the data itself will never
        // be unsound.
        assert!(
            superblock.num_blocks == device.num_blocks(),
            "Filesystem doesn't match the device. Maybe call `mkfs` instead?"
        );
        Self { superblock, device }
    }

    /// Create a new blob.
    ///
    /// Notice the absence of a name as names are constructed from directories.
    pub fn make_blob(&mut self) -> BlobHandle {
        todo!();
    }
}

/// A unique handle for a blob.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct BlobHandle(u32);

#[cfg(test)]
mod tests {
    use super::inode::INode;
    use super::*;
    use crate::block_device::{BlockDevice, MemoryDevice};

    #[test]
    fn makefs_superblock_is_correct() {
        let device: MemoryDevice<4096> = MemoryDevice::new(200);
        let fs = Blobstore::mkfs(device);

        let superblock: Superblock = fs.device.read_into(0);
        // we want 200 / 10 (20 inodes). Each block holds 8 inodes. We need 3 blocks to hold the inodes.
        let iblocks = 3;
        assert_eq!(superblock, Superblock::new(1, iblocks, 200, iblocks + 1));
        assert_eq!(superblock, fs.superblock);
    }

    #[test]
    fn makefs_inodes_are_empty() {
        let device: MemoryDevice<4096> = MemoryDevice::new(200);
        let fs = Blobstore::mkfs(device);

        let superblock = &fs.superblock;
        for i in superblock.ilist_head..(superblock.ilist_head + superblock.num_iblocks) {
            let iblock: INodeBlock = fs.device.read_into(i);
            for inode in &iblock.0 {
                assert_eq!(inode, &INode::new());
            }
        }
    }

    #[test]
    fn makefs_freelist_is_correct() {
        let device: MemoryDevice<4096> = MemoryDevice::new(200);
        let fs = Blobstore::mkfs(device);

        let superblock = &fs.superblock;
        let mut bnum = superblock.free_list_head;
        let mut free_blocks = [false; 200];
        // The head of the free list is free.
        free_blocks[bnum as usize] = true;
        while bnum != 0 {
            assert!(
                bnum > 0 && bnum < superblock.num_blocks,
                "Block number {bnum} out of bounds"
            );
            let freenode: FreeNodeBlock = fs.device.read_into(bnum);
            for num in freenode.0 {
                if num != 0 {
                    assert!(!free_blocks[num as usize], "Block already marked as free");
                    free_blocks[num as usize] = true;
                }
            }

            bnum = freenode.next_block();
        }
        for (i, is_free) in free_blocks.iter().enumerate() {
            if i < superblock.free_list_head as usize {
                assert!(
                    !is_free,
                    "Block {i} is marked free when it should be NOT free"
                );
            } else {
                assert!(
                    is_free,
                    "Block {i} is marked not free when it should be free"
                );
            }
        }
    }
}
