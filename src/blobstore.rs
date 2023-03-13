//! The blob storage layer of the filesystem.

mod blocks;
mod inode;
mod superblock;

use core::fmt::Debug;

pub use inode::{Kind, Metadata, Mode, Timestamp};

use self::blocks::{BitmapBlock, FreeNodeBlock, RawBlock};
use self::inode::{INode, INodeBlock};
use self::superblock::Superblock;
use crate::block_device::BlockDevice;

/// Block size used by the file system.
pub const BLOCK_SIZE: u64 = 4096;

/// The blob store that abstracts over a collection of data (the blob).
///
/// In escense, the blob associates a unit of data into a logically contiguous store of bytes.
/// In reality, these bytes may be nothing but contiguous, but this layer abstracts over this.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Blobstore<D: BlockDevice<BLOCK_SIZE>> {
    /// Cached here, but stored in disk at block 0.
    superblock: Superblock,

    /// The actual storage device.
    device: D,
}

/// Error type returned on some of the blobstore operations.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum BlobstoreError {
    /// Returned when the system ran out of INodes and isn't able to create new blobs.
    OutOfInodes,
}

impl<D: BlockDevice<BLOCK_SIZE>> Blobstore<D> {
    /// Constructs the filesystem and returns the `Blobstore`.
    pub fn mkfs(mut device: D) -> Self {
        // First we make a superblock with the device's metadata.
        let superblock = {
            let num_blocks = device.num_blocks();

            // Eventually we actually just pad to utilize the entire block.
            let num_inodes = num_blocks / 10;
            let num_iblocks = (num_inodes - 1) / INodeBlock::inodes_per_block() + 1;

            // One bit per inode.
            let num_bitmap_blocks = (num_inodes - 1) / (BLOCK_SIZE * 8) + 1;

            Superblock::new(num_bitmap_blocks, num_iblocks, device.num_blocks())
        };
        device.write(0, &superblock);

        // Next the bitmap. We just need to zero out the memory.
        for i in superblock.bitmap_head()..superblock.ilist_head() {
            let data_block = BitmapBlock::zeros();
            device.write(i, &data_block);
        }

        // Zero out the inode blocks to guarantee they are valid INodes.
        for i in superblock.ilist_head()..superblock.data_block_head() {
            let block = INodeBlock::empty();
            device.write(i, &block);
        }
        // Now the free list.
        let mut bnum = superblock.free_list_head;
        while bnum != 0 {
            let node = FreeNodeBlock::for_block(bnum, superblock.num_blocks);
            device.write(bnum, &node);
            debug_assert!(node.next_block() == 0 || node.next_block() == bnum + 1024);
            bnum = node.next_block();
        }

        Self { superblock, device }
    }

    /// Opens an existing filesystem originally created with mkfs.
    pub fn openfs(device: D) -> Self {
        // TODO: Use a magic number to check for correctness.
        let superblock: Superblock = device.read_into(0);
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
    pub fn make_blob(&mut self) -> Result<BlobHandle, BlobstoreError> {
        // Find a free INode, the index number of the inode will be the value of the handle.
        for (block_idx, block_num, mut bitmap_block) in (0..self.superblock.num_bitmap_blocks())
            .map(|block_idx| (block_idx, block_idx + self.superblock.bitmap_head()))
            .map(|(block_idx, bnum)| (block_idx, bnum, self.device.read_into::<BitmapBlock>(bnum)))
        {
            let bitmap = bitmap_block.as_bitmap_mut();
            let Some(idx) = bitmap.first_zero() else {
                continue;
            };
            let inum: u64 = idx as u64 + block_idx as u64 * BLOCK_SIZE * 8;
            if inum >= self.superblock.num_inodes() {
                return Err(BlobstoreError::OutOfInodes);
            } else {
                bitmap.set(idx, true);
                self.device.write(block_num, &bitmap_block);
                let handle = BlobHandle(inum);
                // zero out the inode.
                self.write_inode(handle, &INode::empty());
                return Ok(handle);
            }
        }

        Err(BlobstoreError::OutOfInodes)
    }

    /// Writes the data to the specified blob at the given offset.
    ///
    /// The write will copy all the bytes in `data`, possibly extending the length of the blob.
    /// If the offset is past the end of the file, the holes produced by the write will be filled
    /// with 0s.
    pub fn write(
        &mut self,
        handle: BlobHandle,
        mut offset: u64,
        mut data: &[u8],
    ) -> Result<(), ()> {
        let inode = self.read_inode(handle);
        let required_len = offset.checked_add(data.len() as u64).ok_or(())?;
        let new_len = u64::max(required_len, inode.metadata.file_size_bytes);

        if new_len > inode.metadata.file_size_bytes {
            self.resize(handle, new_len)?;
        }

        while !data.is_empty() {
            let block = offset / BLOCK_SIZE;
            let index = (offset % BLOCK_SIZE) as usize;
            let count = usize::min(data.len(), BLOCK_SIZE as usize - index);

            // FIXME: Some unnecessary copies here and bit of clumsy logic.
            let mut datablock = RawBlock::zero();
            datablock.raw_mut()[index..(index + count)].copy_from_slice(&data[..count]);

            let bnum = inode.get_block(block).unwrap();
            self.device.write(bnum, &datablock);

            // go to the start of the next block.
            offset += count as u64;
            data = &data[count..];
        }
        Ok(())
    }

    /// Reads the data starting at `offset` into `out`.
    ///
    /// The read will continue until reaching the length of `out` or until reaching EOF.
    ///
    /// # Returns
    ///
    /// The number of bytes actually read.
    pub fn read(&self, handle: BlobHandle, mut offset: u64, mut out: &mut [u8]) -> Result<u64, ()> {
        let inode = self.read_inode(handle);

        let to_read = u64::min(out.len() as u64, inode.metadata.file_size_bytes - offset);

        while !out.is_empty() {
            let block = offset / BLOCK_SIZE;
            let index = (offset % BLOCK_SIZE) as usize;
            let count = usize::min(out.len(), BLOCK_SIZE as usize - index);

            // FIXME: Some unnecessary copies here and bit of clumsy logic.
            let mut datablock = RawBlock::zero();
            let bnum = inode.get_block(block).unwrap();
            self.device.read(bnum, &mut datablock);

            out[..count].copy_from_slice(&datablock.raw()[index..(index + count)]);

            // go to the start of the next block.
            offset += count as u64;
            out = &mut out[count..];
        }

        Ok(to_read)
    }

    /// Resizes the given blob to the specified length.
    ///
    /// * If the length of the blob is < `new_len`, this is will extend the file with 0s.
    /// * If the length of the blob is > `new_len`, this is will truncate the file.
    pub fn resize(&mut self, handle: BlobHandle, new_len: u64) -> Result<(), ()> {
        let mut inode = self.read_inode(handle);

        let required_blocks = new_len / BLOCK_SIZE;
        let current_block_count = inode.metadata.num_blocks;
        if required_blocks < current_block_count {
            self.free_inode_blocks(&mut inode, current_block_count - required_blocks);
        } else if required_blocks > current_block_count {
            if let Err(_) =
                self.allocate_blocks_to_inode(&mut inode, required_blocks - current_block_count)
            {
                let extra_blocks = inode.metadata.num_blocks - current_block_count;
                self.free_inode_blocks(&mut inode, extra_blocks);
                return Err(());
            }
        }
        assert_eq!(inode.metadata.num_blocks, required_blocks);
        inode.metadata.file_size_bytes = new_len;
        self.write_inode(handle, &inode);
        Ok(())
    }

    fn free_inode_blocks(&mut self, inode: &mut INode, count: u64) {
        todo!();
    }

    fn allocate_blocks_to_inode(&mut self, inode: &mut INode, count: u64) -> Result<(), ()> {
        todo!();
    }

    /// Crecres a hard link to the file.
    pub fn link(&mut self, handle: BlobHandle) -> Result<(), LinkError> {
        let mut inode = self.read_inode(handle);
        inode
            .metadata
            .increment_link()
            .map_err(|_| LinkError::HardLinkLimit)?;

        self.write_inode(handle, &inode);
        Ok(())
    }

    /// Decrements the hard links.
    ///
    /// This function will clean up the blob if no more links exist and
    /// the open count is 0.
    pub fn unlink(&mut self, handle: BlobHandle) -> Result<(), UnlinkError> {
        let mut inode = self.read_inode(handle);
        inode
            .metadata
            .decrement_link()
            .map_err(|_| UnlinkError::NoLinks)?;

        self.write_inode(handle, &inode);

        if inode.metadata.hard_links == 0 && inode.metadata.open_count == 0 {
            self.release_blob(handle);
        }
        Ok(())
    }

    /// Increment the open count.
    pub fn open(&mut self, handle: BlobHandle) -> Result<(), OpenError> {
        let mut inode = self.read_inode(handle);
        inode
            .metadata
            .increment_open()
            .map_err(|_| OpenError::OpenCountLimit)?;

        self.write_inode(handle, &inode);
        Ok(())
    }

    /// Decrement the open count.
    ///
    /// This function will clean up the blob if no more links exist and
    /// the open count is 0.
    pub fn close(&mut self, handle: BlobHandle) -> Result<(), CloseError> {
        let mut inode = self.read_inode(handle);
        inode
            .metadata
            .decrement_open()
            .map_err(|_| CloseError::NotOpened)?;

        self.write_inode(handle, &inode);

        if inode.metadata.hard_links == 0 && inode.metadata.open_count == 0 {
            self.release_blob(handle);
        }
        Ok(())
    }

    /// Returns the metadata associated with a blob.
    pub fn metadata(&self, handle: BlobHandle) -> Metadata {
        let inode = self.read_inode(handle);
        inode.metadata
    }

    /// Reads the INode from disk.
    fn read_inode(&self, handle: BlobHandle) -> INode {
        let (block, index) = self.get_inode_indices(handle);
        // SAFETY: all INode blocks are alwasy valid.
        let inode_block: INodeBlock = unsafe { self.device.read_into_unchecked(block) };
        inode_block[index]
    }

    /// Writes the INode to disk.
    fn write_inode(&mut self, handle: BlobHandle, inode: &INode) {
        let (block, index) = self.get_inode_indices(handle);
        // SAFETY: all INode blocks are alwasy valid.
        let mut inode_block: INodeBlock = unsafe { self.device.read_into_unchecked(block) };
        inode_block[index] = *inode;
        self.device.write(block, &inode_block);
    }

    fn release_blob(&mut self, handle: BlobHandle) {
        // FIXME: We read the INode from disk here but also earlier.
        // release the blocks

        // Free up all of the data in the blob.
        self.resize(handle, 0).unwrap();
        self.deallocate_inode(handle);
    }

    /// Returns the inode block and the block index for the given blob.
    fn get_inode_indices(&self, handle: BlobHandle) -> (u64, u64) {
        let number = handle.0;
        let block = self.superblock.ilist_head() + number / INodeBlock::inodes_per_block();
        let index = number % INodeBlock::inodes_per_block();
        (block, index)
    }

    /// Returns the inode to the system.
    fn deallocate_inode(&mut self, handle: BlobHandle) {
        let inode = self.read_inode(handle);
        assert_eq!(
            inode.metadata.num_blocks, 0,
            "Inode is not ready for deallocation"
        );
        assert_eq!(
            inode.metadata.hard_links, 0,
            "Inode is not ready for deallocation"
        );
        assert_eq!(
            inode.metadata.open_count, 0,
            "Inode is not ready for deallocation"
        );

        let bitnum = self.superblock.bitmap_head() + handle.0 / (BLOCK_SIZE * 8);
        let bitindex = (handle.0 % (BLOCK_SIZE * 8)) as usize;

        let mut bitblock: BitmapBlock = self.device.read_into(bitnum);
        let bitmap = bitblock.as_bitmap_mut();
        bitmap.set(bitindex, false);
        self.device.write(bitnum, &bitblock);
    }
}

/// Errors opening a blob.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum OpenError {
    /// Maximum number of open counts would be exceeded.
    OpenCountLimit,
}

/// Errors hard-linking a blob.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum CloseError {
    /// The blob is not currently open.
    NotOpened,
}

/// Errors hard-linking a blob.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum LinkError {
    /// Maximum number of hard links would be exceeded.
    HardLinkLimit,
}

/// Errors hard-linking a blob.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum UnlinkError {
    /// Link count is already 0.
    NoLinks,
}

/// A unique handle for a blob.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct BlobHandle(u64);

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::inode::INode;
    use super::*;
    use crate::block_device::{BlockDevice, MemoryDevice};

    #[test]
    fn makefs_superblock_is_correct() {
        let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
        let fs = Blobstore::mkfs(device);

        let superblock: Superblock = fs.device.read_into(0);
        // we want 200 / 10 (20 inodes). Each block holds 8 inodes. We need 3 blocks to hold the inodes.
        let iblocks: u64 = (200 / 10 - 1) / INodeBlock::inodes_per_block() + 1;
        assert_eq!(superblock, Superblock::new(1, iblocks, 200));
        assert_eq!(superblock, fs.superblock);
    }

    #[test]
    fn bitmap_is_initialized_correctly() {
        let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
        let fs = Blobstore::mkfs(device);

        for i in 0..fs.superblock.num_bitmap_blocks() {
            let block_num = i + fs.superblock.bitmap_head();
            let block: BitmapBlock = fs.device.read_into(block_num);
            let bitmap = block.as_bitmap();
            assert!(bitmap.first_one().is_none());
        }
    }

    #[test]
    fn makefs_inodes_are_empty() {
        let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
        let fs = Blobstore::mkfs(device);

        let superblock = &fs.superblock;
        for i in superblock.ilist_head()..superblock.data_block_head() {
            // SAFETY: All INodes should always be initialized.
            let iblock: INodeBlock = unsafe { fs.device.read_into_unchecked(i) };
            for inode in &iblock.0 {
                assert_eq!(inode, &INode::empty());
            }
        }
    }

    #[test]
    fn makefs_freelist_is_correct() {
        let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
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
            for num in freenode.raw().iter().copied() {
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

    #[test]
    fn make_blob_test() {
        let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
        let mut fs = Blobstore::mkfs(device);

        let mut allocated_inodes = HashSet::new();
        let num_inodes = fs.superblock.num_iblocks * INodeBlock::inodes_per_block();

        for i in 0..num_inodes {
            let handle = fs
                .make_blob()
                .expect(&format!("Failed blob allocation {i}"));
            println!("Got handle: {handle:?}");
            assert!(
                allocated_inodes.insert(handle.0),
                "Reaused the same INode: {}",
                handle.0
            );
        }

        fs.make_blob().expect_err("Should be out of INodes");
        fs.make_blob().expect_err("Should be out of INodes");
    }
}
