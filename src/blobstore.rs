//! The blob storage layer of the filesystem.

mod blocks;
mod inode;
mod superblock;

use core::fmt::Debug;
use core::num::NonZeroU64;

pub use inode::{Kind, Metadata, Mode, Timestamp};
use thiserror::Error;

use self::blocks::{AddrBlock, BitmapBlock, FreeNodeBlock, RawBlock};
use self::inode::{INode, INodeBlock, InodeBlockIndex};
use self::superblock::Superblock;
use crate::block_device::BlockDevice;

/// Block size used by the file system.
pub const BLOCK_SIZE: u64 = 4096;

fn bytes_to_blocks(bytes: u64) -> u64 {
    if bytes == 0 {
        return 0;
    }
    (bytes - 1) / BLOCK_SIZE + 1
}

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

impl<D: BlockDevice<BLOCK_SIZE>> Blobstore<D> {
    /// Constructs the filesystem and returns the `Blobstore`.
    pub fn mkfs(mut device: D) -> Self {
        // First we make a superblock with the device's metadata.
        let superblock = {
            let num_blocks = device.num_blocks();

            // We pad to utilize the entire block.
            let num_inodes = num_blocks / 10;
            let num_iblocks = (num_inodes - 1) / INodeBlock::inodes_per_block() + 1;

            // One bit per inode.
            let num_bitmap_blocks = (num_inodes - 1) / (BLOCK_SIZE * 8) + 1;

            Superblock::new(num_bitmap_blocks, num_iblocks, num_blocks)
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
        debug_assert!(bnum.is_some());
        while let Some(block) = bnum {
            let node = FreeNodeBlock::for_block(block.get(), superblock.num_blocks);
            device.write(block.get(), &node);
            bnum = node.next_block();
        }

        Self { superblock, device }
    }

    /// Opens an existing filesystem originally created with mkfs.
    pub fn openfs(device: D) -> Result<Self, FileSystemError> {
        // TODO: Use a magic number to check for correctness.
        let superblock: Superblock = device.read_into(0);
        // Though the superblock could logically be incorrect, the data itself will never
        // be unsound.
        if superblock.num_blocks != device.num_blocks() {
            return Err(FileSystemError::InvalidFilesystem);
        }
        Ok(Self { superblock, device })
    }

    /// Create a new blob.
    ///
    /// Notice the absence of a name as names are constructed from directories.
    pub fn make_blob(
        &mut self,
        uid: u16,
        gid: u16,
        mode: Mode,
        kind: Kind,
    ) -> Result<BlobHandle, BlobstoreError> {
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
                self.write_inode(handle, &INode::new(uid, gid, mode, kind));
                self.superblock.num_free_inodes -= 1;
                self.device.write(0, &self.superblock);
                return Ok(handle);
            }
        }

        Err(BlobstoreError::OutOfInodes)
    }

    /// Writes the contents of `data` into the block.
    ///
    /// The write will begin at `block_offset` and will end at the end of the block or the
    /// end of the `data` slice. Any "untouched" bytes are guaranteed to keep their original
    /// data.
    ///
    /// # Returns
    ///
    /// The number of bytes actually written to the block.
    fn write_to_block(&mut self, data: &[u8], block_num: u64, block_offset: u64) -> u64 {
        let block_offset = block_offset as usize;
        let count = usize::min(BLOCK_SIZE as usize - block_offset, data.len());
        if count == BLOCK_SIZE as usize {
            debug_assert!(block_offset == 0);
            debug_assert!(data.len() >= BLOCK_SIZE as usize);
            unsafe { self.device.write_slice_unchecked(block_num, data) }
        } else {
            let mut datablock: RawBlock = self.device.read_into(block_num);
            datablock.raw_mut()[block_offset..(block_offset + count)]
                .copy_from_slice(&data[..count]);
            self.device.write(block_num, &datablock);
        }

        count as u64
    }

    /// Reads the contents of the block into out.
    ///
    /// The read will begin at `block_offset` and will end at the end of the block or the
    /// end of the `data` slice.
    ///
    /// # Returns
    ///
    /// The number of bytes actually read.
    fn read_from_block(&self, out: &mut [u8], block_num: u64, block_offset: u64) -> u64 {
        let block_offset = block_offset as usize;
        let count = usize::min(BLOCK_SIZE as usize - block_offset, out.len());
        if count == BLOCK_SIZE as usize {
            debug_assert!(block_offset == 0);
            debug_assert!(out.len() >= BLOCK_SIZE as usize);
            unsafe { self.device.read_slice_unchecked(block_num, out) }
        } else {
            let datablock: RawBlock = self.device.read_into(block_num);
            out[..count].copy_from_slice(&datablock.raw()[block_offset..(block_offset + count)]);
        }

        count as u64
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
    ) -> Result<(), WriteError> {
        // FIXME: Incredibly awkward transition with resize.
        let inode = self.read_inode(handle);
        let required_len = offset
            .checked_add(data.len() as u64)
            .ok_or(WriteError::OffsetOverflow)?;
        let new_len = u64::max(required_len, inode.metadata.file_size_bytes);
        self.resize(handle, new_len)?;

        // Inode may have changed after resize.
        let inode = self.read_inode(handle);

        while !data.is_empty() {
            let block = offset / BLOCK_SIZE;
            let block_offset = offset % BLOCK_SIZE;
            let bnum = self.get_block_in_inode(&inode, block).unwrap();

            let written = self.write_to_block(data, bnum, block_offset);

            // go to the start of the next block.
            offset += written;
            data = &data[written as usize..];
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
    pub fn read(&mut self, handle: BlobHandle, mut offset: u64, mut out: &mut [u8]) -> u64 {
        let inode = self.read_inode(handle);

        let to_read = u64::min(out.len() as u64, inode.metadata.file_size_bytes - offset);
        let mut read_count = 0u64;

        while read_count < to_read {
            let block = offset / BLOCK_SIZE;
            let block_offset = offset % BLOCK_SIZE;
            let bnum = self.get_block_in_inode(&inode, block).unwrap();

            let count = self.read_from_block(out, bnum, block_offset);

            offset += count;
            read_count += count;
            out = &mut out[count as usize..];
        }
        to_read
    }

    /// Resizes the given blob to the specified length.
    ///
    /// * If the length of the blob is < `new_len`, this is will extend the file with 0s.
    /// * If the length of the blob is > `new_len`, this is will truncate the file.
    pub fn resize(&mut self, handle: BlobHandle, new_len: u64) -> Result<(), ResizeError> {
        let mut inode = self.read_inode(handle);

        let required_blocks = bytes_to_blocks(new_len);
        let current_block_count = inode.metadata.num_blocks;
        if required_blocks < current_block_count {
            self.deallocate_blocks_from_inode(&mut inode, current_block_count - required_blocks);
        } else if required_blocks > current_block_count {
            self.allocate_blocks_to_inode(&mut inode, required_blocks - current_block_count)?
        }
        assert_eq!(inode.metadata.num_blocks, required_blocks);
        inode.metadata.file_size_bytes = new_len;
        self.write_inode(handle, &inode);
        Ok(())
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

    /// Get the total number of blocks in the device.
    pub fn get_num_blocks(&self) -> u64 {
        self.superblock.num_blocks
    }

    /// Get the number of free blocks in the device.
    pub fn get_num_free_blocks(&self) -> u64 {
        self.superblock.num_free_blocks
    }

    /// Returns the total number of INodes.
    pub fn get_num_inodes(&self) -> u64 {
        self.superblock.num_inodes()
    }

    /// Returns the number of unallocated INodes.
    pub fn get_num_free_inodes(&self) -> u64 {
        self.superblock.num_free_inodes()
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
        self.superblock.num_free_inodes += 1;
        self.device.write(0, &self.superblock);
    }

    /// Allocates a block from the system.
    fn allocate_block(&mut self) -> Option<u64> {
        let head = self.superblock.free_list_head?.get();
        let mut free_list: FreeNodeBlock = self.device.read_into(head);

        if let Some(block) = free_list.take_next_available() {
            self.device.write(head, &free_list);
            self.device.write(block, &RawBlock::zero());
            self.superblock.num_free_blocks -= 1;
            self.device.write(0, &self.superblock);
            return Some(block);
        }

        // We need to use the free list head and set the new free list head.
        self.superblock.free_list_head = free_list.next_block();
        self.device.write(0, &self.superblock);
        self.device.write(head, &RawBlock::zero());
        self.superblock.num_free_blocks -= 1;
        self.device.write(0, &self.superblock);
        Some(head)
    }

    /// Deallocates a block from the system.
    fn deallocate_block(&mut self, bnum: u64) {
        self.superblock.num_free_blocks += 1;
        self.device.write(0, &self.superblock);

        let Some(head) = self.superblock.free_list_head else {
            self.device.write(bnum, &FreeNodeBlock::empty(0));
            self.superblock.free_list_head = Some(NonZeroU64::new(bnum).unwrap());
            self.device.write(0, &self.superblock);
            return;
        };

        let head = head.get();
        let mut free_node: FreeNodeBlock = self.device.read_into(head);
        if let Ok(()) = free_node.set_empty_slot(bnum) {
            // Found an empty slot for the block.
            self.device.write(head, &free_node);
            return;
        }

        // Head of free list is full, make a new head with the block `bnum`.
        let new_head = FreeNodeBlock::empty(head);
        self.device.write(bnum, &new_head);
        self.superblock.free_list_head = Some(NonZeroU64::new(bnum).unwrap());
        self.device.write(0, &self.superblock);
    }

    fn allocate_blocks_to_inode(
        &mut self,
        inode: &mut INode,
        count: u64,
    ) -> Result<(), ResizeError> {
        for i in 0..count {
            if let Err(e) = self.allocate_block_to_inode(inode) {
                self.deallocate_blocks_from_inode(inode, i);
                return Err(e);
            }
        }
        Ok(())
    }

    fn deallocate_blocks_from_inode(&mut self, inode: &mut INode, count: u64) {
        assert!(inode.metadata.num_blocks >= count);
        for _ in 0..count {
            self.deallocate_block_from_inode(inode);
        }
    }

    fn allocate_block_to_inode(&mut self, inode: &mut INode) -> Result<(), ResizeError> {
        if inode.metadata.num_blocks >= INode::max_capacity() {
            return Err(ResizeError::MaxCapacityReached);
        }

        let next_block = inode.metadata.num_blocks;
        match inode.get_block_index(next_block) {
            InodeBlockIndex::Direct(idx) => {
                let block = self.allocate_block().ok_or(ResizeError::OutOfBlocks)?;
                inode.l0_blocks[idx as usize] = block;
            }
            InodeBlockIndex::Indirect(l1_idx) => {
                if l1_idx == 0 {
                    if self.get_num_free_blocks() < 2 {
                        return Err(ResizeError::OutOfBlocks);
                    }
                    let block = self.allocate_block().ok_or(ResizeError::OutOfBlocks)?;
                    inode.l1_block = block;
                }
                let mut l1: AddrBlock = self.device.read_into(inode.l1_block);
                let block = self.allocate_block().ok_or(ResizeError::OutOfBlocks)?;
                l1[l1_idx] = block;
                self.device.write(inode.l1_block, &l1);
            }
            InodeBlockIndex::DoubleIndirect(l2_idx, l1_idx) => {
                if l2_idx == 0 && l1_idx == 0 {
                    if self.get_num_free_blocks() < 3 {
                        return Err(ResizeError::OutOfBlocks);
                    }
                    let block = self.allocate_block().ok_or(ResizeError::OutOfBlocks)?;
                    inode.l2_block = block;
                }
                let mut l2: AddrBlock = self.device.read_into(inode.l2_block);
                if l1_idx == 0 {
                    if self.get_num_free_blocks() < 2 {
                        return Err(ResizeError::OutOfBlocks);
                    }
                    let block = self.allocate_block().ok_or(ResizeError::OutOfBlocks)?;
                    l2[l2_idx] = block;
                    self.device.write(inode.l2_block, &l2);
                }
                let mut l1: AddrBlock = self.device.read_into(l2[l2_idx]);
                let block = self.allocate_block().ok_or(ResizeError::OutOfBlocks)?;
                l1[l1_idx] = block;
                self.device.write(l2[l2_idx], &l1);
            }
            InodeBlockIndex::TripleIndirect(l3_idx, l2_idx, l1_idx) => {
                if l3_idx == 0 && l2_idx == 0 && l1_idx == 0 {
                    if self.get_num_free_blocks() < 4 {
                        return Err(ResizeError::OutOfBlocks);
                    }
                    let block = self.allocate_block().ok_or(ResizeError::OutOfBlocks)?;
                    inode.l3_block = block;
                }
                let mut l3: AddrBlock = self.device.read_into(inode.l3_block);
                if l2_idx == 0 && l1_idx == 0 {
                    if self.get_num_free_blocks() < 3 {
                        return Err(ResizeError::OutOfBlocks);
                    }
                    let block = self.allocate_block().ok_or(ResizeError::OutOfBlocks)?;
                    l3[l3_idx] = block;
                    self.device.write(inode.l3_block, &l3);
                }
                let mut l2: AddrBlock = self.device.read_into(l3[l3_idx]);
                if l1_idx == 0 {
                    if self.get_num_free_blocks() < 2 {
                        return Err(ResizeError::OutOfBlocks);
                    }
                    let block = self.allocate_block().ok_or(ResizeError::OutOfBlocks)?;
                    l2[l2_idx] = block;
                    self.device.write(l3[l3_idx], &l2);
                }
                let mut l1: AddrBlock = self.device.read_into(l2[l2_idx]);
                let block = self.allocate_block().ok_or(ResizeError::OutOfBlocks)?;
                l1[l1_idx] = block;
                self.device.write(l2[l2_idx], &l1);
            }
        }
        inode.metadata.num_blocks += 1;
        Ok(())
    }

    fn deallocate_block_from_inode(&mut self, inode: &mut INode) {
        assert!(inode.metadata.num_blocks > 0);
        let last_block = inode.metadata.num_blocks - 1;
        match inode.get_block_index(last_block) {
            InodeBlockIndex::Direct(idx) => {
                self.deallocate_block(inode.l0_blocks[idx as usize]);
            }
            InodeBlockIndex::Indirect(l1_idx) => {
                let l1: AddrBlock = self.device.read_into(inode.l1_block);

                self.deallocate_block(l1[l1_idx]);
                if l1_idx == 0 {
                    self.deallocate_block(inode.l1_block);
                }
            }
            InodeBlockIndex::DoubleIndirect(l2_idx, l1_idx) => {
                let l2: AddrBlock = self.device.read_into(inode.l2_block);
                let l1: AddrBlock = self.device.read_into(l2[l2_idx]);

                self.deallocate_block(l1[l1_idx]);
                if l1_idx == 0 {
                    self.deallocate_block(l2[l2_idx]);
                    if l2_idx == 0 {
                        self.deallocate_block(inode.l2_block);
                    }
                }
            }
            InodeBlockIndex::TripleIndirect(l3_idx, l2_idx, l1_idx) => {
                let l3: AddrBlock = self.device.read_into(inode.l3_block);
                let l2: AddrBlock = self.device.read_into(l3[l3_idx]);
                let l1: AddrBlock = self.device.read_into(l2[l2_idx]);

                self.deallocate_block(l1[l1_idx]);
                if l1_idx == 0 {
                    self.deallocate_block(l2[l2_idx]);
                    if l2_idx == 0 {
                        self.deallocate_block(l3[l3_idx]);
                        if l3_idx == 0 {
                            self.deallocate_block(inode.l3_block);
                        }
                    }
                }
            }
        }
        inode.metadata.num_blocks -= 1;
    }

    fn get_block_in_inode(&self, inode: &INode, index: u64) -> Option<u64> {
        if index >= inode.metadata.num_blocks {
            return None;
        }
        Some(match inode.get_block_index(index) {
            InodeBlockIndex::Direct(idx) => inode.l0_blocks[idx as usize],
            InodeBlockIndex::Indirect(l1_idx) => {
                let l1: AddrBlock = self.device.read_into(inode.l1_block);
                l1[l1_idx]
            }
            InodeBlockIndex::DoubleIndirect(l2_idx, l1_idx) => {
                let l2: AddrBlock = self.device.read_into(inode.l2_block);
                let l1: AddrBlock = self.device.read_into(l2[l2_idx]);
                l1[l1_idx]
            }
            InodeBlockIndex::TripleIndirect(l3_idx, l2_idx, l1_idx) => {
                let l3: AddrBlock = self.device.read_into(inode.l3_block);
                let l2: AddrBlock = self.device.read_into(l3[l3_idx]);
                let l1: AddrBlock = self.device.read_into(l2[l2_idx]);
                l1[l1_idx]
            }
        })
    }

    pub fn chmod(&mut self, handle: BlobHandle, mode: Mode) {
        let mut inode = self.read_inode(handle);
        inode.metadata.mode = mode;
        self.write_inode(handle, &inode);
        // FIXME: timestmaps.
    }

    pub fn chown(&mut self, handle: BlobHandle, uid: u16, gid: u16) {
        let mut inode = self.read_inode(handle);
        inode.metadata.uid = uid;
        inode.metadata.gid = gid;
        self.write_inode(handle, &inode);
        // FIXME: timestmaps.
    }
}

/// Error type returned on some of the blobstore operations.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
pub enum BlobstoreError {
    #[error("Returned when the system ran out of INodes and isn't able to create new blobs")]
    OutOfInodes,
}

/// Errors with the filesystem.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
#[allow(missing_docs)]
pub enum FileSystemError {
    #[error("Invalid filesystem")]
    InvalidFilesystem,
}

/// Errors opening a blob.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
#[allow(missing_docs)]
pub enum OpenError {
    #[error("Maximum number of open counts would be exceeded")]
    OpenCountLimit,
}

/// Errors hard-linking a blob.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
#[allow(missing_docs)]
pub enum CloseError {
    #[error("The blob is not currently open")]
    NotOpened,
}

/// Errors hard-linking a blob.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
#[allow(missing_docs)]
pub enum LinkError {
    #[error("Maximum number of hard links would be exceeded")]
    HardLinkLimit,
}

/// Errors hard-linking a blob.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
#[allow(missing_docs)]
pub enum UnlinkError {
    #[error("Link count is already 0")]
    NoLinks,
}

/// Inode expansion errors
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
#[allow(missing_docs)]
pub enum ResizeError {
    #[error("Tried to add a block to an INode that is full")]
    MaxCapacityReached,
    #[error("No more blocks in the filesystem")]
    OutOfBlocks,
}

/// Inode expansion errors
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
#[allow(missing_docs)]
pub enum WriteError {
    #[error("Tried to add a block to an INode that is full")]
    ExtendError(#[from] ResizeError),
    #[error("File offset and length too large")]
    OffsetOverflow,
}

/// A unique handle for a blob.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct BlobHandle(pub u64);

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::inode::INode;
    use super::*;
    use crate::block_device::{BlockDevice, MemoryDevice};

    #[derive(Debug)]
    struct FileTester<'a, D: BlockDevice<BLOCK_SIZE>> {
        contents: Rc<RefCell<Vec<u8>>>,
        handle: BlobHandle,
        fs: &'a RefCell<Blobstore<D>>,
    }

    impl<'a, D: BlockDevice<BLOCK_SIZE>> FileTester<'a, D> {
        pub fn make(fs: &'a RefCell<Blobstore<D>>) -> Result<Self, BlobstoreError> {
            let handle = fs
                .borrow_mut()
                .make_blob(0, 0, Mode::restrictive(), Kind::Regular)?;
            Ok(Self {
                handle,
                contents: Rc::new(RefCell::new(Vec::new())),
                fs,
            })
        }

        pub fn write(&mut self, offset: u64, data: &[u8]) -> Result<(), WriteError> {
            self.fs.borrow_mut().write(self.handle, offset, data)?;
            if offset as usize + data.len() > self.contents.borrow().len() {
                self.contents
                    .borrow_mut()
                    .resize(offset as usize + data.len(), 0);
            }

            self.contents.borrow_mut()[offset as usize..(offset as usize + data.len())]
                .copy_from_slice(data);
            self.check();
            Ok(())
        }

        pub fn link(&self) -> Result<Self, LinkError> {
            self.fs.borrow_mut().link(self.handle)?;
            Ok(Self {
                handle: self.handle,
                contents: Rc::clone(&self.contents),
                fs: self.fs,
            })
        }

        pub fn metadata(&self) -> Metadata {
            self.fs.borrow().metadata(self.handle)
        }

        pub fn check(&self) {
            let meta = self.fs.borrow().metadata(self.handle);
            assert_eq!(meta.file_size_bytes, self.contents.borrow().len() as u64);

            let mut actual = vec![0; self.contents.borrow().len()];
            assert_eq!(
                self.fs.borrow_mut().read(self.handle, 0, &mut actual),
                self.contents.borrow().len() as u64
            );
            assert_eq!(&*self.contents.borrow(), &actual);
        }

        pub fn read(&self, offset: u64, count: u64) -> Vec<u8> {
            let mut out = vec![0; count as usize];
            let read = self.fs.borrow_mut().read(self.handle, offset, &mut out);
            out.resize(read as usize, 0);

            assert_eq!(
                &self.contents.borrow()[offset as usize..(offset as usize + read as usize)],
                &out
            );
            out
        }
    }

    impl<D: BlockDevice<BLOCK_SIZE>> Drop for FileTester<'_, D> {
        fn drop(&mut self) {
            self.check();
            self.fs.borrow_mut().unlink(self.handle).unwrap();
        }
    }

    struct ResourceChecker<'a, D: BlockDevice<BLOCK_SIZE>> {
        original_num_available_blocks: u64,
        original_num_available_inodes: u64,
        fs: &'a RefCell<Blobstore<D>>,
    }

    impl<'a, D: BlockDevice<BLOCK_SIZE>> ResourceChecker<'a, D> {
        pub fn new(fs: &'a RefCell<Blobstore<D>>) -> Self {
            Self {
                original_num_available_blocks: fs.borrow().get_num_free_blocks(),
                original_num_available_inodes: fs.borrow().get_num_free_inodes(),
                fs,
            }
        }
    }

    impl<D: BlockDevice<BLOCK_SIZE>> Drop for ResourceChecker<'_, D> {
        fn drop(&mut self) {
            assert_eq!(
                self.original_num_available_blocks,
                self.fs.borrow().get_num_free_blocks()
            );

            assert_eq!(
                self.original_num_available_inodes,
                self.fs.borrow().get_num_free_inodes()
            );
        }
    }

    #[test]
    fn makefs_superblock_is_correct() {
        crate::tests::init();
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
        crate::tests::init();
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
        crate::tests::init();
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
        crate::tests::init();
        let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
        let fs = Blobstore::mkfs(device);

        let superblock = &fs.superblock;
        let mut bnum = superblock.free_list_head;
        let mut free_blocks = [false; 200];
        // The head of the free list is free.
        free_blocks[bnum.unwrap().get() as usize] = true;
        while let Some(block) = bnum {
            let block = block.get();
            assert!(
                block > 0 && block < superblock.num_blocks,
                "Block number {block} out of bounds"
            );
            let freenode: FreeNodeBlock = fs.device.read_into(block);
            for num in freenode.raw().iter().copied() {
                if num != 0 {
                    assert!(!free_blocks[num as usize], "Block already marked as free");
                    free_blocks[num as usize] = true;
                }
            }

            bnum = freenode.next_block();
        }
        for (i, is_free) in free_blocks.iter().enumerate() {
            if i < superblock.free_list_head.unwrap().get() as usize {
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
        crate::tests::init();
        let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
        let mut fs = Blobstore::mkfs(device);

        let mut allocated_inodes = HashSet::new();
        let num_inodes = fs.superblock.num_iblocks * INodeBlock::inodes_per_block();

        for i in 0..num_inodes {
            let handle = fs
                .make_blob(0, 0, Mode::restrictive(), Kind::Regular)
                .expect(&format!("Failed blob allocation {i}"));
            println!("Got handle: {handle:?}");
            assert!(
                allocated_inodes.insert(handle.0),
                "Reaused the same INode: {}",
                handle.0
            );
        }

        fs.make_blob(0, 0, Mode::restrictive(), Kind::Regular)
            .expect_err("Should be out of INodes");
        fs.make_blob(0, 0, Mode::restrictive(), Kind::Regular)
            .expect_err("Should be out of INodes");
    }

    #[test]
    fn rdwr_blob_test() {
        crate::tests::init();
        let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
        let mut fs = Blobstore::mkfs(device);

        let blob = fs
            .make_blob(0, 0, Mode::restrictive(), Kind::Regular)
            .expect("Failed blob allocation");
        fs.write(blob, 0, b"hello world!")
            .expect("Failed to write to blob");
        let mut buf = [0u8; 12];
        assert_eq!(fs.read(blob, 0, &mut buf), 12);
        assert_eq!(&buf, b"hello world!");
    }

    #[test]
    fn write_empty_string() {
        crate::tests::init();
        let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
        let mut fs = Blobstore::mkfs(device);

        let blob = fs
            .make_blob(0, 0, Mode::restrictive(), Kind::Regular)
            .expect("Failed blob allocation");
        fs.write(blob, 0, b"").expect("Failed to write to blob");
        let mut buf = [0u8; 0];
        assert_eq!(fs.read(blob, 0, &mut buf), 0);
        assert_eq!(&buf, b"");
    }

    #[test]
    fn write_sequentially() {
        crate::tests::init();
        let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
        let mut fs = Blobstore::mkfs(device);

        let blob = fs
            .make_blob(0, 0, Mode::restrictive(), Kind::Regular)
            .expect("Failed blob allocation");
        fs.write(blob, 0, &[]).expect("Failed to write to blob");
        fs.write(blob, 0, &[1]).expect("Failed to write to blob");
        fs.write(blob, 1, &[0]).expect("Failed to write to blob");
        let mut buf = [0u8; 2];
        assert_eq!(fs.read(blob, 0, &mut buf), 2);
        assert_eq!(&buf, &[1, 0]);
    }

    #[test]
    fn many_files() {
        crate::tests::init();
        let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
        let fs = RefCell::new(Blobstore::mkfs(device));
        let _resource_checker = ResourceChecker::new(&fs);

        let mut blobs = Vec::new();

        let num_blobs = fs.borrow().get_num_inodes();

        for i in 0..num_blobs {
            let mut tester = FileTester::make(&fs).unwrap();
            tester
                .write(0, format!("Hello, this is blob {i}").as_bytes())
                .unwrap();

            blobs.push(tester);
        }

        FileTester::make(&fs).unwrap_err();

        for (i, tester) in blobs.iter().enumerate() {
            assert_eq!(
                &tester.read(0, 100),
                format!("Hello, this is blob {i}").as_bytes()
            )
        }
    }

    #[test]
    fn many_files_repeated() {
        crate::tests::init();
        let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
        let fs = RefCell::new(Blobstore::mkfs(device));
        let _resource_checker = ResourceChecker::new(&fs);

        for _ in 0..100 {
            let _resource_checker = ResourceChecker::new(&fs);

            let mut blobs = Vec::new();

            let num_blobs = fs.borrow().get_num_inodes();

            for i in 0..num_blobs {
                let mut tester = FileTester::make(&fs).unwrap();
                tester
                    .write(0, format!("Hello, this is blob {i}").as_bytes())
                    .unwrap();

                blobs.push(tester);
            }

            FileTester::make(&fs).unwrap_err();

            for (i, tester) in blobs.iter().enumerate() {
                assert_eq!(
                    &tester.read(0, 100),
                    format!("Hello, this is blob {i}").as_bytes()
                )
            }
        }
    }

    #[test]
    fn large_file() {
        crate::tests::init();
        let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
        let fs = RefCell::new(Blobstore::mkfs(device));
        let _resource_checker = ResourceChecker::new(&fs);

        let mut large_blob = FileTester::make(&fs).unwrap();
        let mut end = 0;
        let mut half_block = [0u8; BLOCK_SIZE as usize / 2];
        let mut rng = StdRng::seed_from_u64(971);
        rng.fill(&mut half_block);
        while let Ok(()) = large_blob.write(end, &half_block) {
            rng.fill(&mut half_block);
            end += BLOCK_SIZE / 2;
        }

        assert!(fs.borrow().get_num_free_blocks() < 4);
    }

    #[test]
    fn large_file_repeated() {
        crate::tests::init();
        let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
        let fs = RefCell::new(Blobstore::mkfs(device));
        let _resource_checker = ResourceChecker::new(&fs);

        for _ in 0..10 {
            let _resource_checker = ResourceChecker::new(&fs);
            let mut large_blob = FileTester::make(&fs).unwrap();
            let mut end = 0;
            let mut half_block = [0u8; BLOCK_SIZE as usize / 2];
            let mut rng = StdRng::seed_from_u64(971);
            rng.fill(&mut half_block);
            while let Ok(()) = large_blob.write(end, &half_block) {
                rng.fill(&mut half_block);
                end += BLOCK_SIZE / 2;
            }

            assert!(fs.borrow().get_num_free_blocks() < 4);
        }
    }

    #[test]
    fn links() {
        crate::tests::init();
        let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
        let fs = RefCell::new(Blobstore::mkfs(device));
        let _resource_checker = ResourceChecker::new(&fs);

        let mut blob = FileTester::make(&fs).unwrap();
        assert_eq!(blob.metadata().hard_links, 1);
        let mut linked = blob.link().unwrap();
        assert_eq!(blob.metadata().hard_links, 2);
        assert_eq!(linked.metadata().hard_links, 2);

        blob.write(0, b"hello world!").unwrap();
        assert_eq!(linked.read(0, 100), b"hello world!");

        linked.write(0, b"hello my friend").unwrap();
        assert_eq!(blob.read(0, 100), b"hello my friend");
    }

    mod proptests {
        use proptest::*;

        use super::*;

        proptest! {
            #[test]
            fn write_read_round_trip(s: Vec<u8>) {
                let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
                let mut fs = Blobstore::mkfs(device);

                let blob = fs.make_blob(0, 0, Mode::restrictive(), Kind::Regular).expect("Failed blob allocation");
                fs.write(blob, 0, &s)
                    .expect("Failed to write to blob");
                let mut buf = vec![0u8; s.len()];
                prop_assert_eq!(fs.read(blob, 0, &mut buf), s.len() as u64);
                prop_assert_eq!(&buf, &s);
            }

            #[test]
            fn writes_in_sequence(mut s1: Vec<u8>, s2: Vec<u8>, s3: Vec<u8>) {
                let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
                let mut fs = Blobstore::mkfs(device);

                let blob = fs.make_blob(0, 0, Mode::restrictive(), Kind::Regular).expect("Failed blob allocation");
                fs.write(blob, 0, &s1)
                    .expect("Failed to write to blob");
                fs.write(blob, s1.len() as u64, &s2)
                    .expect("Failed to write to blob");
                fs.write(blob, (s1.len() + s2.len()) as u64, &s3)
                    .expect("Failed to write to blob");
                let mut buf = vec![0u8; s1.len() + s2.len() + s3.len()];
                prop_assert_eq!(fs.read(blob, 0, &mut buf), buf.len() as u64);
                s1.extend(s2);
                s1.extend(s3);
                prop_assert_eq!(&buf, &s1);
            }

            #[test]
            fn no_block_leaks(blocks in 1u64..160, extra in 0u64..BLOCK_SIZE) {
                let device: MemoryDevice<BLOCK_SIZE> = MemoryDevice::new(200);
                let mut fs = Blobstore::mkfs(device);

                let blob = fs.make_blob(0, 0, Mode::restrictive(), Kind::Regular).expect("Failed blob allocation");
                let pre_available_blocks = fs.get_num_free_blocks();
                fs.resize(blob, blocks * BLOCK_SIZE + extra).unwrap();
                assert!(fs.get_num_free_blocks() < pre_available_blocks);
                fs.resize(blob, 0).unwrap();
                prop_assert_eq!(fs.get_num_free_blocks(), pre_available_blocks);
            }
        }
    }
}
