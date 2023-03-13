//! Utility blocks used by the blobstore.

#![allow(dead_code)]

use core::fmt::Debug;
use core::num::NonZeroU64;
use core::ops::{Index, IndexMut};

use bitvec::prelude::*;

use super::BLOCK_SIZE;
use crate::block_device::BlockData;

/// A block that just contains data (say for a blob).
#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct RawBlock([u8; BLOCK_SIZE as usize]);
// SAFETY: Representation of data block is transparent.
unsafe impl BlockData<BLOCK_SIZE> for RawBlock {}

impl Debug for RawBlock {
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

impl RawBlock {
    pub fn zero() -> Self {
        Self([0; BLOCK_SIZE as usize])
    }

    pub fn raw(&self) -> &[u8; BLOCK_SIZE as usize] {
        &self.0
    }

    pub fn raw_mut(&mut self) -> &mut [u8; BLOCK_SIZE as usize] {
        &mut self.0
    }
}

/// A block that contains addresses of free blocks as well as a chain to the next FreeNodeBlock.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct FreeNodeBlock([u64; BLOCK_SIZE as usize / 8]);
// SAFETY: Representation of free node block is transparent.
unsafe impl BlockData<BLOCK_SIZE> for FreeNodeBlock {}

impl FreeNodeBlock {
    /// Create a FreeNodeBlock that is zeroed out.
    pub fn empty(next_block: u64) -> Self {
        let mut this = Self([0; BLOCK_SIZE as usize / 8]);
        this.0[511] = next_block;
        this
    }

    /// Initializes the FreeNodeBlock for handling `bnum`.
    pub fn for_block(bnum: u64, num_blocks: u64) -> Self {
        let mut block = FreeNodeBlock::empty(0);
        for (i, elem) in block.0.iter_mut().enumerate() {
            let free_addr = bnum + i as u64 + 1;
            if free_addr >= num_blocks {
                break;
            }
            *elem = free_addr;
        }
        block
    }

    /// Returns the underlying data.
    pub fn raw(&self) -> &[u64; BLOCK_SIZE as usize / 8] {
        &self.0
    }

    /// Returns the next block in the free list.
    ///
    /// There's nothing special aobut this. It just returns the last element in the block.
    pub fn next_block(&self) -> Option<NonZeroU64> {
        NonZeroU64::new(self.0[511])
    }

    pub fn take_next_available(&mut self) -> Option<u64> {
        // We can take any blocks 0..511, but not the last one which links to the next one.
        let block = self.0.iter_mut().take(511).find(|b| **b != 0)?;
        let result = *block;
        *block = 0;
        Some(result)
    }

    pub fn set_empty_slot(&mut self, block: u64) -> Result<(), ()> {
        let empty_slot = self.0.iter_mut().take(511).find(|b| **b == 0).ok_or(())?;
        *empty_slot = block;
        Ok(())
    }
}

/// A block that contains addresses of free blocks as well as a chain to the next FreeNodeBlock.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct BitmapBlock([u64; BLOCK_SIZE as usize / 8]);
// SAFETY: Representation of bitmap block is transparent.
unsafe impl BlockData<BLOCK_SIZE> for BitmapBlock {}

impl BitmapBlock {
    /// Create a bitmap block with all bits 0.
    pub fn zeros() -> Self {
        Self([0; BLOCK_SIZE as usize / 8])
    }

    /// View the bitmap.
    pub fn as_bitmap(&self) -> &BitSlice<u64> {
        self.0.view_bits::<Lsb0>()
    }

    /// View the bitmap mutably.
    pub fn as_bitmap_mut(&mut self) -> &mut BitSlice<u64> {
        self.0.view_bits_mut::<Lsb0>()
    }
}

/// A block that contains addresses of free blocks as well as a chain to the next FreeNodeBlock.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct AddrBlock([u64; BLOCK_SIZE as usize / 8]);
// SAFETY: Representation of free node block is transparent.
unsafe impl BlockData<BLOCK_SIZE> for AddrBlock {}

impl Index<u64> for AddrBlock {
    type Output = u64;

    fn index(&self, index: u64) -> &Self::Output {
        &self.0[index as usize]
    }
}

impl IndexMut<u64> for AddrBlock {
    fn index_mut(&mut self, index: u64) -> &mut Self::Output {
        &mut self.0[index as usize]
    }
}

impl AddrBlock {}
