//! Block device abstraction and testing utilities.

use std::mem::MaybeUninit;

/// A block device is a simple abstraction over a storage medium that operates in blocks of bytes.
pub unsafe trait BlockDevice<const BLOCK_SIZE: usize> {
    /// Reads the block `bnum` into `out`.
    ///
    /// This function may panic if the `bnum` is out of bounds.
    ///
    /// # Safety
    ///
    /// `out`'s length must be at least a BLOCK_SIZE.
    unsafe fn read_unchecked(&self, bnum: u32, out: &mut [u8]);

    /// Writes the data in `inp` to the block `bnum`.
    ///
    /// This function may panic if the `bnum` is out of bounds.
    ///
    /// # Safety
    ///
    /// `inp`'s length must be at least a BLOCK_SIZE.
    unsafe fn write_unchecked(&mut self, bnum: u32, inp: &[u8]);

    /// Reads the block `bnum` into `out`.
    ///
    /// This function may panic if `bnum` is out of bounds or if `out` is smaller than BLOCK_SIZE.
    fn read_slice(&self, bnum: u32, out: &mut [u8]) {
        assert!(
            out.len() >= BLOCK_SIZE,
            "Slice must be at least the length of the block"
        );

        // SAFETY: Slice is long enough.
        unsafe { self.read_unchecked(bnum, out) }
    }

    /// Writes the data in `inp` to the block `bnum`.
    ///
    /// This function may panic if `bnum` is out of bounds or if `inp` is smaller than BLOCK_SIZE.
    fn write_slice(&mut self, bnum: u32, inp: &[u8]) {
        assert!(
            inp.len() >= BLOCK_SIZE,
            "Slice must be at least the length of the block"
        );

        // SAFETY: Slice is long enough.
        unsafe { self.write_unchecked(bnum, inp) }
    }

    /// Reads the data in the block into the `out` by effectively performing a memcpy.
    ///
    /// This is safe because of the invariant for a type to implement `BlockData`.
    fn read<T>(&self, bnum: u32, out: &mut T)
    where
        T: BlockData<BLOCK_SIZE>,
    {
        const {
            use core::mem::size_of;
            assert!(size_of::<T>() == BLOCK_SIZE);
        };
        unsafe {
            let out = core::slice::from_raw_parts_mut(out as *mut T as *mut u8, BLOCK_SIZE);
            self.read_unchecked(bnum, out);
        }
    }

    /// Writes the data in in `inp` to disk by effectively performing a memcpy.
    fn write<T>(&mut self, bnum: u32, inp: &T)
    where
        T: BlockData<BLOCK_SIZE>,
    {
        const {
            use core::mem::size_of;
            assert!(size_of::<T>() == BLOCK_SIZE);
        };
        unsafe {
            let inp = core::slice::from_raw_parts(inp as *const T as *const u8, BLOCK_SIZE);
            self.write_unchecked(bnum, inp);
        }
    }
}

/// This trait is used by the block device to perform read and write operations directly on
/// user defined data types.
///
/// # Safety
///
/// The implementor must guarantee that a read from won't cause the data in Self to be
/// invalid.
pub unsafe trait BlockData<const BLOCK_SIZE: usize>: Sized {}

unsafe impl<const BLOCK_SIZE: usize> BlockData<BLOCK_SIZE> for [u8; BLOCK_SIZE] {}
unsafe impl<const BLOCK_SIZE: usize> BlockData<BLOCK_SIZE> for [i8; BLOCK_SIZE] {}
unsafe impl<T, const BLOCK_SIZE: usize> BlockData<BLOCK_SIZE> for MaybeUninit<T> where
    T: BlockData<BLOCK_SIZE>
{
}

/// An implementation of `BlockDevice` that stores all the data in an internal buffer.
///
/// This struct can be used for testing.
#[derive(Debug, Clone)]
pub struct MemoryDevice<const BLOCK_SIZE: usize> {
    buffer: Box<[u8]>,
}

impl<const BLOCK_SIZE: usize> MemoryDevice<BLOCK_SIZE> {
    /// Constructs a new MemoryDevice.
    pub fn new(num_blocks: u32) -> Self {
        let num_blocks = num_blocks as usize;
        let capacity = num_blocks * BLOCK_SIZE;
        Self {
            buffer: vec![0; capacity].into_boxed_slice(),
        }
    }

    /// Returns the number of blocks associated with this memory device.
    pub fn num_blocks(&self) -> u32 {
        (self.buffer.len() / BLOCK_SIZE) as u32
    }
}

unsafe impl<const BLOCK_SIZE: usize> BlockDevice<BLOCK_SIZE> for MemoryDevice<BLOCK_SIZE> {
    unsafe fn read_unchecked(&self, bnum: u32, out: &mut [u8]) {
        assert!(bnum < self.num_blocks());
        let bnum = bnum as usize;

        unsafe {
            core::ptr::copy(
                self.buffer.as_ptr().add(bnum * BLOCK_SIZE),
                out.as_mut_ptr(),
                BLOCK_SIZE,
            );
        }
    }

    unsafe fn write_unchecked(&mut self, bnum: u32, inp: &[u8]) {
        assert!(bnum < self.num_blocks());
        let bnum = bnum as usize;

        unsafe {
            core::ptr::copy(
                inp.as_ptr(),
                self.buffer.as_mut_ptr().add(bnum * BLOCK_SIZE),
                BLOCK_SIZE,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_read_write() {
        let mut device: MemoryDevice<4096> = MemoryDevice::new(20);
        assert_eq!(device.num_blocks(), 20);

        let write_block: [u8; 4096] = (0..=255)
            .into_iter()
            .cycle()
            .take(4096)
            .collect::<Vec<u8>>()
            .try_into()
            .unwrap();
        device.write_slice(0, &write_block);
        device.write_slice(18, &write_block);
        device.write_slice(12, &write_block);

        let mut read_block = [0u8; 4096];
        device.read_slice(0, &mut read_block);
        assert!(read_block == write_block);
        device.read_slice(18, &mut read_block);
        assert!(read_block == write_block);
        device.read_slice(12, &mut read_block);
        assert!(read_block == write_block);
    }

    #[test]
    fn reading_and_writing_arrays() {
        let mut device: MemoryDevice<4096> = MemoryDevice::new(20);
        assert_eq!(device.num_blocks(), 20);

        let write_block: [u8; 4096] = (0..=255)
            .into_iter()
            .cycle()
            .take(4096)
            .collect::<Vec<u8>>()
            .try_into()
            .unwrap();
        device.write(0, &write_block);
        device.write(18, &write_block);
        device.write(12, &write_block);

        let mut read_block: MaybeUninit<[u8; 4096]> = MaybeUninit::uninit();
        device.read(0, &mut read_block);
        assert!(unsafe { read_block.assume_init_read() } == write_block);
        device.read(18, &mut read_block);
        assert!(unsafe { read_block.assume_init_read() } == write_block);
        device.read(12, &mut read_block);
        assert!(unsafe { read_block.assume_init_read() } == write_block);
    }

    #[repr(C, packed)]
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    struct CustomBlockType {
        a: u32,
        b: u32,
        c: u8,
        d: u8,
        e: u8,
        f: u8,
        _pad: [u8; 4084],
    }

    unsafe impl BlockData<4096> for CustomBlockType {}

    #[test]
    fn read_and_writing_custom_type() {
        let mut device: MemoryDevice<4096> = MemoryDevice::new(20);
        assert_eq!(device.num_blocks(), 20);

        let write_block = CustomBlockType {
            a: 1,
            b: 2,
            c: 3,
            d: 4,
            e: 5,
            f: 6,
            _pad: [0u8; 4084],
        };
        device.write(0, &write_block);
        device.write(18, &write_block);
        device.write(12, &write_block);

        let mut read_block = CustomBlockType {
            a: 0,
            b: 0,
            c: 0,
            d: 0,
            e: 0,
            f: 0,
            _pad: [0u8; 4084],
        };

        device.read(0, &mut read_block);
        assert!(read_block == write_block);
        device.read(18, &mut read_block);
        assert!(read_block == write_block);
        device.read(12, &mut read_block);
        assert!(read_block == write_block);
    }

    #[test]
    fn read_and_writing_custom_uninit_type() {
        let mut device: MemoryDevice<4096> = MemoryDevice::new(20);
        assert_eq!(device.num_blocks(), 20);

        let write_block = CustomBlockType {
            a: 1,
            b: 2,
            c: 3,
            d: 4,
            e: 5,
            f: 6,
            _pad: [0u8; 4084],
        };
        device.write(0, &write_block);
        device.write(18, &write_block);
        device.write(12, &write_block);

        let mut read_block: MaybeUninit<CustomBlockType> = MaybeUninit::uninit();
        device.read(0, &mut read_block);
        // SAFETY: Has been initialized.
        let mut read_block = unsafe { read_block.assume_init() };
        assert!(read_block == write_block);
        device.read(18, &mut read_block);
        assert!(read_block == write_block);
        device.read(12, &mut read_block);
        assert!(read_block == write_block);
    }
}
