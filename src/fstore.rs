//! File layer of the file system.
//!
//! This layer adds directories and essentially provides names to files.

use std::ffi::OsStr;
use std::mem::{size_of, MaybeUninit};
use std::os::unix::prelude::OsStrExt;
use std::path::Path;

use thiserror::Error;

pub use crate::blobstore::{
    BlobHandle, CloseError, LinkError, Metadata, OpenError, ResizeError, UnlinkError, WriteError,
};
use crate::blobstore::{Blobstore, BlobstoreError, Kind, Mode, BLOCK_SIZE};
use crate::block_device::BlockDevice;

const FILENAME_LENGTH: usize = 256 - size_of::<BlobHandle>();

/// A file store that operates on top of a blobstore to provide file hirearchies.
#[derive(Debug)]
pub struct FStore<D: BlockDevice<BLOCK_SIZE>> {
    store: Blobstore<D>,
}

#[repr(packed)]
struct DirEntry {
    blob: BlobHandle,
    filename: [u8; FILENAME_LENGTH],
}
const ENTRY_SIZE: u64 = size_of::<DirEntry>() as u64;

const _SIZE_OF_ENTRY: () = {
    assert!(ENTRY_SIZE == 256);
};

const MAX_PATH_LEN: usize = 4096;

impl<D: BlockDevice<BLOCK_SIZE>> FStore<D> {
    /// Constructs the file storage layer from the store.
    ///
    /// This does no sort of initialization. For that, `mkfs` should be called on the
    /// store beforehand.
    pub fn new(store: Blobstore<D>) -> Self {
        Self { store }
    }

    pub fn namei(&mut self, path: &Path) -> Result<BlobHandle, LookupError> {
        if path.as_os_str().len() > MAX_PATH_LEN {
            return Err(LookupError::PathTooLong);
        }
        let parent = match path.parent() {
            Some(parent) => self.namei(parent)?,
            None => return Ok(BlobHandle(0)),
        };

        if !self.is_dir(parent) {
            return Err(LookupError::NonDirComp);
        }

        let Some((entry, _)) = self.find_dir_entry(parent, path.file_name().unwrap_or(OsStr::new(".."))) else {
            return Err(LookupError::NonExistant);
        };

        Ok(entry)
    }

    pub fn is_dir(&self, handle: BlobHandle) -> bool {
        let meta = self.store.metadata(handle);
        meta.kind == Kind::Directory
    }

    pub fn create(
        &mut self,
        path: &Path,
        uid: u16,
        gid: u16,
        mode: Mode,
    ) -> Result<BlobHandle, CreateError> {
        let filename = path.file_name().ok_or(CreateError::InvalidFileName)?;
        let dir_handle = self.namei(path.parent().ok_or(CreateError::IsRoot)?)?;
        let file_handle = self.store.make_blob(uid, gid, mode, Kind::Regular)?;

        if let Err(e) = self.add_entry_to_dir(dir_handle, file_handle, filename) {
            self.store.unlink(file_handle);
            return Err(e.into());
        }
        Ok(file_handle)
    }

    fn add_entry_to_dir(
        &mut self,
        dir: BlobHandle,
        file: BlobHandle,
        name: &OsStr,
    ) -> Result<(), DirEntryError> {
        let name = name.as_bytes();
        let length = name.len();
        if length > FILENAME_LENGTH {
            return Err(DirEntryError::FilenameTooLong);
        }

        let mut filename = [0; FILENAME_LENGTH];
        filename[..length].copy_from_slice(name);

        let entry = DirEntry {
            blob: file,
            filename,
        };

        self.write_dir_entry(dir, entry, self.num_entries_in_dir(dir));
        Ok(())
    }

    fn write_dir_entry(
        &mut self,
        dir: BlobHandle,
        entry: DirEntry,
        n: u64,
    ) -> Result<(), WriteError> {
        debug_assert!(n <= self.num_entries_in_dir(dir));
        self.store.write(dir, n * ENTRY_SIZE, entry.as_u8_slice())?;
        Ok(())
    }

    fn rm_entry_from_dir(
        &mut self,
        dir: BlobHandle,
        name: &OsStr,
    ) -> Result<BlobHandle, RmEntryError> {
        let (handle, idx) = self
            .find_dir_entry(dir, name)
            .ok_or(RmEntryError::NonExistant)?;
        let num_entries = self.num_entries_in_dir(dir);

        // If this is not the last one, then we swap it with the last entry.
        if idx < num_entries - 1 {
            let last_entry = self.get_dir_entry(dir, num_entries - 1).unwrap();
            self.write_dir_entry(dir, last_entry, idx);
        }
        self.store.resize(dir, (num_entries - 1) * ENTRY_SIZE);
        Ok(handle)
    }

    fn num_entries_in_dir(&self, dir: BlobHandle) -> u64 {
        let dir_size = self.store.metadata(dir).file_size_bytes;
        debug_assert!(dir_size % 256 == 0);
        dir_size / ENTRY_SIZE
    }

    fn get_dir_entry(&self, dir: BlobHandle, n: u64) -> Option<DirEntry> {
        let mut entry = MaybeUninit::uninit();
        let num_entries = self.num_entries_in_dir(dir);
        if n < num_entries {
            unsafe {
                let read = self
                    .store
                    .read(dir, n * ENTRY_SIZE, entry.as_u8_slice_mut());
                debug_assert!(read == ENTRY_SIZE);
                Some(entry.assume_init())
            }
        } else {
            None
        }
    }

    fn find_dir_entry(&self, dir: BlobHandle, name: &OsStr) -> Option<(BlobHandle, u64)> {
        let mut n = 0;

        while let Some(entry) = self.get_dir_entry(dir, n) {
            // FIXME: This might not take into account null termination.
            let filename = OsStr::from_bytes(&entry.filename);
            if filename == name {
                return Some((entry.blob, n));
            }
        }
        None
    }

    pub fn mkdir(
        &mut self,
        path: &Path,
        uid: u16,
        gid: u16,
        mode: Mode,
    ) -> Result<BlobHandle, CreateError> {
        todo!();
    }

    pub fn chmod(&mut self, handle: BlobHandle, mode: Mode) {
        self.store.chmod(handle, mode);
    }

    pub fn chown(&mut self, handle: BlobHandle, uid: u16, gid: u16) {
        self.store.chown(handle, uid, gid);
    }

    pub fn rename(&mut self, handle: BlobHandle, new_name: &Path) -> Result<(), RenameError> {
        todo!();
    }

    pub fn symlink(&mut self, from: &Path, to: &Path) -> Result<BlobHandle, SymlinkError> {
        todo!();
    }

    /// Marks the file as opened.
    pub fn open(&mut self, handle: BlobHandle) -> Result<(), OpenError> {
        self.store.open(handle)
    }

    /// Marks the previously opened file as closed.
    pub fn close(&mut self, handle: BlobHandle) -> Result<(), CloseError> {
        self.store.close(handle)
    }
    /// Reads the data starting at `offset` into `out`.
    ///
    /// The read will continue until reaching the length of `out` or until reaching EOF.
    ///
    /// # Returns
    ///
    /// The number of bytes actually read.
    pub fn read(&mut self, handle: BlobHandle, offset: u64, out: &mut [u8]) -> u64 {
        self.store.read(handle, offset, out)
    }

    /// Writes the data to the specified blob at the given offset.
    ///
    /// The write will copy all the bytes in `data`, possibly extending the length of the blob.
    /// If the offset is past the end of the file, the holes produced by the write will be filled
    /// with 0s.
    pub fn write(
        &mut self,
        handle: BlobHandle,
        offset: u64,
        data: &[u8],
    ) -> Result<(), WriteError> {
        self.store.write(handle, offset, data)
    }

    /// Resizes the given blob to the specified length.
    ///
    /// * If the length of the blob is < `new_len`, this is will extend the file with 0s.
    /// * If the length of the blob is > `new_len`, this is will truncate the file.
    pub fn resize(&mut self, handle: BlobHandle, new_len: u64) -> Result<(), ResizeError> {
        self.store.resize(handle, new_len)
    }

    /// Creates a link at the provided path.
    pub fn link(&mut self, handle: BlobHandle, linkpath: &Path) -> Result<(), LinkError> {
        todo!();
        self.store.link(handle)
    }

    /// Decrements the hard link count for the file.
    pub fn unlink(&mut self, path: &Path) -> Result<(), UnlinkError> {
        todo!();
        self.store.unlink(handle)
    }

    /// Returns the metadata associated with a given file.
    pub fn metadata(&self, handle: BlobHandle) -> Metadata {
        self.store.metadata(handle)
    }

    /// Get the total number of blocks in the device.
    pub fn get_num_blocks(&self) -> u64 {
        self.store.get_num_blocks()
    }

    /// Get the number of free blocks in the device.
    pub fn get_num_free_blocks(&self) -> u64 {
        self.store.get_num_free_blocks()
    }

    /// Returns the total number of INodes.
    pub fn get_num_inodes(&self) -> u64 {
        self.store.get_num_inodes()
    }

    /// Returns the number of unallocated INodes.
    pub fn get_num_free_inodes(&self) -> u64 {
        self.store.get_num_free_inodes()
    }
}

/// Errors translating a path to the underlying file.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
pub enum LookupError {
    #[error("The path is longer than the supported maximum")]
    PathTooLong,
    #[error("The path contains a component that isn't a directory")]
    NonDirComp,
    #[error("A component in the path doesn't exist")]
    NonExistant,
}

/// Errors creating a file.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
pub enum CreateError {
    #[error("Parent directory doesn't exist")]
    ParentDirLookup(#[from] LookupError),
    #[error("Attempted to create a file in that is the root directory")]
    IsRoot,
    #[error("Couldn't allocate the new blob in the store")]
    MakeBlobError(#[from] BlobstoreError),
    #[error("File name is not valid")]
    InvalidFileName,
    #[error("Error adding filename to directory")]
    DirectoryEntryError(#[from] DirEntryError),
}

/// Errors extending a directory.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
pub enum DirEntryError {
    #[error("Filename is longer than 240 bytes")]
    FilenameTooLong,
    #[error("Error adding the entry to the directory")]
    BlobstoreError(#[from] WriteError),
}

/// Errors removing an entry from the directory.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
pub enum RmEntryError {
    #[error("The entry can't be removed because it doesn't exist")]
    NonExistant,
}

/// Errors renaming a file.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
pub enum RenameError {}

/// Errors renaming a file.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
pub enum SymlinkError {}

trait AsU8Slice
where
    Self: Sized,
{
    fn as_u8_slice(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self as *const Self as *const u8, size_of::<Self>()) }
    }

    unsafe fn as_u8_slice_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self as *mut Self as *mut u8, size_of::<Self>()) }
    }
}

impl<T: Sized> AsU8Slice for T {}
