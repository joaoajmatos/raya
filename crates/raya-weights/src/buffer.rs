use std::fs::File;
use std::io::Read;
use std::path::Path;

use raya_core::{Error, Result};

/// A file read once into 4-byte-aligned storage, so `f32` views of it never need a copy.
pub struct AlignedBuf {
    words: Vec<f32>,
    len: usize,
}

impl AlignedBuf {
    pub fn as_bytes(&self) -> &[u8] {
        &bytemuck::cast_slice::<f32, u8>(&self.words)[..self.len]
    }
}

pub fn read_file(path: &Path) -> Result<AlignedBuf> {
    let io = |source| Error::Io {
        path: path.to_path_buf(),
        source,
    };
    let mut file = File::open(path).map_err(io)?;
    let len = usize::try_from(file.metadata().map_err(io)?.len()).map_err(|_| {
        io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "file larger than the address space",
        ))
    })?;
    let mut words = vec![0.0f32; len.div_ceil(4)];
    let bytes = &mut bytemuck::cast_slice_mut::<f32, u8>(&mut words)[..len];
    file.read_exact(bytes).map_err(io)?;
    Ok(AlignedBuf { words, len })
}
