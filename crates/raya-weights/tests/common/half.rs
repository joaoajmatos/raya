//! Generates half-precision copies of the h64 fixture in a temp dir. The narrowing and the
//! reference widening are written from the formats' definitions, independent of the crate.

use std::path::{Path, PathBuf};

use safetensors::Dtype;

use super::{h64_path, read_all, write_all};

/// bfloat16 from f32, round to nearest even (NaN stays NaN).
pub fn narrow_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return u16::try_from((b >> 16) | 0x0040).unwrap();
    }
    u16::try_from(b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16).unwrap()
}

/// binary16 from f32, round to nearest even, with subnormals and overflow to infinity.
#[allow(clippy::many_single_char_names)]
pub fn narrow_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = u16::try_from((b >> 16) & 0x8000).unwrap();
    let exp = i32::try_from((b >> 23) & 0xff).unwrap();
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man == 0 { 0 } else { 0x0200 };
    }
    let e = exp - 112;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = u32::try_from(14 - e).unwrap();
        let mut h = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        if rem > halfway || (rem == halfway && h & 1 == 1) {
            h += 1;
        }
        return sign | u16::try_from(h).unwrap();
    }
    let mut h = (u32::try_from(e).unwrap() << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | u16::try_from(h).unwrap()
}

pub fn widen_bf16(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

pub fn widen_f16(h: u16) -> f32 {
    let sign = if h & 0x8000 == 0 { 1.0f64 } else { -1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f64::from(h & 0x3ff);
    let v = match exp {
        0 => sign * man * 2f64.powi(-24),
        31 => panic!("test weights must stay finite in fp16"),
        _ => sign * (1.0 + man / 1024.0) * 2f64.powi(exp - 15),
    };
    // Every f16 is exactly representable as an f32.
    #[allow(clippy::cast_possible_truncation)]
    let out = v as f32;
    out
}

/// h64 with the tensors `pick` selects narrowed to `dtype` (BF16 or F16), the rest left F32.
/// Written into `dir`; nothing under `tests/data/` is touched.
pub fn narrowed_h64(dir: &Path, dtype: Dtype, pick: impl Fn(&str) -> bool) -> PathBuf {
    let mut tensors = read_all(&h64_path());
    for (name, (dt, _, data)) in &mut tensors {
        if !pick(name) {
            continue;
        }
        assert_eq!(*dt, Dtype::F32);
        let mut out = Vec::with_capacity(data.len() / 2);
        for c in data.chunks_exact(4) {
            let x = f32::from_le_bytes(c.try_into().unwrap());
            let h = match dtype {
                Dtype::BF16 => narrow_bf16(x),
                Dtype::F16 => narrow_f16(x),
                other => panic!("{other:?}"),
            };
            out.extend_from_slice(&h.to_le_bytes());
        }
        *dt = dtype;
        *data = out;
    }
    let path = dir.join(format!("{dtype:?}.safetensors"));
    write_all(&path, &tensors);
    path
}
