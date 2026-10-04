//! Exact widening of half-precision weights to fp32 (every f16 and bf16 value is an f32).

/// bfloat16 is the top half of an f32, so widening is a shift.
pub(crate) fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// IEEE 754 binary16 to binary32, keeping sign of zero, infinities and NaN payloads.
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let bits = u32::from(h);
    let sign = (bits & 0x8000) << 16;
    let exp = (bits >> 10) & 0x1f;
    let man = bits & 0x3ff;
    let out = match exp {
        0 if man == 0 => sign,
        0 => {
            // Subnormal: man * 2^-24. With `p` the index of the top set bit it is
            // 1.xxx * 2^(p - 24), a normal f32.
            let p = man.ilog2();
            sign | ((p + 103) << 23) | ((man << (23 - p)) & 0x007f_ffff)
        }
        31 => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent reference: the value as an f64 computed from the format's definition.
    fn f16_reference(h: u16) -> f64 {
        let sign = if h & 0x8000 == 0 { 1.0 } else { -1.0 };
        let exp = i32::from((h >> 10) & 0x1f);
        let man = f64::from(h & 0x3ff);
        match exp {
            0 => sign * man * 2f64.powi(-24),
            31 if man == 0.0 => sign * f64::INFINITY,
            31 => f64::NAN,
            _ => sign * (1.0 + man / 1024.0) * 2f64.powi(exp - 15),
        }
    }

    #[test]
    fn every_f16_pattern_matches_the_definition() {
        for h in 0..=u16::MAX {
            let got = f16_to_f32(h);
            let want = f16_reference(h);
            if want.is_nan() {
                assert!(got.is_nan(), "{h:#06x}");
                // The payload moves up by 13 bits and the sign is kept.
                assert_eq!(
                    got.to_bits() & 0x807f_ffff,
                    (u32::from(h & 0x8000) << 16) | (u32::from(h & 0x3ff) << 13),
                    "{h:#06x}"
                );
            } else {
                assert_eq!(f64::from(got).to_bits(), want.to_bits(), "{h:#06x}");
                assert_eq!(got.is_sign_negative(), h & 0x8000 != 0, "{h:#06x}");
            }
        }
    }

    #[test]
    fn f16_special_values() {
        for (h, want) in [
            (0x0000, 0.0f32),
            (0x3c00, 1.0),
            (0xc000, -2.0),
            (0x7bff, 65504.0),
            (0x0400, 2f32.powi(-14)),
            (0x0001, 2f32.powi(-24)),
            (0x03ff, 1023.0 * 2f32.powi(-24)),
            (0x7c00, f32::INFINITY),
            (0xfc00, f32::NEG_INFINITY),
        ] {
            assert_eq!(f16_to_f32(h).to_bits(), want.to_bits(), "{h:#06x}");
        }
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn bf16_is_the_high_half_of_f32() {
        for (h, want) in [
            (0x0000, 0.0f32),
            (0x8000, -0.0),
            (0x3f80, 1.0),
            (0xc000, -2.0),
            (0x3f00, 0.5),
            (0x7f80, f32::INFINITY),
            (0xff80, f32::NEG_INFINITY),
            (0x0080, f32::MIN_POSITIVE),
            (0x7f7f, f32::from_bits(0x7f7f_0000)),
        ] {
            assert_eq!(bf16_to_f32(h).to_bits(), want.to_bits(), "{h:#06x}");
        }
        assert!(bf16_to_f32(0x7fc0).is_nan());
        // Every pattern widens to an f32 whose low 16 bits are zero and whose high bits are `h`.
        for h in 0..=u16::MAX {
            assert_eq!(bf16_to_f32(h).to_bits(), u32::from(h) << 16);
        }
    }
}
