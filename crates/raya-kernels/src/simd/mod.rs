//! Micro-kernels shared by the tiled driver. `portable` always exists and is the fallback. On
//! `x86_64` the `avx2` kernels are compiled into every build and used only when the CPU reports AVX2
//! and FMA at run time; `neon` stays an optional cargo feature, also runtime-detected.

pub mod portable;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(all(feature = "neon", target_arch = "aarch64"))]
mod neon;

/// Score tile stride: one row of scores holds at most this many keys.
pub const NC: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Portable,
    Avx2,
    Neon,
}

/// Whether this CPU can run the AVX2 kernels (AVX2 and FMA). Always false off `x86_64`.
fn cpu_has_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Whether the NEON kernels are compiled in and this CPU can run them.
fn cpu_has_neon() -> bool {
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
    {
        false
    }
}

impl Backend {
    /// Best backend this build contains and this CPU supports; `Portable` when detection fails
    /// or the CPU lacks the feature.
    pub fn detect() -> Self {
        Self::select(cpu_has_avx2_fma(), cpu_has_neon())
    }

    /// The selection rule, separate from the CPU probes so it can be tested for every
    /// combination.
    fn select(avx2_fma: bool, neon: bool) -> Self {
        if avx2_fma {
            Backend::Avx2
        } else if neon {
            Backend::Neon
        } else {
            Backend::Portable
        }
    }

    /// Parses `portable`, `avx2` or `neon` (as printed by `name`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "portable" => Some(Backend::Portable),
            "avx2" => Some(Backend::Avx2),
            "neon" => Some(Backend::Neon),
            _ => None,
        }
    }

    pub fn is_available(self) -> bool {
        match self {
            Backend::Portable => true,
            Backend::Avx2 => cpu_has_avx2_fma(),
            Backend::Neon => cpu_has_neon(),
        }
    }

    /// The requested backend if usable here, otherwise `Portable`.
    #[must_use]
    pub fn or_fallback(self) -> Self {
        if self.is_available() {
            self
        } else {
            Backend::Portable
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Backend::Portable => "portable",
            Backend::Avx2 => "avx2",
            Backend::Neon => "neon",
        }
    }
}

/// Scores for `mr` query rows against `nk` key rows: `out[r * NC + j] = q_r . k_j`.
/// `q` is `mr * d`, `k` is `nk * d`, `nk <= NC`, `d % 8 == 0`.
///
/// A backend the CPU cannot run is served by the portable kernel, so passing `Backend::Avx2`
/// directly (the enum is public) is always sound.
pub fn dot_tile(
    backend: Backend,
    q: &[f32],
    k: &[f32],
    d: usize,
    mr: usize,
    nk: usize,
    out: &mut [f32],
) {
    match backend {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the guard just checked that the CPU supports AVX2 and FMA.
        Backend::Avx2 if cpu_has_avx2_fma() => unsafe { avx2::dot_tile(q, k, d, mr, nk, out) },
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        // SAFETY: the guard just checked that the CPU supports NEON.
        Backend::Neon if cpu_has_neon() => unsafe { neon::dot_tile(q, k, d, mr, nk, out) },
        _ => portable::dot_tile(q, k, d, mr, nk, out),
    }
}

/// `acc += sum_j p[j] * v_j` with `v` holding `p.len()` rows of length `d`. Backend handling as
/// in `dot_tile`.
pub fn axpy_rows(backend: Backend, acc: &mut [f32], p: &[f32], v: &[f32], d: usize) {
    match backend {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: see `dot_tile`.
        Backend::Avx2 if cpu_has_avx2_fma() => unsafe { avx2::axpy_rows(acc, p, v, d) },
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        // SAFETY: see `dot_tile`.
        Backend::Neon if cpu_has_neon() => unsafe { neon::axpy_rows(acc, p, v, d) },
        _ => portable::axpy_rows(acc, p, v, d),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_prefers_avx2_then_neon_then_portable() {
        assert_eq!(Backend::select(true, true), Backend::Avx2);
        assert_eq!(Backend::select(true, false), Backend::Avx2);
        assert_eq!(Backend::select(false, true), Backend::Neon);
        assert_eq!(Backend::select(false, false), Backend::Portable);
    }

    #[test]
    fn names_round_trip() {
        for b in [Backend::Portable, Backend::Avx2, Backend::Neon] {
            assert_eq!(Backend::from_name(b.name()), Some(b));
        }
        assert_eq!(Backend::from_name("avx512"), None);
        assert_eq!(Backend::from_name(""), None);
    }

    #[test]
    fn detect_agrees_with_the_cpu_probes() {
        let want = Backend::select(cpu_has_avx2_fma(), cpu_has_neon());
        assert_eq!(Backend::detect(), want);
        assert!(Backend::detect().is_available());
        assert!(Backend::Portable.is_available());
        #[cfg(not(target_arch = "x86_64"))]
        assert!(!Backend::Avx2.is_available());
    }
}
