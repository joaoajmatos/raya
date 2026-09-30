//! Micro-kernels shared by the tiled driver. `portable` is the default and always present;
//! `avx2` and `neon` are optional cargo features selected at construction by runtime detection.

pub mod portable;

#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
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

impl Backend {
    /// Best backend compiled in and supported by this CPU; `Portable` with default features.
    pub fn detect() -> Self {
        #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Backend::Avx2;
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Backend::Neon;
        }
        Backend::Portable
    }

    pub fn is_available(self) -> bool {
        self == Backend::Portable || self == Self::detect()
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
        #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
        // SAFETY: `Backend::Avx2` is only produced by `detect`/`or_fallback` after the CPU
        // reported AVX2 and FMA.
        Backend::Avx2 => unsafe { avx2::dot_tile(q, k, d, mr, nk, out) },
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        // SAFETY: `Backend::Neon` is only produced after NEON was detected.
        Backend::Neon => unsafe { neon::dot_tile(q, k, d, mr, nk, out) },
        _ => portable::dot_tile(q, k, d, mr, nk, out),
    }
}

/// `acc += sum_j p[j] * v_j` with `v` holding `p.len()` rows of length `d`.
pub fn axpy_rows(backend: Backend, acc: &mut [f32], p: &[f32], v: &[f32], d: usize) {
    match backend {
        #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
        // SAFETY: see `dot_tile`.
        Backend::Avx2 => unsafe { avx2::axpy_rows(acc, p, v, d) },
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        // SAFETY: see `dot_tile`.
        Backend::Neon => unsafe { neon::axpy_rows(acc, p, v, d) },
        _ => portable::axpy_rows(acc, p, v, d),
    }
}
