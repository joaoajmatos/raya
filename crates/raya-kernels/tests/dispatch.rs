mod common;

use common::{max_abs_diff, qkv};
use raya_kernels::{AttentionKernel, AttnParams, AttnShape, Backend, Mask, Naive, Tiled, TiledConfig};

#[test]
fn default_features_use_only_the_portable_path() {
    if cfg!(any(feature = "avx2", feature = "neon")) {
        return;
    }
    assert_eq!(Backend::detect(), Backend::Portable);
    assert_eq!(Tiled::default().backend(), Backend::Portable);
    assert_eq!(Tiled::default().name(), "tiled-portable");
}

#[test]
fn unavailable_backend_falls_back_at_construction() {
    for backend in [Backend::Avx2, Backend::Neon] {
        let t = Tiled::new(TiledConfig { threads: Some(1), backend });
        if backend.is_available() {
            assert_eq!(t.backend(), backend);
        } else {
            assert_eq!(t.backend(), Backend::Portable);
        }
    }
}

#[test]
fn detected_backend_matches_the_oracle() {
    let detected = Backend::detect();
    let t = Tiled::new(TiledConfig { threads: Some(2), backend: detected });
    assert_eq!(t.backend(), detected);
    for mask in [Mask::Block3, Mask::Window64] {
        let shape = AttnShape { heads: 2, seq: 1000, head_dim: 64 };
        let params = AttnParams::new(mask, 963, 64);
        let (q, k, v) = qkv(shape, 3);
        let mut want = vec![0.0; shape.numel()];
        Naive.attend(shape, params, &q, &k, &v, &mut want).unwrap();
        let mut got = vec![0.0; shape.numel()];
        t.attend(shape, params, &q, &k, &v, &mut got).unwrap();
        assert!(max_abs_diff(&got, &want) < 1e-5, "{} {mask:?}", t.name());
    }
}

#[test]
fn explicit_thread_count_is_respected() {
    assert_eq!(Tiled::new(TiledConfig { threads: Some(3), backend: Backend::Portable }).threads(), 3);
    assert!(Tiled::default().threads() >= 1);
}
