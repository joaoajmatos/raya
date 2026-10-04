#![allow(clippy::many_single_char_names, clippy::similar_names)]
mod common;

use common::{max_abs_diff, qkv};
use raya_kernels::{
    AttentionKernel, AttnParams, AttnShape, Backend, Mask, Naive, Tiled, TiledConfig,
};

#[test]
fn default_build_picks_the_best_backend_this_cpu_supports() {
    let detected = Backend::detect();
    assert_eq!(Tiled::default().backend(), detected);
    assert_eq!(
        Tiled::default().name(),
        format!("tiled-{}", detected.name())
    );
    #[cfg(target_arch = "x86_64")]
    {
        let cpu_has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        assert_eq!(
            detected,
            if cpu_has_avx2 {
                Backend::Avx2
            } else {
                Backend::Portable
            }
        );
    }
    // Without the `neon` feature (the default, and what macos-14 CI builds) nothing but the
    // portable path exists off x86_64.
    #[cfg(all(not(target_arch = "x86_64"), not(feature = "neon")))]
    assert_eq!(detected, Backend::Portable);
}

#[test]
fn unavailable_backend_falls_back_at_construction() {
    for backend in [Backend::Avx2, Backend::Neon] {
        let t = Tiled::new(TiledConfig {
            threads: Some(1),
            backend,
        });
        if backend.is_available() {
            assert_eq!(t.backend(), backend);
        } else {
            assert_eq!(t.backend(), Backend::Portable);
        }
    }
}

/// Every backend this CPU can run (portable always) must match the naive oracle; the default
/// build now uses AVX2 on capable `x86_64` CPUs, so the portable path is exercised explicitly.
#[test]
fn every_available_backend_matches_the_oracle() {
    let backends: Vec<Backend> = [Backend::Portable, Backend::Avx2, Backend::Neon]
        .into_iter()
        .filter(|b| b.is_available())
        .collect();
    assert!(backends.contains(&Backend::Portable));
    assert!(backends.contains(&Backend::detect()));
    for backend in backends {
        let t = Tiled::new(TiledConfig {
            threads: Some(2),
            backend,
        });
        assert_eq!(t.backend(), backend);
        for mask in [Mask::Block3, Mask::Window64] {
            for (seq, valid, head_dim) in [(1000, 963, 64), (300, 300, 72), (200, 130, 8)] {
                let shape = AttnShape {
                    heads: 2,
                    seq,
                    head_dim,
                };
                let params = AttnParams::new(mask, valid, head_dim);
                let (q, k, v) = qkv(shape, 3);
                let mut want = vec![0.0; shape.numel()];
                Naive.attend(shape, params, &q, &k, &v, &mut want).unwrap();
                let mut got = vec![0.0; shape.numel()];
                t.attend(shape, params, &q, &k, &v, &mut got).unwrap();
                assert!(
                    max_abs_diff(&got, &want) < 1e-5,
                    "{} {mask:?} seq={seq} d={head_dim}",
                    t.name()
                );
            }
        }
    }
}

#[test]
fn explicit_thread_count_is_respected() {
    assert_eq!(
        Tiled::new(TiledConfig {
            threads: Some(3),
            backend: Backend::Portable
        })
        .threads(),
        3
    );
    assert!(Tiled::default().threads() >= 1);
}
