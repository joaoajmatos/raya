#![allow(clippy::many_single_char_names, clippy::similar_names)]
mod common;

use common::qkv;
use raya_kernels::{AttnParams, AttnShape, Mask, Tiled, TiledConfig};

#[test]
fn stage_percentages_sum_to_one_hundred_when_timed() {
    let shape = AttnShape {
        heads: 2,
        seq: 512,
        head_dim: 64,
    };
    let params = AttnParams::new(Mask::Window64, 512, 64);
    let (q, k, v) = qkv(shape, 4);
    let mut out = vec![0.0; shape.numel()];
    let tiled = Tiled::new(TiledConfig {
        threads: Some(2),
        ..TiledConfig::default()
    });
    let times = tiled
        .attend_with_stages(shape, params, &q, &k, &v, &mut out)
        .unwrap();
    if cfg!(feature = "stage-timing") {
        let (a, b, c) = times.percentages().expect("timed");
        assert!((a + b + c - 100.0).abs() < 1e-6);
        assert!(times.tiles > 0);
    } else {
        assert_eq!((times.qk_ns, times.softmax_ns, times.pv_ns), (0, 0, 0));
        assert!(times.percentages().is_none());
    }
}
