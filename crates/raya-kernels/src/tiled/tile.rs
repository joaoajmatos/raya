use super::stages::{Clock, StageTimes};
use crate::attention::{AttnParams, AttnShape, BLOCK};
use crate::simd::portable::exp_shifted_sum;
use crate::simd::{axpy_rows, dot_tile, Backend, NC};

/// Query rows processed together.
pub const MR: usize = 8;

/// Per-worker scratch: one micro-tile of scores and outputs, never a score matrix.
pub struct Scratch {
    scores: [f32; MR * NC],
    acc: Vec<f32>,
}

impl Scratch {
    pub fn new(head_dim: usize) -> Self {
        Self { scores: [0.0; MR * NC], acc: vec![0.0; MR * head_dim] }
    }
}

/// Computes query block `b` of one head into `out` (the block's rows, `<= BLOCK * d` floats).
/// `q`, `k`, `v` are the head's full `[seq][d]` slices.
#[allow(clippy::too_many_arguments)]
pub fn run_block(
    shape: AttnShape,
    params: &AttnParams,
    backend: Backend,
    (q, k, v): (&[f32], &[f32], &[f32]),
    b: usize,
    out: &mut [f32],
    scratch: &mut Scratch,
    times: &mut StageTimes,
) {
    let d = shape.head_dim;
    let row0 = b * BLOCK;
    let rows = out.len() / d;
    let mut i0 = 0;
    while i0 < rows {
        let mr = MR.min(rows - i0);
        let gi0 = row0 + i0;
        let valid = params.valid_len.saturating_sub(gi0).min(mr);
        let (done, pad) = out[i0 * d..(i0 + mr) * d].split_at_mut(valid * d);
        pad.fill(0.0);
        if valid > 0 {
            micro_tile(shape, params, backend, (q, k, v), gi0, valid, done, scratch, times);
        }
        i0 += mr;
    }
}

#[allow(clippy::too_many_arguments)]
fn micro_tile(
    shape: AttnShape,
    params: &AttnParams,
    backend: Backend,
    (q, k, v): (&[f32], &[f32], &[f32]),
    gi0: usize,
    mr: usize,
    out: &mut [f32],
    s: &mut Scratch,
    times: &mut StageTimes,
) {
    let d = shape.head_dim;
    let ranges: [std::ops::Range<usize>; MR] =
        std::array::from_fn(|r| params.mask.key_range(gi0 + r.min(mr - 1), params.valid_len));
    let ks = ranges[0].start;
    let ke = ranges[mr - 1].end;
    let mut m = [f32::NEG_INFINITY; MR];
    let mut l = [0.0f32; MR];
    s.acc[..mr * d].fill(0.0);
    let qt = &q[gi0 * d..(gi0 + mr) * d];
    let mut clock = Clock::start();
    let mut kt = ks;
    while kt < ke {
        let kend = (kt + NC).min(ke);
        let nk = kend - kt;
        dot_tile(backend, qt, &k[kt * d..kend * d], d, mr, nk, &mut s.scores);
        times.qk_ns += clock.lap();
        times.tiles += 1;

        let mut bounds = [(0usize, 0usize); MR];
        for r in 0..mr {
            let lo = ranges[r].start.max(kt);
            let hi = ranges[r].end.min(kend);
            if lo >= hi {
                continue;
            }
            let (a, z) = (lo - kt, hi - kt);
            bounds[r] = (a, z);
            let row = &mut s.scores[r * NC..r * NC + NC];
            let mut tmax = f32::NEG_INFINITY;
            for x in &mut row[a..z] {
                *x *= params.scale;
                tmax = tmax.max(*x);
            }
            let new_m = m[r].max(tmax);
            let alpha = if m[r] == f32::NEG_INFINITY { 0.0 } else { (m[r] - new_m).exp() };
            let sum = exp_shifted_sum(&mut row[a..z], new_m);
            l[r] = l[r] * alpha + sum;
            m[r] = new_m;
            if alpha != 1.0 {
                for x in &mut s.acc[r * d..(r + 1) * d] {
                    *x *= alpha;
                }
            }
        }
        times.softmax_ns += clock.lap();

        for r in 0..mr {
            let (a, z) = bounds[r];
            if a < z {
                let p = &s.scores[r * NC + a..r * NC + z];
                axpy_rows(backend, &mut s.acc[r * d..(r + 1) * d], p, &v[(kt + a) * d..(kt + z) * d], d);
            }
        }
        times.pv_ns += clock.lap();
        kt = kend;
    }
    for r in 0..mr {
        let inv = if l[r] > 0.0 { 1.0 / l[r] } else { 0.0 };
        for (o, a) in out[r * d..(r + 1) * d].iter_mut().zip(&s.acc[r * d..(r + 1) * d]) {
            *o = a * inv;
        }
    }
}
