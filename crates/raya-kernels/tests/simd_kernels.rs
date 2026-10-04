#![allow(clippy::many_single_char_names, clippy::similar_names)]
mod common;

use common::{max_abs_diff, rand_vec};
use raya_kernels::simd::{axpy_rows, dot_tile, portable, Backend, NC};

fn all_backends() -> [Backend; 3] {
    [Backend::Portable, Backend::Avx2, Backend::Neon]
}

/// Backends this CPU can run, plus the ones it cannot: the latter must still give the portable
/// answer because the dispatch re-checks the CPU.
#[test]
fn dot_tile_matches_portable_for_every_head_dim_and_tile_shape() {
    for d in [8, 16, 24, 64, 72, 128, 136] {
        for mr in [1, 2, 4] {
            for nk in [1, 5, 63, 64] {
                let q = rand_vec(mr * d, 11 + d as u64);
                let k = rand_vec(nk * d, 17 + nk as u64);
                let mut want = vec![0.0f32; (mr - 1) * NC + nk];
                portable::dot_tile(&q, &k, d, mr, nk, &mut want);
                for backend in all_backends() {
                    let mut got = vec![0.0f32; want.len()];
                    dot_tile(backend, &q, &k, d, mr, nk, &mut got);
                    // Row r occupies out[r * NC .. r * NC + nk]; compare only written cells.
                    for r in 0..mr {
                        let (a, b) = (&got[r * NC..r * NC + nk], &want[r * NC..r * NC + nk]);
                        assert!(
                            max_abs_diff(a, b) < 1e-4,
                            "{} d={d} mr={mr} nk={nk} row {r}",
                            backend.name()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn axpy_rows_matches_portable_for_every_head_dim_and_row_count() {
    for d in [8, 16, 24, 64, 72, 128, 136] {
        for rows in [1, 3, 64] {
            let p = rand_vec(rows, 5 + rows as u64);
            let v = rand_vec(rows * d, 9 + d as u64);
            let start = rand_vec(d, 23);
            let mut want = start.clone();
            portable::axpy_rows(&mut want, &p, &v, d);
            for backend in all_backends() {
                let mut got = start.clone();
                axpy_rows(backend, &mut got, &p, &v, d);
                assert!(
                    max_abs_diff(&got, &want) < 1e-4,
                    "{} d={d} rows={rows}",
                    backend.name()
                );
            }
        }
    }
}

/// `Backend` is a public enum, so a caller can name `Avx2` on a CPU (or architecture) that lacks
/// it. The dispatch must not reach the unsafe kernel then.
#[test]
fn naming_an_unavailable_backend_is_sound_and_portable() {
    let (d, mr, nk) = (64, 2, 8);
    let q = rand_vec(mr * d, 1);
    let k = rand_vec(nk * d, 2);
    let mut want = vec![0.0f32; NC + nk];
    portable::dot_tile(&q, &k, d, mr, nk, &mut want);
    for backend in all_backends().into_iter().filter(|b| !b.is_available()) {
        let mut got = vec![0.0f32; NC + nk];
        dot_tile(backend, &q, &k, d, mr, nk, &mut got);
        // Same code path as portable, so bit for bit.
        assert_eq!(got, want, "{}", backend.name());
    }
}
