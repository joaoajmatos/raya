mod stages;
mod tile;

pub use stages::StageTimes;

use std::sync::Mutex;

use crate::attention::{validate, AttentionKernel, AttnParams, AttnShape, KernelError, BLOCK};
use crate::simd::Backend;
use crate::threads::default_threads;

#[derive(Debug, Clone, Copy)]
pub struct TiledConfig {
    /// Worker count; `None` reads `RAYA_THREADS`, then the physical core count.
    pub threads: Option<usize>,
    pub backend: Backend,
}

impl Default for TiledConfig {
    fn default() -> Self {
        Self {
            threads: None,
            backend: Backend::detect(),
        }
    }
}

/// FlashAttention-style tiled kernel: online softmax, no score matrix, parallel over
/// (head, query block).
#[derive(Debug, Clone, Copy)]
pub struct Tiled {
    threads: usize,
    backend: Backend,
}

impl Default for Tiled {
    fn default() -> Self {
        Self::new(TiledConfig::default())
    }
}

impl Tiled {
    pub fn new(config: TiledConfig) -> Self {
        Self {
            threads: config.threads.unwrap_or_else(default_threads).max(1),
            backend: config.backend.or_fallback(),
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Like `attend`, also returning per-stage times (all zero without `stage-timing`).
    pub fn attend_with_stages(
        &self,
        shape: AttnShape,
        params: AttnParams,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
    ) -> Result<StageTimes, KernelError> {
        validate(shape, &params, q, k, v, out)?;
        let d = shape.head_dim;
        let head_len = shape.seq * d;
        if head_len == 0 {
            return Ok(StageTimes::default());
        }
        let mut tasks: Vec<(usize, usize, &mut [f32])> = Vec::new();
        for (h, head_out) in out.chunks_mut(head_len).enumerate() {
            for (b, block_out) in head_out.chunks_mut(BLOCK * d).enumerate() {
                tasks.push((h, b, block_out));
            }
        }
        tasks.reverse();
        let queue = Mutex::new(tasks);
        let total = Mutex::new(StageTimes::default());
        let workers = self
            .threads
            .min(shape.heads * shape.seq.div_ceil(BLOCK))
            .max(1);
        let backend = self.backend;
        let run = || {
            let mut scratch = tile::Scratch::new(d);
            let mut times = StageTimes::default();
            loop {
                let task = queue.lock().expect("task queue").pop();
                let Some((h, b, block_out)) = task else { break };
                let head = h * head_len..(h + 1) * head_len;
                tile::run_block(
                    shape,
                    &params,
                    backend,
                    (&q[head.clone()], &k[head.clone()], &v[head]),
                    b,
                    block_out,
                    &mut scratch,
                    &mut times,
                );
            }
            total.lock().expect("stage totals").merge(&times);
        };
        if workers == 1 {
            run();
        } else {
            std::thread::scope(|scope| {
                for _ in 0..workers {
                    scope.spawn(run);
                }
            });
        }
        Ok(total.into_inner().expect("stage totals"))
    }
}

impl AttentionKernel for Tiled {
    fn name(&self) -> &'static str {
        match self.backend {
            Backend::Portable => "tiled-portable",
            Backend::Avx2 => "tiled-avx2",
            Backend::Neon => "tiled-neon",
        }
    }

    fn attend(
        &self,
        shape: AttnShape,
        params: AttnParams,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
    ) -> Result<(), KernelError> {
        self.attend_with_stages(shape, params, q, k, v, out)
            .map(|_| ())
    }
}
