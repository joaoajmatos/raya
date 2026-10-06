//! US3: a second `AttentionKernel` runs the unchanged model code.
#![allow(
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
mod common;

use std::sync::Mutex;

use raya_core::{Error, ForwardInput, Model};
use raya_kernels::{AttentionKernel, AttnParams, AttnShape, KernelError, Mask, Naive};

/// Delegates to `Naive` and records every call.
struct Recording {
    calls: Mutex<Vec<(AttnShape, Mask, usize)>>,
}

impl AttentionKernel for Recording {
    fn name(&self) -> &'static str {
        "recording"
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
        self.calls
            .lock()
            .unwrap()
            .push((shape, params.mask, params.valid_len));
        Naive.attend(shape, params, q, k, v, out)
    }
}

struct Refusing;

impl AttentionKernel for Refusing {
    fn name(&self) -> &'static str {
        "refusing"
    }

    fn attend(
        &self,
        _: AttnShape,
        params: AttnParams,
        _: &[f32],
        _: &[f32],
        _: &[f32],
        _: &mut [f32],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported(format!(
            "mask {}",
            params.mask.name()
        )))
    }
}

fn forward(kernel: &dyn AttentionKernel) -> Result<raya_core::ForwardOutput, Error> {
    let model = Model::new(common::config()).unwrap();
    let params = common::params();
    let net = model.bind(&params).unwrap();
    let ids = common::ids(11);
    let input = ForwardInput {
        input_ids: &ids,
        valid_len: 9,
        marker_pos: &[2, 5],
        marker_mask: &[true, true],
        qtype: 1,
    };
    net.forward(&input, kernel, None)
}

#[test]
fn a_second_kernel_runs_unchanged_and_sees_the_documented_masks() {
    let rec = Recording {
        calls: Mutex::new(Vec::new()),
    };
    let via_rec = forward(&rec).unwrap();
    assert_eq!(via_rec, forward(&Naive).unwrap());

    let calls = rec.calls.lock().unwrap();
    // layer 0 global, layer 1 local (local_attention 4 -> 2 per side), then the head layer
    let masks: Vec<Mask> = calls.iter().map(|c| c.1).collect();
    assert_eq!(masks, [Mask::Full, Mask::Sliding(2), Mask::Full]);
    assert!(calls.iter().all(|c| c.2 == 9), "valid_len passed through");
    assert_eq!(
        (calls[0].0.heads, calls[0].0.seq, calls[0].0.head_dim),
        (2, 11, 8)
    );
    assert_eq!((calls[2].0.heads, calls[2].0.head_dim), (1, common::D));
}

#[test]
fn a_kernel_failure_surfaces_as_an_error() {
    let err = forward(&Refusing).unwrap_err();
    assert!(
        matches!(&err, Error::Kernel(m) if m.contains("unsupported") && m.contains("full")),
        "{err}"
    );
}
