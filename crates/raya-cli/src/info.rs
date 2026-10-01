use std::path::Path;

use raya_core::ModelConfig;
use raya_weights::WeightStore;

/// The `key: value` lines of `raya info`, computed from the loaded tensors.
pub fn render(dir: &Path, config: &ModelConfig, store: &WeightStore) -> String {
    let e = &config.encoder;
    format!(
        "checkpoint: {}\nhidden_size: {}\nencoder_layers: {}\nattention_heads: {}\nhead_layers: {}\n\
         tensors: {}\nparameters: {}\nbytes: {}\n",
        dir.display(),
        e.hidden_size,
        e.num_hidden_layers,
        e.num_attention_heads,
        config.agent.head_layers,
        store.tensor_count(),
        store.parameter_count(),
        store.total_bytes(),
    )
}
