/// Intermediate activations of one forward pass, in execution order, under the names of the
/// golden fixtures (`embedding_output`, `encoder_layer_00`, ..., `act_logits`).
#[derive(Debug, Default, Clone)]
pub struct Trace {
    entries: Vec<(String, Vec<usize>, Vec<f32>)>,
}

impl Trace {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, name: impl Into<String>, shape: &[usize], data: &[f32]) {
        self.entries
            .push((name.into(), shape.to_vec(), data.to_vec()));
    }

    /// Shape and data of the tensor recorded under `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<(&[usize], &[f32])> {
        self.entries
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, s, d)| (s.as_slice(), d.as_slice()))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(n, _, _)| n.as_str())
    }
}
