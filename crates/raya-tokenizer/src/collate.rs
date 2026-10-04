use crate::builder::Item;

/// A padded batch, row-major (`collate_items` without the training-only fields).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch {
    pub n: usize,
    /// Padded sequence length.
    pub len: usize,
    /// Padded marker count.
    pub max_markers: usize,
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub marker_pos: Vec<u32>,
    pub marker_mask: Vec<bool>,
    pub qtype: Vec<u8>,
}

/// Flattens the groups in order and pads to the longest item. No items gives `None`, as in
/// Laya.
///
/// `min_len` is NOT part of Laya's `collate_items`: it exists only so tests can reproduce the
/// forward goldens, which pad beyond the batch maximum. Pass 0 for Laya's behavior; raya:006
/// decides how real inference pads.
#[must_use]
pub fn collate(groups: &[Vec<Item>], pad_id: u32, min_len: usize) -> Option<Batch> {
    let items: Vec<&Item> = groups.iter().flatten().collect();
    if items.is_empty() {
        return None;
    }
    let n = items.len();
    let len = items
        .iter()
        .map(|it| it.ids.len())
        .max()
        .unwrap_or(0)
        .max(min_len);
    let k = items.iter().map(|it| it.markers.len()).max().unwrap_or(0);
    let mut batch = Batch {
        n,
        len,
        max_markers: k,
        input_ids: vec![pad_id; n * len],
        attention_mask: vec![0; n * len],
        marker_pos: vec![0; n * k],
        marker_mask: vec![false; n * k],
        qtype: Vec::with_capacity(n),
    };
    for (i, it) in items.iter().enumerate() {
        batch.input_ids[i * len..i * len + it.ids.len()].copy_from_slice(&it.ids);
        batch.attention_mask[i * len..i * len + it.ids.len()].fill(1);
        for (j, m) in it.markers.iter().enumerate() {
            batch.marker_pos[i * k + j] = u32::try_from(*m).expect("marker fits u32");
            batch.marker_mask[i * k + j] = true;
        }
        batch.qtype.push(it.qtype.id());
    }
    Some(batch)
}
