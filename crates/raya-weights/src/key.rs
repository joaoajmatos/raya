/// Part of one decision-head transformer layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeadPart {
    InProjW,
    InProjB,
    OutProjW,
    OutProjB,
    Lin1W,
    Lin1B,
    Lin2W,
    Lin2B,
    Norm1W,
    Norm1B,
    Norm2W,
    Norm2B,
}

/// Typed identity of one Laya parameter; Laya's string names live only in `mapping.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamKey {
    TokEmbeddings,
    EmbedNorm,
    AttnNorm(usize),
    Wqkv(usize),
    AttnWo(usize),
    MlpNorm(usize),
    MlpWi(usize),
    MlpWo(usize),
    FinalNorm,
    Head { layer: usize, part: HeadPart },
    TypeEmb,
    Scorer { idx: usize, bias: bool },
    ActHead { idx: usize, bias: bool },
    Temperature,
}
