/// Single entry point for attention so kernels can be swapped without touching model code.
/// Signatures arrive with features 006 and 007; shapes are not fixed yet.
pub trait AttentionKernel: Send + Sync {
    fn name(&self) -> &'static str;
}
