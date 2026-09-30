/// Dense matrix multiply behind a trait so a BLAS library can be chosen later (feature 006).
pub trait GemmBackend: Send + Sync {
    fn name(&self) -> &'static str;
}
