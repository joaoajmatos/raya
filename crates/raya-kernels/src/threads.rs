/// Default worker count: `RAYA_THREADS` if it parses to a positive number, else the number of
/// physical cores.
pub fn default_threads() -> usize {
    std::env::var("RAYA_THREADS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| num_cpus::get_physical().max(1))
}
