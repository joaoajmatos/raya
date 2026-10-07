use serde::Serialize;

use crate::simd::Backend;

pub const LABEL_CLOUD: &str = "cloud-agent hardware, indicative only";
/// Written when `RAYA_TARGET_MACHINE` is not `1`: the run makes no claim about which machine it ran on.
pub const LABEL_UNLABELED: &str =
    "unlabeled (set RAYA_TARGET_MACHINE=1 to label the target machine), indicative only";
pub const LABEL_TARGET: &str = "target machine";

/// Machine and build description stored next to benchmark results (constitution III).
#[derive(Debug, Clone, Serialize)]
pub struct BenchMetadata {
    pub label: String,
    pub cpu_model: String,
    pub logical_cpus: usize,
    pub physical_cpus: usize,
    pub threads: usize,
    /// What is compiled in (cargo features and, on `x86_64`, `avx2`), not what ran: see `backend`.
    pub features: Vec<String>,
    /// The micro-kernel backend the run used (`portable`, `avx2` or `neon`).
    pub backend: String,
    pub rustflags: String,
    pub rustc: String,
    pub git_commit: String,
    pub timestamp: String,
}

fn cpu_model() -> String {
    #[cfg(target_os = "linux")]
    if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo") {
        if let Some(line) = s.lines().find(|l| l.starts_with("model name")) {
            if let Some((_, v)) = line.split_once(':') {
                return v.trim().to_string();
            }
        }
    }
    #[cfg(target_os = "macos")]
    if let Ok(o) = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
    {
        if o.status.success() {
            return String::from_utf8_lossy(&o.stdout).trim().to_string();
        }
    }
    #[cfg(target_os = "windows")]
    if let Ok(v) = std::env::var("PROCESSOR_IDENTIFIER") {
        return v;
    }
    "unknown".to_string()
}

fn enabled_features() -> Vec<String> {
    let mut f = vec!["portable".to_string()];
    // Compiled in by default on x86_64; whether it ran is `backend`.
    if cfg!(target_arch = "x86_64") {
        f.push("avx2".into());
    }
    if cfg!(feature = "neon") {
        f.push("neon".into());
    }
    if cfg!(feature = "stage-timing") {
        f.push("stage-timing".into());
    }
    f
}

impl BenchMetadata {
    /// Metadata for a run with the given worker count. The label is `target machine` when
    /// `RAYA_TARGET_MACHINE=1` and the unlabeled text otherwise, never a hardware claim.
    pub fn collect(threads: usize) -> Self {
        Self::collect_with(
            threads,
            std::env::var("RAYA_TARGET_MACHINE").ok().as_deref(),
        )
    }

    fn collect_with(threads: usize, target_env: Option<&str>) -> Self {
        let on_target = target_env == Some("1");
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        Self {
            label: if on_target {
                LABEL_TARGET
            } else {
                LABEL_UNLABELED
            }
            .to_string(),
            cpu_model: cpu_model(),
            logical_cpus: num_cpus::get(),
            physical_cpus: num_cpus::get_physical(),
            threads,
            features: enabled_features(),
            backend: Backend::detect().name().to_string(),
            rustflags: option_env!("RUSTFLAGS").unwrap_or("").to_string(),
            rustc: env!("RAYA_RUSTC_VERSION").to_string(),
            git_commit: env!("RAYA_GIT_COMMIT").to_string(),
            timestamp: format!("unix:{secs}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialises_every_field() {
        let json = serde_json::to_value(BenchMetadata::collect(2)).unwrap();
        for key in [
            "label",
            "cpu_model",
            "logical_cpus",
            "physical_cpus",
            "threads",
            "features",
            "backend",
            "rustflags",
            "rustc",
            "git_commit",
            "timestamp",
        ] {
            assert!(json.get(key).is_some(), "missing {key}");
        }
        assert_eq!(json["threads"], 2);
    }

    #[test]
    fn label_never_claims_hardware_without_the_env_var() {
        for env in [None, Some(""), Some("0"), Some("yes")] {
            let label = BenchMetadata::collect_with(1, env).label;
            assert_eq!(label, LABEL_UNLABELED, "env {env:?}");
            assert!(!label.contains("cloud-agent"));
        }
    }

    #[test]
    fn label_is_target_machine_when_env_is_one() {
        assert_eq!(
            BenchMetadata::collect_with(1, Some("1")).label,
            LABEL_TARGET
        );
    }
}
