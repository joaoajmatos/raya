/// Time spent per stage, summed over workers. The `*_ns` fields are zero unless built with
/// `stage-timing`; `tiles` always counts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct StageTimes {
    pub qk_ns: u64,
    pub softmax_ns: u64,
    pub pv_ns: u64,
    pub tiles: u64,
}

impl StageTimes {
    pub fn merge(&mut self, other: &StageTimes) {
        self.qk_ns += other.qk_ns;
        self.softmax_ns += other.softmax_ns;
        self.pv_ns += other.pv_ns;
        self.tiles += other.tiles;
    }

    /// `(qk, softmax, pv)` as percentages of the measured total, or `None` if nothing was timed.
    #[allow(clippy::cast_precision_loss)]
    pub fn percentages(&self) -> Option<(f64, f64, f64)> {
        let total = (self.qk_ns + self.softmax_ns + self.pv_ns) as f64;
        (total > 0.0).then(|| {
            (
                100.0 * self.qk_ns as f64 / total,
                100.0 * self.softmax_ns as f64 / total,
                100.0 * self.pv_ns as f64 / total,
            )
        })
    }
}

/// Stopwatch that compiles to nothing without the `stage-timing` feature.
pub struct Clock {
    #[cfg(feature = "stage-timing")]
    last: std::time::Instant,
}

impl Clock {
    #[inline]
    pub fn start() -> Self {
        Self {
            #[cfg(feature = "stage-timing")]
            last: std::time::Instant::now(),
        }
    }

    /// Nanoseconds since the previous lap (or start); 0 without the feature.
    #[inline]
    #[allow(clippy::unused_self)]
    pub fn lap(&mut self) -> u64 {
        #[cfg(feature = "stage-timing")]
        {
            let now = std::time::Instant::now();
            let ns = u64::try_from(now.duration_since(self.last).as_nanos()).unwrap_or(u64::MAX);
            self.last = now;
            ns
        }
        #[cfg(not(feature = "stage-timing"))]
        0
    }
}
