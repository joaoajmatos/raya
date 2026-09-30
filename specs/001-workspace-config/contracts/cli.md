# Contract: `raya` command line (stubs)

```text
raya <COMMAND>

Commands:
  info      Inspect a checkpoint        (implemented by feature 003)
  predict   Run a decision              (implemented by feature 006)
  bench     Benchmark on this machine   (implemented by feature 008)
```

- `raya --help` and `raya <command> --help` exit 0.
- Each command currently writes `not implemented (feature NNN)` to stderr, nothing to stdout, and exits with
  status 2.
- Commands take no arguments in this feature; later features add theirs.
