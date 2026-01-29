# Rust CI/CD Configuration

## Pre-commit Setup

### Installation

```bash
# Install pre-commit (requires Python)
pip install pre-commit

# Install hooks
pre-commit install
```

### .pre-commit-config.yaml

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: detect-private-key

  - repo: local
    hooks:
      - id: cargo-fmt
        name: cargo fmt
        entry: cargo fmt --all --
        language: system
        types: [rust]
        pass_filenames: false

      - id: cargo-clippy
        name: cargo clippy
        entry: cargo clippy --all-targets --all-features -- -D warnings
        language: system
        types: [rust]
        pass_filenames: false

      - id: cargo-check
        name: cargo check
        entry: cargo check --all-targets --all-features
        language: system
        types: [rust]
        pass_filenames: false

      - id: cargo-test
        name: cargo test
        entry: cargo test
        language: system
        types: [rust]
        pass_filenames: false
        stages: [pre-push]

  # Secret detection
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.4
    hooks:
      - id: gitleaks

  # Commit message validation
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v3.3.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
```

## Clippy Configuration

### clippy.toml

```toml
# Cognitive complexity threshold
cognitive-complexity-threshold = 25

# Maximum lines in a function
too-many-lines-threshold = 100

# Maximum arguments
too-many-arguments-threshold = 7

# Type complexity threshold
type-complexity-threshold = 250

# MSRV (Minimum Supported Rust Version)
msrv = "1.75"
```

### Cargo.toml Clippy Configuration

```toml
[lints.rust]
unsafe_code = "forbid"

[lints.clippy]
# Security
unwrap_used = "warn"
expect_used = "warn"
panic = "warn"

# Correctness
all = "warn"
correctness = "deny"

# Complexity
cognitive_complexity = "warn"
too_many_arguments = "warn"
too_many_lines = "warn"

# Style
pedantic = "warn"

# Nursery (experimental but useful)
nursery = "warn"

# Allow some pedantic lints that are too noisy
module_name_repetitions = "allow"
must_use_candidate = "allow"
missing_errors_doc = "allow"
missing_panics_doc = "allow"
```

## GitHub Actions CI Workflow

### .github/workflows/ci.yml

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: -Dwarnings

jobs:
  fmt:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt

      - name: Check formatting
        run: cargo fmt --all -- --check

  clippy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy

      - name: Cache
        uses: Swatinem/rust-cache@v2

      - name: Clippy
        run: cargo clippy --all-targets --all-features -- -D warnings

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Cache
        uses: Swatinem/rust-cache@v2

      - name: Run tests
        run: cargo test --all-features

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Install cargo-audit
        run: cargo install cargo-audit

      - name: Security audit
        run: cargo audit

      - name: Install cargo-deny
        run: cargo install cargo-deny

      - name: Check licenses and advisories
        run: cargo deny check

  build:
    runs-on: ubuntu-latest
    needs: [fmt, clippy, test]
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Cache
        uses: Swatinem/rust-cache@v2

      - name: Build release
        run: cargo build --release
```

### .github/workflows/ci.yml (Multi-platform)

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, beta]

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}

      - name: Cache
        uses: Swatinem/rust-cache@v2

      - name: Test
        run: cargo test --all-features

  msrv:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install MSRV
        uses: dtolnay/rust-toolchain@1.75  # Your MSRV

      - name: Cache
        uses: Swatinem/rust-cache@v2

      - name: Check MSRV
        run: cargo check --all-features
```

## Security Auditing

### cargo-audit

```bash
# Install
cargo install cargo-audit

# Run
cargo audit

# Generate lockfile if missing and audit
cargo audit --deny warnings

# JSON output
cargo audit --json
```

### cargo-deny Configuration

Create `deny.toml`:

```toml
[advisories]
db-path = "~/.cargo/advisory-db"
db-urls = ["https://github.com/rustsec/advisory-db"]
vulnerability = "deny"
unmaintained = "warn"
yanked = "warn"
notice = "warn"

[licenses]
unlicensed = "deny"
copyleft = "warn"
allow = [
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Zlib",
    "MPL-2.0",
    "Unicode-DFS-2016",
]

[[licenses.clarify]]
name = "ring"
expression = "MIT AND ISC AND OpenSSL"
license-files = [
    { path = "LICENSE", hash = 0xbd0eed23 }
]

[bans]
multiple-versions = "warn"
wildcards = "allow"
highlight = "all"

# Deny specific crates
deny = [
    # Example: { name = "openssl" }
]

# Skip specific crate versions
skip = [
    # Example: { name = "ansi_term", version = "=0.11.0" }
]

[sources]
unknown-registry = "warn"
unknown-git = "warn"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
```

### Run cargo-deny

```bash
# Install
cargo install cargo-deny

# Check all
cargo deny check

# Check specific
cargo deny check advisories
cargo deny check licenses
cargo deny check bans
cargo deny check sources
```

## Coverage

### .github/workflows/coverage.yml

```yaml
name: Coverage

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          components: llvm-tools-preview

      - name: Install cargo-llvm-cov
        uses: taiki-e/install-action@cargo-llvm-cov

      - name: Generate coverage
        run: cargo llvm-cov --all-features --lcov --output-path lcov.info

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: lcov.info
```

## Common Issues and Solutions

### Clippy warnings on dependencies

```rust
// In main.rs or lib.rs
#![allow(clippy::pedantic)] // Top-level allow

// Or allow specific warnings
#![allow(clippy::module_name_repetitions)]
```

### cargo audit false positives

```toml
# In Cargo.toml
[package.metadata.cargo-audit]
ignore = [
    "RUSTSEC-2020-0071",  # Add advisory ID to ignore
]
```

### Slow CI builds

Use the Swatinem/rust-cache action (already in examples above).

For even faster builds:

```yaml
- name: Cache
  uses: Swatinem/rust-cache@v2
  with:
    cache-on-failure: true
    shared-key: "ci"
```

### MSRV testing

Add to Cargo.toml:

```toml
[package]
rust-version = "1.75"  # Minimum Supported Rust Version
```

Then test in CI against that version.

### Cross-compilation

```yaml
jobs:
  build:
    strategy:
      matrix:
        include:
          - target: x86_64-unknown-linux-gnu
            os: ubuntu-latest
          - target: aarch64-unknown-linux-gnu
            os: ubuntu-latest
          - target: x86_64-apple-darwin
            os: macos-latest
          - target: aarch64-apple-darwin
            os: macos-latest

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}

      - name: Build
        run: cargo build --release --target ${{ matrix.target }}
```

## Makefile (Optional)

```makefile
.PHONY: fmt lint test audit build

fmt:
	cargo fmt --all

lint:
	cargo clippy --all-targets --all-features -- -D warnings

test:
	cargo test --all-features

audit:
	cargo audit
	cargo deny check

build:
	cargo build --release

ci: fmt lint test audit build
```

## Recommended Tools

```bash
# Core
rustup component add rustfmt clippy

# Security
cargo install cargo-audit cargo-deny

# Coverage
cargo install cargo-llvm-cov

# Unused dependencies
cargo install cargo-machete

# Outdated dependencies
cargo install cargo-outdated

# Better test output
cargo install cargo-nextest
```
