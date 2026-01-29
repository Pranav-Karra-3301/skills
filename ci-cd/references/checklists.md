# CI/CD Security Checklists

## Pre-commit Checklist

### Essential (All Projects)

- [ ] **Linting** - Code style and basic errors
  - Node: ESLint
  - Python: Ruff
  - Go: golangci-lint
  - Rust: clippy

- [ ] **Formatting** - Consistent code style
  - Node: Prettier
  - Python: Ruff format (or Black)
  - Go: gofmt
  - Rust: rustfmt

- [ ] **Secret Detection** - Prevent credential commits
  - Node: eslint-plugin-no-secrets
  - Python: detect-secrets
  - Universal: Gitleaks

- [ ] **File Hygiene**
  - Trailing whitespace removal
  - End-of-file newline
  - No large files (>1MB typically)
  - No merge conflict markers

### Recommended

- [ ] **Type Checking** (if using typed language)
  - TypeScript: `tsc --noEmit`
  - Python: mypy
  - Go: Built-in

- [ ] **Commit Message Validation**
  - Node: commitlint
  - Python: commitizen
  - Universal: conventional-pre-commit

- [ ] **Branch Protection**
  - No direct commits to main/master
  - Use: pre-commit-hooks `no-commit-to-branch`

### Optional (Based on Project)

- [ ] **Security Linting**
  - Node: eslint-plugin-security
  - Python: Bandit
  - Go: gosec

- [ ] **Dependency Validation**
  - Lock file freshness
  - License compliance (cargo-deny, etc.)

- [ ] **Tests** (as pre-push hook, not pre-commit)
  - Faster feedback loop
  - Prevent broken pushes

## CI/CD Checklist

### Build & Test (Every PR)

- [ ] **Checkout** - `actions/checkout@v4`

- [ ] **Setup Runtime** - With caching
  - Node: `actions/setup-node@v4` with `cache: 'npm'`
  - Python: `actions/setup-python@v5` with `cache: 'pip'`
  - Go: `actions/setup-go@v5` with `cache: true`
  - Rust: `dtolnay/rust-toolchain` + `Swatinem/rust-cache`

- [ ] **Install Dependencies** - With lock file
  - `npm ci` (not `npm install`)
  - `pip install -e ".[dev]"` with caching
  - `go mod download`
  - `cargo build` (caches automatically)

- [ ] **Lint** - Run full linting (not just staged files)

- [ ] **Type Check** - If applicable

- [ ] **Test** - With coverage
  - Coverage reporting to Codecov/Coveralls

- [ ] **Build** - Verify compilation succeeds

### Concurrency & Performance

- [ ] **Concurrency Controls**
  ```yaml
  concurrency:
    group: ${{ github.workflow }}-${{ github.ref }}
    cancel-in-progress: true
  ```

- [ ] **Dependency Caching** - Language-specific caching enabled

- [ ] **Matrix Builds** - If testing multiple versions
  - Only include versions you actually support
  - Don't test every combination unless necessary

- [ ] **Job Dependencies** - Proper `needs:` ordering
  - Lint → Test → Build (typically)
  - Security can run in parallel

### Permissions

- [ ] **Minimal Permissions**
  ```yaml
  permissions:
    contents: read
  ```

- [ ] **Explicit Permissions** - Don't rely on defaults

## Security Checklist

### SAST (Static Analysis)

- [ ] **CodeQL or Semgrep** configured
  - Scheduled weekly run
  - Runs on PRs to main
  - Extended/security queries enabled

- [ ] **SARIF Upload** - Results visible in Security tab

- [ ] **Language Coverage** - All languages in repo covered

### SCA (Dependency Scanning)

- [ ] **Dependabot** enabled
  - Package ecosystem(s) covered
  - GitHub Actions updates included
  - Weekly schedule

- [ ] **Dependency Review** (for PRs)
  ```yaml
  - uses: actions/dependency-review-action@v4
    with:
      fail-on-severity: high
  ```

- [ ] **License Compliance** (if required)
  - cargo-deny for Rust
  - Snyk for comprehensive

### Secret Scanning

- [ ] **GitHub Secret Scanning** enabled (repo settings)

- [ ] **Gitleaks** in CI
  - Full history scan on schedule
  - Incremental on PRs

- [ ] **Pre-commit Secret Detection**
  - Prevents accidental commits
  - Baseline file for false positives

### Container Security (if applicable)

- [ ] **Image Scanning**
  - Trivy or Grype
  - Scan before push to registry

- [ ] **Base Image Updates**
  - Dependabot for Dockerfile
  - Or Renovate

- [ ] **Minimal Base Images**
  - Use distroless/alpine when possible
  - No unnecessary packages

## Repository Settings Checklist

### Branch Protection (GitHub Settings → Branches)

- [ ] **Require PR reviews**
  - At least 1 approval
  - Dismiss stale reviews

- [ ] **Require status checks**
  - CI workflow must pass
  - Security checks must pass

- [ ] **Require up-to-date branches**
  - Force rebase before merge

- [ ] **No force pushes** to protected branches

- [ ] **Include administrators** in rules

### Security Settings (GitHub Settings → Security)

- [ ] **Dependabot alerts** enabled

- [ ] **Dependabot security updates** enabled

- [ ] **Secret scanning** enabled

- [ ] **Code scanning** configured (CodeQL)

### Repository Settings

- [ ] **Default branch** is `main` (not `master`)

- [ ] **Delete head branches** after merge

- [ ] **Only allow squash merging** (optional but clean)

## Gap Analysis Template

Use this template to assess a project's CI/CD security:

```markdown
## Project: [Name]

### Stack Detected
- [ ] Language: ________
- [ ] Framework: ________
- [ ] Package Manager: ________

### Pre-commit Status
| Check | Status | Tool |
|-------|--------|------|
| Linting | ⬜ Missing / ✅ Present | ________ |
| Formatting | ⬜ Missing / ✅ Present | ________ |
| Secret Detection | ⬜ Missing / ✅ Present | ________ |
| Commit Messages | ⬜ Missing / ✅ Present | ________ |
| Type Checking | ⬜ Missing / ✅ Present / N/A | ________ |

### CI/CD Status
| Check | Status | Location |
|-------|--------|----------|
| CI Workflow | ⬜ Missing / ✅ Present | ________ |
| Tests in CI | ⬜ Missing / ✅ Present | ________ |
| SAST | ⬜ Missing / ✅ Present | ________ |
| SCA | ⬜ Missing / ✅ Present | ________ |
| Secret Scanning | ⬜ Missing / ✅ Present | ________ |

### Security Status
| Check | Status |
|-------|--------|
| GitHub Secret Scanning | ⬜ Off / ✅ On |
| Dependabot Alerts | ⬜ Off / ✅ On |
| Branch Protection | ⬜ Off / ✅ On |

### Priority Recommendations
1. ________
2. ________
3. ________
```

## Quick Setup Commands

### Node.js

```bash
# Pre-commit
npm install -D husky lint-staged @commitlint/cli @commitlint/config-conventional
npx husky init
echo "npx lint-staged" > .husky/pre-commit
echo "npx commitlint --edit \$1" > .husky/commit-msg

# Security
npm install -D eslint-plugin-security eslint-plugin-no-secrets
```

### Python

```bash
# Pre-commit
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg

# Create .pre-commit-config.yaml (see python.md)
```

### Go

```bash
# Pre-commit
pip install pre-commit
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
pre-commit install

# Create .golangci.yml (see go.md)
```

### Rust

```bash
# Pre-commit
pip install pre-commit
cargo install cargo-audit cargo-deny
pre-commit install

# Create deny.toml (see rust.md)
```
