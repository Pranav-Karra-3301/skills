---
name: ci-cd
description: |
  Security-hardening skill for CI/CD pipelines and pre-commit hooks. Scans codebases to detect missing security infrastructure and generates configurations. Use when:
  (1) Setting up CI/CD for a new project ("add CI", "setup GitHub Actions", "create workflow")
  (2) Adding pre-commit hooks ("add husky", "setup pre-commit", "add git hooks")
  (3) Security hardening ("add SAST", "setup CodeQL", "add dependabot", "security scan")
  (4) Auditing existing CI/CD ("check my workflows", "what security am I missing")
  (5) Adding commit standards ("add commitlint", "enforce conventional commits")
  Triggers: "CI/CD", "GitHub Actions", "pre-commit", "husky", "CodeQL", "Semgrep", "dependabot", "SAST", "security scanning", "lint-staged", "commitlint", "git hooks"
---

# CI/CD Security Hardening

## Overview

This skill detects your project's technology stack, identifies missing CI/CD security infrastructure, and generates production-ready configurations. Targets vibecoded/AI-generated projects that often lack proper security tooling.

**Key capabilities:**
- Stack detection (Node.js, Python, Go, Rust, multi-language)
- Pre-commit hook setup (Husky, lint-staged, pre-commit framework)
- GitHub Actions workflows (CI, CodeQL, Semgrep, dependency scanning)
- Commit message standards (Conventional Commits)
- Gap analysis against security best practices
- **Local workflow testing with [act](https://github.com/nektos/act)** (macOS/Linux)

## Core Workflow

### Phase 1: Detection

Scan the codebase to identify:

1. **Language/Framework Detection**
   ```
   Check for: package.json, requirements.txt, pyproject.toml, go.mod, Cargo.toml
   Also: tsconfig.json, setup.py, poetry.lock, Pipfile
   ```

2. **Existing CI/CD Infrastructure**
   ```
   Look for:
   - .github/workflows/*.yml
   - .gitlab-ci.yml
   - Jenkinsfile
   - .circleci/config.yml
   - .travis.yml
   ```

3. **Existing Pre-commit Setup**
   ```
   Look for:
   - .husky/ directory (Node.js)
   - .pre-commit-config.yaml (Python/universal)
   - .git/hooks/ (manual hooks)
   - lint-staged config in package.json
   ```

4. **Existing Security Tools**
   ```
   Check for:
   - .github/dependabot.yml
   - .github/workflows/codeql*.yml
   - semgrep.yml or .semgrep/
   - .gitleaks.toml
   - eslint-plugin-security (Node.js)
   - bandit.yml (Python)
   ```

### Phase 2: Gap Analysis

Compare detected infrastructure against security checklist:

#### Pre-commit Checklist
- [ ] Pre-commit hooks installed and configured
- [ ] Linting runs on staged files
- [ ] Formatting runs on staged files
- [ ] Type checking (if applicable)
- [ ] Secret detection (detect-secrets, gitleaks)
- [ ] Commit message validation (commitlint/commitizen)

#### CI/CD Checklist
- [ ] Build workflow exists
- [ ] Tests run in CI
- [ ] Linting runs in CI
- [ ] SAST scanning (CodeQL or Semgrep)
- [ ] Dependency scanning (Dependabot or Snyk)
- [ ] Secret scanning enabled
- [ ] Branch protection configured
- [ ] Concurrency controls (prevent duplicate runs)

#### Security Checklist
- [ ] Security-focused linting enabled
- [ ] OWASP dependency check or equivalent
- [ ] License compliance scanning
- [ ] Container scanning (if Dockerized)

See [references/checklists.md](references/checklists.md) for detailed checklists.

### Phase 3: Configuration Generation

Based on detected stack, generate appropriate configurations:

| Stack | Pre-commit | CI Workflow | Security |
|-------|------------|-------------|----------|
| Node.js/TypeScript | Husky + lint-staged | [node.md](references/node.md) | ESLint security plugin |
| Python | pre-commit framework | [python.md](references/python.md) | Bandit + Ruff |
| Go | pre-commit + golangci-lint | [go.md](references/go.md) | gosec |
| Rust | cargo-audit + clippy | [rust.md](references/rust.md) | cargo-audit |
| Multi-language | pre-commit framework | Combined workflow | Per-language tools |

**Important principles:**
- Prefer tools already in the ecosystem (ESLint for Node, Ruff for Python)
- Keep configurations minimal but secure
- Use caching in CI for performance
- Add concurrency controls to prevent duplicate workflow runs

### Phase 4: Verification

After generating configurations:

1. **Validate YAML syntax** - Run `yamllint` or equivalent
2. **Test pre-commit hooks** - Run `pre-commit run --all-files` or equivalent
3. **Verify CI workflow** - Check for common issues:
   - Missing checkout step
   - Wrong Node/Python version
   - Missing environment variables
   - Incorrect file paths
4. **Run linters** - Ensure no immediate failures on existing code
5. **Test workflows locally with act** (if available) - See [Local Testing](#local-testing-with-act)

### Phase 5: Local Testing with Act (Optional)

[Act](https://github.com/nektos/act) runs GitHub Actions locally using Docker, eliminating the commit-push-wait cycle.

**Check if act is available:**
```bash
# Check for act
command -v act > /dev/null 2>&1 && echo "act is installed" || echo "act is NOT installed"

# Check Docker is running (required)
docker info > /dev/null 2>&1 && echo "Docker is running" || echo "Docker is NOT running"
```

**If act is not installed, ask the user:**
> "act is not installed. Would you like to install it to test workflows locally?
> - macOS: `brew install act`
> - Linux: `curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash`
> - Other options: See [local-testing.md](references/local-testing.md)"

**Quick validation with act:**
```bash
# List workflows (validates YAML)
act -l

# Dry run (see execution plan without running)
act -n

# Test specific job
act -j lint
act -j test

# Full workflow test
act -W .github/workflows/ci.yml

# With secrets
act -s GITHUB_TOKEN="$(gh auth token)" --secret-file .secrets
```

See [references/local-testing.md](references/local-testing.md) for comprehensive act usage.

## Language-Specific Setup

### Node.js / TypeScript

See [references/node.md](references/node.md) for complete configuration.

**Quick setup:**
```bash
# Install dependencies
npm install -D husky lint-staged @commitlint/cli @commitlint/config-conventional

# Initialize Husky (v9+)
npx husky init

# Configure lint-staged in package.json
```

**Generated files:**
- `.husky/pre-commit` - Runs lint-staged
- `.husky/commit-msg` - Runs commitlint
- `commitlint.config.js` - Conventional commits
- `package.json` - lint-staged configuration

### Python

See [references/python.md](references/python.md) for complete configuration.

**Quick setup:**
```bash
# Install pre-commit
pip install pre-commit

# Create config
# (skill generates .pre-commit-config.yaml)

# Install hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

**Generated files:**
- `.pre-commit-config.yaml` - All hooks configuration
- `pyproject.toml` - Tool configurations (Ruff, mypy)

### Go

See [references/go.md](references/go.md) for complete configuration.

**Quick setup:**
```bash
# Install golangci-lint
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Install pre-commit
pip install pre-commit
pre-commit install
```

**Generated files:**
- `.golangci.yml` - Linter configuration with gosec
- `.pre-commit-config.yaml` - Go-specific hooks

### Rust

See [references/rust.md](references/rust.md) for complete configuration.

**Quick setup:**
```bash
# Install cargo-audit
cargo install cargo-audit

# Configure in CI
# (skill generates workflow)
```

## GitHub Actions Workflows

See [references/workflows.md](references/workflows.md) for complete templates.

### Core Workflows to Generate

1. **ci.yml** - Build, lint, test
   - Runs on push and PR
   - Caching for dependencies
   - Concurrency controls
   - Matrix builds if needed

2. **codeql.yml** - Security scanning
   - Scheduled + on push to main
   - Extended security queries
   - Language-specific configuration

3. **dependabot.yml** - Dependency updates
   - Weekly schedule
   - Package ecosystem detection
   - GitHub Actions updates

### Security Tools Comparison

See [references/security-tools.md](references/security-tools.md) for detailed comparison.

| Tool | Type | Best For | Free Tier |
|------|------|----------|-----------|
| CodeQL | SAST | Deep analysis, GitHub native | Unlimited for public repos |
| Semgrep | SAST | Fast, customizable rules | 20 rules/month free |
| Dependabot | SCA | GitHub native, automatic PRs | Free |
| Snyk | SCA | Detailed vuln info, IDE plugins | 200 tests/month |
| Gitleaks | Secrets | Fast, pre-commit friendly | Free |

## Common Tasks

### "Set up CI/CD for this project"
1. Run detection phase
2. Show gap analysis results
3. Ask about preferences (CodeQL vs Semgrep, etc.)
4. Generate all configurations
5. Provide setup commands
6. **Offer to test locally with act** (if Docker available)

### "Add pre-commit hooks"
1. Detect language/framework
2. Generate appropriate hook configuration
3. Provide installation commands
4. Test hooks work

### "What security am I missing?"
1. Run full detection phase
2. Compare against checklists
3. Prioritize gaps by severity
4. Recommend specific tools for each gap

### "Add Dependabot"
1. Detect package ecosystems
2. Generate `.github/dependabot.yml`
3. Include github-actions ecosystem
4. Set appropriate schedule

### "Enforce conventional commits"
1. Detect language for tooling choice
2. Generate commitlint config (Node) or commitizen config (Python)
3. Add commit-msg hook
4. Provide examples of valid commits

### "Test my workflows locally"
1. Check if act is installed (`command -v act`)
2. If not installed, offer installation options for user's platform
3. Check if Docker is running
4. Run `act -l` to list and validate workflows
5. Run `act -n` for dry run
6. Test specific jobs: `act -j lint`, `act -j test`
7. Help debug any failures

## When to Ask the User

Ask before proceeding when:

- **Tool preference**: "CodeQL or Semgrep for SAST? CodeQL is deeper, Semgrep is faster."
- **Strictness level**: "Strict or lenient linting rules? Strict may require fixing existing code."
- **Existing code issues**: "Pre-commit hooks will fail on current code. Fix first or bypass initially?"
- **Multiple ecosystems**: "Detected Node.js and Python. Set up hooks for both?"
- **Breaking changes**: "Adding type checking will require changes. Proceed?"
- **Local testing setup**: "act is not installed. Would you like to install it to test workflows locally?"
- **Docker not running**: "Docker is required for act. Would you like me to help start it?"

## Quality Checklist

Before marking setup complete:

- [ ] Pre-commit hooks run successfully
- [ ] CI workflow passes on current code
- [ ] SAST scanning configured and running
- [ ] Dependency scanning enabled
- [ ] Commit message validation works
- [ ] No secrets in repository
- [ ] Branch protection recommended (if not set)
- [ ] All generated YAML is valid
- [ ] Workflows tested locally with act (if available)
