# Security Tools Comparison

## SAST (Static Application Security Testing)

### CodeQL vs Semgrep

| Feature | CodeQL | Semgrep |
|---------|--------|---------|
| **Provider** | GitHub | Semgrep Inc. |
| **Analysis Depth** | Deep dataflow | Pattern matching |
| **Speed** | Slower (thorough) | Fast |
| **Languages** | 10+ major languages | 30+ languages |
| **Custom Rules** | QL language (steep learning) | YAML (easy) |
| **GitHub Integration** | Native | Via SARIF upload |
| **Free Tier** | Unlimited (public repos) | 10 private repos |
| **False Positives** | Lower (deeper analysis) | Higher (faster) |
| **Best For** | Production security, compliance | Quick feedback, CI |

### When to Use CodeQL

- Public repositories (free unlimited)
- Need deep dataflow analysis
- Compliance requirements (SOC2, etc.)
- Finding complex vulnerabilities
- Enterprise GitHub plans

### When to Use Semgrep

- Fast CI feedback loop
- Custom rules for org patterns
- Multiple languages in one repo
- Quick setup, lower learning curve
- Need real-time scanning

### Example: Same Vulnerability, Different Approaches

SQL Injection detection:

**CodeQL** (deep analysis):
```ql
// Tracks data from source to sink through multiple functions
import javascript

from DataFlow::PathNode source, DataFlow::PathNode sink
where SqlInjection::query(source, sink)
select sink.getNode(), source, sink, "SQL injection from $@.", source.getNode(), "user input"
```

**Semgrep** (pattern matching):
```yaml
rules:
  - id: sql-injection
    patterns:
      - pattern: $DB.query($X + ...)
      - pattern-not: $DB.query($CONST)
    message: Possible SQL injection
    severity: ERROR
```

## SCA (Software Composition Analysis)

### Dependabot vs Snyk vs Renovate

| Feature | Dependabot | Snyk | Renovate |
|---------|------------|------|----------|
| **Provider** | GitHub | Snyk | Mend |
| **Auto PRs** | ✅ | ✅ | ✅ |
| **Vuln Database** | GitHub Advisory | Snyk DB (richer) | OSV + others |
| **Grouping** | Basic | Advanced | Advanced |
| **Monorepo** | Limited | Good | Excellent |
| **Config Location** | .github/ | .snyk | renovate.json |
| **Free Tier** | Unlimited | 200 tests/month | Unlimited |
| **IDE Plugin** | ❌ | ✅ | ❌ |
| **License Scanning** | ❌ | ✅ | ❌ |

### When to Use Dependabot

- GitHub-native experience
- Simple dependency management
- Don't need advanced features
- Already using GitHub ecosystem

### When to Use Snyk

- Need detailed vulnerability info
- License compliance requirements
- IDE integration desired
- Container/IaC scanning needed

### When to Use Renovate

- Complex monorepo setup
- Need advanced grouping/scheduling
- Self-hosted option needed
- GitLab or Bitbucket users

## Secret Scanning

### Gitleaks vs detect-secrets vs TruffleHog

| Feature | Gitleaks | detect-secrets | TruffleHog |
|---------|----------|----------------|------------|
| **Speed** | Fast | Medium | Fast |
| **Git History** | Full scan | Baseline only | Full scan |
| **Pre-commit** | ✅ | ✅ | ✅ |
| **Custom Rules** | TOML | Python plugins | YAML |
| **Language** | Go | Python | Go |
| **CI Integration** | GitHub Action | Manual | GitHub Action |
| **Entropy Detection** | ✅ | ✅ | ✅ |
| **Verification** | Some | ❌ | ✅ (can test if live) |

### When to Use Gitleaks

- Fast Git history scanning
- GitHub Actions integration
- Don't need baseline tracking
- Simple TOML configuration

### When to Use detect-secrets

- Python ecosystem
- Need baseline file to track
- Custom detector plugins
- Already using pre-commit

### When to Use TruffleHog

- Need to verify if secrets are live
- Scanning multiple sources (S3, etc.)
- Prefer Go tooling
- Need detailed output formats

## Container Security

### Trivy vs Grype vs Snyk Container

| Feature | Trivy | Grype | Snyk Container |
|---------|-------|-------|----------------|
| **Speed** | Fast | Fast | Medium |
| **Vuln DB** | Multiple | Anchore | Snyk |
| **OS Packages** | ✅ | ✅ | ✅ |
| **App Deps** | ✅ | ✅ | ✅ |
| **Secrets** | ✅ | ❌ | ❌ |
| **IaC Scanning** | ✅ | ❌ | ✅ (separate) |
| **SBOM** | ✅ | ✅ | ✅ |
| **Free Tier** | Unlimited | Unlimited | 100 tests/month |

### When to Use Trivy

- All-in-one scanning (containers, IaC, secrets)
- Kubernetes environments
- Need SBOM generation
- Free for all use cases

### When to Use Grype

- Already using Anchore ecosystem
- Need specific SBOM format support
- Prefer Anchore vulnerability data

## Pre-commit Frameworks

### Husky vs pre-commit vs lefthook

| Feature | Husky | pre-commit | lefthook |
|---------|-------|------------|----------|
| **Ecosystem** | Node.js | Python | Go |
| **Speed** | Medium | Slower | Fast |
| **Config Format** | Shell scripts | YAML | YAML |
| **Hook Sharing** | npm packages | Git repos | Git repos |
| **Parallel Execution** | With lint-staged | ✅ | ✅ |
| **Skip Mechanism** | Env vars | Env vars | Env vars |

### When to Use Husky

- Node.js/JavaScript projects
- Already using npm/yarn/pnpm
- Want lint-staged integration
- Team familiar with Node tooling

### When to Use pre-commit

- Python projects
- Multi-language repositories
- Want hooks as Git repos
- Need extensive hook ecosystem

### When to Use lefthook

- Need maximum speed
- Go projects
- Don't want Node or Python deps
- Simple YAML configuration

## Recommended Combinations

### Node.js Project (Typical)

```
Pre-commit: Husky + lint-staged
SAST: CodeQL (public) or Semgrep (private)
SCA: Dependabot
Secrets: ESLint no-secrets + Gitleaks
```

### Python Project (Typical)

```
Pre-commit: pre-commit framework
SAST: CodeQL + Bandit
SCA: Dependabot
Secrets: detect-secrets
```

### Enterprise/Compliance

```
Pre-commit: Language-specific
SAST: CodeQL (extended queries)
SCA: Snyk (for license compliance)
Secrets: Gitleaks + GitHub secret scanning
Container: Trivy + Snyk Container
```

### Startup/Fast Iteration

```
Pre-commit: Husky (Node) or lefthook
SAST: Semgrep (fast feedback)
SCA: Dependabot (free, native)
Secrets: Gitleaks
```

## Cost Comparison

| Tool | Free Tier | Paid Starting |
|------|-----------|---------------|
| CodeQL | Unlimited (public) | GitHub Enterprise |
| Semgrep | 10 private repos | $40/dev/month |
| Dependabot | Unlimited | - |
| Snyk | 200 tests/month | $52/dev/month |
| Renovate | Unlimited | Enterprise features |
| Gitleaks | Unlimited | Enterprise license |
| Trivy | Unlimited | Enterprise support |

## Integration Priority

For a new project, add in this order:

1. **Pre-commit hooks** - Immediate feedback, blocks bad commits
2. **CI linting** - Catches what pre-commit misses
3. **Dependabot** - Free, automatic, low effort
4. **Secret scanning** - Critical for security
5. **SAST (CodeQL/Semgrep)** - Deeper analysis
6. **Container scanning** - If using Docker
