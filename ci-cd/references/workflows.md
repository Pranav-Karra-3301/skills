# GitHub Actions Workflow Templates

> **Tip:** Test these workflows locally before pushing using [act](https://github.com/nektos/act). See [local-testing.md](local-testing.md) for setup.

## Core CI Workflow

### Generic CI (language-agnostic structure)

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

# Cancel in-progress runs for the same branch
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      # Add language-specific linting

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      # Add language-specific testing

  build:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4
      # Add language-specific build
```

## CodeQL Security Scanning

### .github/workflows/codeql.yml

```yaml
name: CodeQL

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    # Run weekly on Monday at 00:00 UTC
    - cron: '0 0 * * 1'

permissions:
  actions: read
  contents: read
  security-events: write

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest

    strategy:
      fail-fast: false
      matrix:
        language: ['javascript-typescript']
        # Other options: 'python', 'go', 'java', 'csharp', 'cpp', 'ruby', 'swift'

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: ${{ matrix.language }}
          # Use extended queries for more thorough analysis
          queries: security-extended,security-and-quality

      # For compiled languages, add build steps here
      # - name: Build
      #   run: make

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3
        with:
          category: "/language:${{ matrix.language }}"
```

### CodeQL for Multiple Languages

```yaml
name: CodeQL

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 1'

permissions:
  actions: read
  contents: read
  security-events: write

jobs:
  analyze:
    name: Analyze (${{ matrix.language }})
    runs-on: ubuntu-latest

    strategy:
      fail-fast: false
      matrix:
        include:
          - language: javascript-typescript
            build-mode: none
          - language: python
            build-mode: none
          - language: go
            build-mode: autobuild

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: ${{ matrix.language }}
          queries: security-extended

      - if: matrix.build-mode == 'autobuild'
        name: Autobuild
        uses: github/codeql-action/autobuild@v3

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3
```

## Semgrep Security Scanning

### .github/workflows/semgrep.yml

```yaml
name: Semgrep

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 1'

permissions:
  contents: read
  security-events: write

jobs:
  semgrep:
    runs-on: ubuntu-latest

    container:
      image: semgrep/semgrep

    steps:
      - uses: actions/checkout@v4

      - name: Run Semgrep
        run: semgrep scan --config auto --sarif --output semgrep.sarif
        env:
          SEMGREP_APP_TOKEN: ${{ secrets.SEMGREP_APP_TOKEN }}

      - name: Upload SARIF
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: semgrep.sarif
        if: always()
```

### Semgrep with Custom Rules

```yaml
name: Semgrep

on:
  push:
    branches: [main]
  pull_request:

jobs:
  semgrep:
    runs-on: ubuntu-latest

    container:
      image: semgrep/semgrep

    steps:
      - uses: actions/checkout@v4

      - name: Run Semgrep
        run: |
          semgrep scan \
            --config "p/default" \
            --config "p/security-audit" \
            --config "p/secrets" \
            --config "p/owasp-top-ten" \
            --sarif --output semgrep.sarif

      - name: Upload SARIF
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: semgrep.sarif
```

## Dependency Scanning

### .github/dependabot.yml

```yaml
version: 2

registries:
  npm-npmjs:
    type: npm-registry
    url: https://registry.npmjs.org
    replaces-base: true

updates:
  # Node.js dependencies
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 10
    groups:
      development-dependencies:
        dependency-type: "development"
        update-types:
          - "minor"
          - "patch"
      production-dependencies:
        dependency-type: "production"
        update-types:
          - "patch"
    ignore:
      # Ignore major updates for stability
      - dependency-name: "*"
        update-types: ["version-update:semver-major"]
    commit-message:
      prefix: "chore(deps)"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 5
    commit-message:
      prefix: "ci"

  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    groups:
      python-packages:
        patterns:
          - "*"

  # Docker
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"

  # Go modules
  - package-ecosystem: "gomod"
    directory: "/"
    schedule:
      interval: "weekly"

  # Rust/Cargo
  - package-ecosystem: "cargo"
    directory: "/"
    schedule:
      interval: "weekly"
```

### Dependabot (Minimal)

```yaml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

## Secret Scanning

### .github/workflows/gitleaks.yml

```yaml
name: Gitleaks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read
  security-events: write

jobs:
  gitleaks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run Gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}  # Optional for enterprise features
```

### .gitleaks.toml (Configuration)

```toml
[extend]
useDefault = true

[allowlist]
description = "Allowlist for known false positives"
paths = [
    '''.*_test\.go''',
    '''.*\.md''',
    '''package-lock\.json''',
    '''yarn\.lock''',
]

regexTarget = "match"
regexes = [
    '''EXAMPLE_[A-Z_]+''',
    '''PLACEHOLDER_[A-Z_]+''',
]

[[rules]]
id = "custom-api-key"
description = "Custom API key pattern"
regex = '''(?i)my_service_api_key\s*[=:]\s*['"]?([a-zA-Z0-9]{32,})['"]?'''
secretGroup = 1
entropy = 3.5
```

## Comprehensive Security Workflow

### .github/workflows/security.yml

```yaml
name: Security

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 1'

permissions:
  contents: read
  security-events: write

jobs:
  codeql:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: javascript-typescript
          queries: security-extended

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3

  dependency-review:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4

      - name: Dependency Review
        uses: actions/dependency-review-action@v4
        with:
          fail-on-severity: high

  gitleaks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run Gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  trivy:
    runs-on: ubuntu-latest
    if: hashFiles('Dockerfile') != ''
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: docker build -t app:${{ github.sha }} .

      - name: Run Trivy
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'app:${{ github.sha }}'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy SARIF
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'
```

## Release Workflow

### .github/workflows/release.yml

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: write

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Generate changelog
        id: changelog
        uses: orhun/git-cliff-action@v3
        with:
          config: cliff.toml
          args: --latest --strip header
        env:
          OUTPUT: CHANGELOG.md

      - name: Create release
        uses: softprops/action-gh-release@v2
        with:
          body: ${{ steps.changelog.outputs.content }}
          draft: false
          prerelease: ${{ contains(github.ref, 'alpha') || contains(github.ref, 'beta') || contains(github.ref, 'rc') }}
```

## Workflow Security Best Practices

### Minimal Permissions

```yaml
# At job level
permissions:
  contents: read
  pull-requests: write  # Only if needed

# Or at workflow level (more restrictive)
permissions: read-all
```

### Pin Actions to SHA

```yaml
# Instead of
- uses: actions/checkout@v4

# Use SHA for security-critical workflows
- uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11  # v4.1.1
```

### Avoid Script Injection

```yaml
# Bad - vulnerable to injection
- run: echo "${{ github.event.pull_request.title }}"

# Good - use environment variable
- env:
    PR_TITLE: ${{ github.event.pull_request.title }}
  run: echo "$PR_TITLE"
```

### Use GITHUB_TOKEN Sparingly

```yaml
# Only grant necessary permissions
permissions:
  contents: read

# Never expose GITHUB_TOKEN in logs
- run: |
    # Good
    git push https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/...

    # Bad - token may be logged
    echo ${{ secrets.GITHUB_TOKEN }}
```

### Restrict Workflow Triggers

```yaml
# Limit to specific branches
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
    types: [opened, synchronize]  # Not edited, labeled, etc.
```

## Branch Protection Recommendations

Configure via GitHub Settings → Branches → Add rule:

- **Require pull request reviews**: 1+ approvals
- **Dismiss stale reviews**: When new commits pushed
- **Require status checks**: CI workflow must pass
- **Require branches to be up to date**
- **Require signed commits**: Optional but recommended
- **Require linear history**: Prevents merge commits
- **Include administrators**: Apply rules to admins too
- **Restrict pushes**: Only allow specific users/teams

## Local Testing with Act

Test workflows locally before pushing using [act](https://github.com/nektos/act):

```bash
# Install act
brew install act  # macOS
# or: curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# List workflows (validates YAML)
act -l

# Dry run
act -n

# Test specific job
act -j lint
act -j test

# Full workflow
act -W .github/workflows/ci.yml

# With secrets
act -s GITHUB_TOKEN="$(gh auth token)"
```

### Skipping Steps Locally

Add to steps that shouldn't run in act:

```yaml
- name: Deploy (skip locally)
  if: ${{ !env.ACT }}
  run: ./deploy.sh
```

See [local-testing.md](local-testing.md) for comprehensive act usage, troubleshooting, and configuration.
