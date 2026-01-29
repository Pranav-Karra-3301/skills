# Go CI/CD Configuration

## Pre-commit Setup

### Installation

```bash
# Install pre-commit (requires Python)
pip install pre-commit

# Install golangci-lint
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Or via brew on macOS
brew install golangci-lint

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
      - id: check-json
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: detect-private-key

  - repo: https://github.com/golangci/golangci-lint
    rev: v1.59.1
    hooks:
      - id: golangci-lint
        args: [--fix]

  - repo: https://github.com/dnephin/pre-commit-golang
    rev: v0.5.1
    hooks:
      - id: go-fmt
      - id: go-imports
        args: [-local, github.com/yourorg]
      - id: go-mod-tidy
      - id: go-build
      - id: go-unit-tests

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

## golangci-lint Configuration

### .golangci.yml (Comprehensive)

```yaml
run:
  timeout: 5m
  issues-exit-code: 1
  tests: true
  modules-download-mode: readonly

output:
  formats:
    - format: colored-line-number
  print-issued-lines: true
  print-linter-name: true
  sort-results: true

linters:
  enable:
    # Default linters
    - errcheck
    - gosimple
    - govet
    - ineffassign
    - staticcheck
    - unused

    # Security
    - gosec          # Security scanner
    - bodyclose      # HTTP response body close
    - noctx          # HTTP requests without context

    # Bugs
    - durationcheck  # Duration multiplication
    - exportloopref  # Loop variable capture (pre-Go 1.22)
    - makezero       # Make with non-zero length
    - nilerr         # Nil error returns
    - sqlclosecheck  # SQL rows close

    # Style
    - gofmt
    - goimports
    - gocritic
    - revive
    - misspell

    # Complexity
    - cyclop         # Cyclomatic complexity
    - gocognit       # Cognitive complexity
    - funlen         # Function length

    # Performance
    - prealloc       # Slice preallocation

    # Error handling
    - errname        # Error naming
    - errorlint      # Error wrapping

linters-settings:
  gosec:
    excludes:
      - G104  # Audit errors not checked (too noisy for some codebases)
    config:
      global:
        audit: enabled

  govet:
    enable-all: true
    disable:
      - fieldalignment  # Often too noisy

  goimports:
    local-prefixes: github.com/yourorg

  gocritic:
    enabled-tags:
      - diagnostic
      - style
      - performance
      - experimental
    disabled-checks:
      - hugeParam  # Often too strict

  revive:
    rules:
      - name: var-naming
        disabled: false
      - name: exported
        disabled: false
      - name: error-return
        disabled: false
      - name: error-naming
        disabled: false
      - name: increment-decrement
        disabled: false

  cyclop:
    max-complexity: 15

  gocognit:
    min-complexity: 20

  funlen:
    lines: 100
    statements: 50

issues:
  exclude-rules:
    # Exclude some linters from running on test files
    - path: _test\.go
      linters:
        - funlen
        - gocognit
        - cyclop
        - errcheck

    # Exclude some rules for generated files
    - path: \.pb\.go
      linters:
        - all

    # Allow globals in main and tests
    - path: main\.go
      linters:
        - gochecknoglobals

    - path: _test\.go
      linters:
        - gochecknoglobals

  max-issues-per-linter: 50
  max-same-issues: 3
```

### .golangci.yml (Minimal with Security)

```yaml
run:
  timeout: 3m

linters:
  enable:
    - gosec       # Security
    - bodyclose   # HTTP body close
    - errcheck    # Error checking
    - govet       # Go vet
    - staticcheck # Static analysis
    - gofmt       # Formatting
    - goimports   # Import organization
    - misspell    # Spelling

linters-settings:
  gosec:
    excludes: []
    config:
      global:
        audit: enabled

issues:
  exclude-rules:
    - path: _test\.go
      linters:
        - errcheck
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

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Go
        uses: actions/setup-go@v5
        with:
          go-version: '1.22'
          cache: true

      - name: Run golangci-lint
        uses: golangci/golangci-lint-action@v6
        with:
          version: latest
          args: --timeout=5m

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Go
        uses: actions/setup-go@v5
        with:
          go-version: '1.22'
          cache: true

      - name: Run tests
        run: go test -race -coverprofile=coverage.out -covermode=atomic ./...

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.out

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Go
        uses: actions/setup-go@v5
        with:
          go-version: '1.22'
          cache: true

      - name: Run gosec
        uses: securego/gosec@master
        with:
          args: -exclude-generated ./...

      - name: Run govulncheck
        run: |
          go install golang.org/x/vuln/cmd/govulncheck@latest
          govulncheck ./...

  build:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Go
        uses: actions/setup-go@v5
        with:
          go-version: '1.22'
          cache: true

      - name: Build
        run: go build -v ./...
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
        go-version: ['1.21', '1.22']

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Go
        uses: actions/setup-go@v5
        with:
          go-version: ${{ matrix.go-version }}
          cache: true

      - name: Test
        run: go test -race ./...
```

## Security Scanning with gosec

### Run gosec manually

```bash
# Install
go install github.com/securego/gosec/v2/cmd/gosec@latest

# Run
gosec ./...

# With specific rules
gosec -include=G101,G102,G103 ./...

# Exclude rules
gosec -exclude=G104 ./...

# Output formats
gosec -fmt=json -out=results.json ./...
gosec -fmt=sarif -out=results.sarif ./...
```

### Common gosec rules

| Rule | Description |
|------|-------------|
| G101 | Hardcoded credentials |
| G102 | Bind to all interfaces |
| G103 | Audit the use of unsafe block |
| G104 | Audit errors not checked |
| G107 | URL provided to HTTP request as taint input |
| G108 | Profiling endpoint enabled |
| G109 | Integer overflow conversion |
| G110 | Potential DoS vulnerability (decompression bombs) |
| G201 | SQL query construction using format string |
| G202 | SQL query construction using string concatenation |
| G203 | Use of unescaped data in HTML templates |
| G204 | Audit use of command execution |
| G301 | Poor file permissions |
| G302 | Poor file permissions on file creation |
| G303 | Creating tempfile using predictable path |
| G304 | File path provided as taint input |
| G305 | File traversal when extracting zip archive |
| G306 | Poor file permissions for WriteFile |
| G401 | Detect the usage of DES, RC4, MD5 or SHA1 |
| G402 | TLS InsecureSkipVerify |
| G403 | Ensure minimum RSA key length of 2048 bits |
| G404 | Insecure random number source |
| G501 | Import blacklist: crypto/md5 |
| G502 | Import blacklist: crypto/des |
| G503 | Import blacklist: crypto/rc4 |
| G504 | Import blacklist: net/http/cgi |
| G505 | Import blacklist: crypto/sha1 |
| G601 | Implicit memory aliasing of items in range (pre-1.22) |

## Vulnerability Checking

### govulncheck

```bash
# Install
go install golang.org/x/vuln/cmd/govulncheck@latest

# Run
govulncheck ./...

# JSON output
govulncheck -json ./...
```

## Common Issues and Solutions

### golangci-lint slow

```yaml
# .golangci.yml
run:
  timeout: 10m  # Increase timeout
  skip-dirs:
    - vendor
    - third_party
```

### gosec false positives

```go
// Inline ignore
password := os.Getenv("DB_PASSWORD") // #nosec G101

// Or with specific rule
// #nosec G104 - intentionally ignoring error
_ = file.Close()
```

### Export loop variable (Go < 1.22)

```go
// Bad (captured loop variable)
for _, item := range items {
    go func() {
        process(item) // Bug!
    }()
}

// Good
for _, item := range items {
    item := item // Shadow variable
    go func() {
        process(item)
    }()
}

// Go 1.22+: Fixed by default
```

### Cyclomatic complexity

Split complex functions:

```go
// Instead of one large function with many branches
func processRequest(r *Request) error {
    if err := validateRequest(r); err != nil {
        return err
    }
    if err := authorizeRequest(r); err != nil {
        return err
    }
    return executeRequest(r)
}
```

## Makefile (Optional)

```makefile
.PHONY: lint test build security

lint:
	golangci-lint run

test:
	go test -race -cover ./...

build:
	go build -v ./...

security:
	gosec ./...
	govulncheck ./...

ci: lint test security build
```
