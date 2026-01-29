# Python CI/CD Configuration

## Pre-commit Setup

### Installation

```bash
# Install pre-commit
pip install pre-commit

# Or with pipx (recommended for CLI tools)
pipx install pre-commit

# Install hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

### .pre-commit-config.yaml (Comprehensive)

```yaml
repos:
  # General hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
        args: [--unsafe]  # Allow custom tags
      - id: check-json
      - id: check-toml
      - id: check-added-large-files
        args: [--maxkb=1000]
      - id: check-merge-conflict
      - id: detect-private-key
      - id: no-commit-to-branch
        args: [--branch, main, --branch, master]

  # Ruff - Fast Python linter and formatter
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  # Type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies:
          - types-requests
          - types-PyYAML
        args: [--ignore-missing-imports]

  # Security scanning
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.9
    hooks:
      - id: bandit
        args: [-c, pyproject.toml]
        additional_dependencies: ["bandit[toml]"]

  # Secret detection
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
        args: [--baseline, .secrets.baseline]

  # Commit message validation
  - repo: https://github.com/commitizen-tools/commitizen
    rev: v3.27.0
    hooks:
      - id: commitizen
      - id: commitizen-branch
        stages: [push]
```

### .pre-commit-config.yaml (Minimal)

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: detect-private-key

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.9
    hooks:
      - id: bandit
        args: [-r, src, -ll]  # -ll = medium+ severity
```

## pyproject.toml Configuration

```toml
[project]
name = "myproject"
version = "0.1.0"
requires-python = ">=3.10"

[tool.ruff]
target-version = "py310"
line-length = 88
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "S",      # flake8-bandit (security)
    "T20",    # flake8-print
    "SIM",    # flake8-simplify
    "PTH",    # flake8-use-pathlib
]
ignore = [
    "E501",   # line too long (handled by formatter)
    "S101",   # assert usage (ok in tests)
]

[tool.ruff.lint.per-file-ignores]
"tests/**/*" = ["S101", "PLR2004"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[tool.bandit]
exclude_dirs = ["tests", "venv", ".venv"]
skips = ["B101"]  # Skip assert warnings

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-ra -q --strict-markers"

[tool.coverage.run]
source = ["src"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]

[tool.commitizen]
name = "cz_conventional_commits"
tag_format = "v$version"
version_scheme = "semver"
version_provider = "pep621"
update_changelog_on_bump = true
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

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Install dependencies
        run: uv sync --dev

      - name: Run Ruff linter
        run: uv run ruff check .

      - name: Run Ruff formatter
        run: uv run ruff format --check .

      - name: Run mypy
        run: uv run mypy src

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install bandit
        run: pip install bandit[toml]

      - name: Run Bandit
        run: bandit -c pyproject.toml -r src

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Install dependencies
        run: uv sync --dev

      - name: Run tests
        run: uv run pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: matrix.python-version == '3.12'
        with:
          files: coverage.xml
```

### .github/workflows/ci.yml (pip variant)

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
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Lint with Ruff
        run: |
          ruff check .
          ruff format --check .

      - name: Type check with mypy
        run: mypy src

      - name: Security check with Bandit
        run: bandit -c pyproject.toml -r src

      - name: Test with pytest
        run: pytest --cov=src
```

## Secret Scanning Setup

### Initialize detect-secrets baseline

```bash
# Create baseline (tracks existing "secrets" to ignore)
detect-secrets scan > .secrets.baseline

# Audit baseline (mark false positives)
detect-secrets audit .secrets.baseline
```

### .secrets.baseline example

```json
{
  "version": "1.5.0",
  "plugins_used": [
    {"name": "ArtifactoryDetector"},
    {"name": "AWSKeyDetector"},
    {"name": "AzureStorageKeyDetector"},
    {"name": "BasicAuthDetector"},
    {"name": "CloudantDetector"},
    {"name": "GitHubTokenDetector"},
    {"name": "JwtTokenDetector"},
    {"name": "PrivateKeyDetector"},
    {"name": "SlackDetector"},
    {"name": "StripeDetector"}
  ],
  "filters_used": [
    {"path": "detect_secrets.filters.allowlist.is_line_allowlisted"},
    {"path": "detect_secrets.filters.heuristic.is_potential_uuid"},
    {"path": "detect_secrets.filters.heuristic.is_likely_id_string"}
  ],
  "results": {}
}
```

## Common Issues and Solutions

### pre-commit hooks slow

```bash
# Run only on changed files (default behavior)
pre-commit run

# Skip slow hooks during development
SKIP=mypy pre-commit run

# Run specific hook only
pre-commit run ruff
```

### mypy missing stubs

```bash
# Install type stubs
pip install types-requests types-PyYAML types-redis

# Or ignore missing imports in mypy config
[tool.mypy]
ignore_missing_imports = true
```

### Bandit false positives

```python
# Inline ignore
subprocess.run(cmd, shell=True)  # nosec B602

# Or in pyproject.toml
[tool.bandit]
skips = ["B602"]  # subprocess with shell=True
```

### Ruff vs Black/isort

Ruff replaces both Black and isort. Remove them if present:

```bash
# Remove old tools
pip uninstall black isort flake8

# Ruff handles all formatting and linting
```

## Recommended Dependencies

```toml
# pyproject.toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "mypy>=1.10",
    "ruff>=0.5",
    "bandit[toml]>=1.7",
    "pre-commit>=3.7",
]
```

Or with requirements-dev.txt:

```
pytest>=8.0
pytest-cov>=5.0
mypy>=1.10
ruff>=0.5
bandit[toml]>=1.7
pre-commit>=3.7
detect-secrets>=1.5
commitizen>=3.27
```
