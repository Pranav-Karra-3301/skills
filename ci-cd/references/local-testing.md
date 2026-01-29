# Local Testing with Act

## Overview

[Act](https://github.com/nektos/act) is a CLI tool that runs GitHub Actions locally using Docker. This eliminates the commit-push-wait cycle when developing workflows.

**Key benefits:**
- Test workflow changes instantly without pushing
- Debug failed actions locally
- Validate YAML syntax and job dependencies
- Run workflows offline (with cached images)

## Installation

### Check if Act is Installed

```bash
# Check if act is available
act --version

# Check if Docker is running (required)
docker info > /dev/null 2>&1 && echo "Docker is running" || echo "Docker is NOT running"
```

### Installation Commands

```bash
# macOS (Homebrew)
brew install act

# macOS (MacPorts)
sudo port install act

# Windows (Chocolatey)
choco install act-cli

# Windows (Winget)
winget install nektos.act

# Windows (Scoop)
scoop install act

# Linux (script)
curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Linux (Arch)
pacman -S act

# Linux (Nix)
nix-env -i act

# Go install (any platform with Go 1.20+)
go install github.com/nektos/act@latest

# GitHub CLI extension
gh extension install nektos/gh-act
```

### Docker Requirement

Act requires Docker to be installed and running:

```bash
# macOS
brew install --cask docker
# Then launch Docker Desktop

# Linux (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER  # Logout/login required

# Verify Docker
docker run hello-world
```

## First Run - Choosing Runner Image

On first run, act prompts for a default runner image:

| Image | Size | Use Case |
|-------|------|----------|
| **Micro** | ~200 MB | Simple workflows, fast testing |
| **Medium** | ~500 MB | Most workflows (recommended) |
| **Large** | ~17 GB | Full GitHub runner compatibility |

```bash
# First run prompts for image selection
act

# Or specify explicitly
act -P ubuntu-latest=catthehacker/ubuntu:act-latest  # Medium
act -P ubuntu-latest=catthehacker/ubuntu:full-latest  # Large
act -P ubuntu-latest=node:16-bullseye-slim  # Minimal for Node.js
```

**Recommendation:** Start with Medium. Only use Large if you encounter missing tools.

## Basic Usage

### Running Workflows

```bash
# Run all workflows triggered by 'push' event
act

# Run workflows for specific events
act push                    # Push event (default)
act pull_request            # PR event
act workflow_dispatch       # Manual trigger
act schedule               # Scheduled workflows

# List available workflows and jobs
act -l                      # List all
act -l push                 # List push-triggered workflows
act -l pull_request         # List PR-triggered workflows
```

### Running Specific Jobs/Workflows

```bash
# Run a specific job by name
act -j test
act -j build
act -j lint

# Run a specific workflow file
act -W .github/workflows/ci.yml
act -W .github/workflows/test.yml

# Combine: specific job in specific workflow
act -W .github/workflows/ci.yml -j test
```

### Dry Run (No Execution)

```bash
# See what would run without actually running
act -n
act -n pull_request
```

## Working with Secrets

### Passing Secrets

```bash
# Direct value (visible in shell history - use for testing only)
act -s MY_SECRET=mysecretvalue

# Prompt for secret (secure - recommended)
act -s MY_SECRET

# From environment variable
export MY_SECRET=mysecretvalue
act -s MY_SECRET

# From file (.secrets format: KEY=value per line)
act --secret-file .secrets

# GitHub token (using gh CLI)
act -s GITHUB_TOKEN="$(gh auth token)"
```

### Example .secrets File

```bash
# .secrets (add to .gitignore!)
GITHUB_TOKEN=ghp_xxxxxxxxxxxx
NPM_TOKEN=npm_xxxxxxxxxxxx
CODECOV_TOKEN=xxxxxxxx-xxxx-xxxx-xxxx
```

### Passing Variables

```bash
# Repository variables
act --var MY_VAR=value
act --var-file .vars

# Workflow dispatch inputs
act workflow_dispatch --input name=value
act workflow_dispatch --input-file inputs.json
```

## Runner Configuration

### Custom Runner Images

```bash
# Use specific image for a runner
act -P ubuntu-latest=catthehacker/ubuntu:act-22.04
act -P ubuntu-22.04=catthehacker/ubuntu:act-22.04
act -P ubuntu-20.04=catthehacker/ubuntu:act-20.04

# Node.js optimized (smaller, faster)
act -P ubuntu-latest=node:20-bookworm

# Multiple runners
act -P ubuntu-latest=catthehacker/ubuntu:act-latest \
    -P ubuntu-22.04=catthehacker/ubuntu:act-22.04
```

### Self-Hosted Runner (macOS/Windows native)

```bash
# Run on host machine instead of Docker (for platform-specific actions)
act -P macos-latest=-self-hosted
act -P windows-latest=-self-hosted
```

### Container Architecture (Apple Silicon)

```bash
# Force x86_64 architecture on ARM Macs
act --container-architecture linux/amd64

# Or in .actrc
echo "--container-architecture linux/amd64" >> .actrc
```

## Configuration File (.actrc)

Create `.actrc` in project root or home directory:

```bash
# .actrc - one flag per line
-P ubuntu-latest=catthehacker/ubuntu:act-latest
--container-architecture linux/amd64
--action-offline-mode
--artifact-server-path /tmp/artifacts
--secret-file .secrets
```

**Search order:** `$XDG_CONFIG_HOME/act/actrc` → `~/.actrc` → `./.actrc` → CLI args

## Handling Artifacts

```bash
# Enable artifact server (stores artifacts locally)
act --artifact-server-path ./.artifacts

# Artifacts will be saved to .artifacts/ directory
```

## Debugging

### Verbose Output

```bash
# Verbose mode
act -v

# Very verbose (debug)
act -vv
```

### Reuse Containers (Faster Iteration)

```bash
# Keep container running between runs
act --reuse
```

### Skip Specific Steps

In your workflow, use the `ACT` environment variable:

```yaml
steps:
  - name: Deploy (skip locally)
    if: ${{ !env.ACT }}
    run: ./deploy.sh

  - name: Upload coverage (skip locally)
    if: ${{ !env.ACT }}
    uses: codecov/codecov-action@v4
```

### Skip Based on Event Context

```yaml
jobs:
  deploy:
    # Skip this entire job when running with act
    if: ${{ !github.event.act }}
    runs-on: ubuntu-latest
    steps:
      - run: ./deploy.sh
```

## Common Issues & Solutions

### Issue: Docker not running

```bash
# Error: Cannot connect to Docker daemon
# Solution: Start Docker
open -a Docker  # macOS
sudo systemctl start docker  # Linux
```

### Issue: Image pull fails

```bash
# Error: Error response from daemon: pull access denied
# Solution: Use a different image
act -P ubuntu-latest=catthehacker/ubuntu:act-latest
```

### Issue: Action not found

```bash
# Error: Unable to find action
# Solution: Ensure actions are cached or use offline mode
act --action-offline-mode  # Use cached versions
```

### Issue: Out of disk space

```bash
# Large images consume disk space
# Solution: Clean up Docker
docker system prune -a
```

### Issue: ARM Mac architecture issues

```bash
# Error: exec format error
# Solution: Force x86_64 architecture
act --container-architecture linux/amd64
```

### Issue: Secrets not available

```bash
# Error: secret not found
# Solution: Pass secrets explicitly
act -s GITHUB_TOKEN="$(gh auth token)" -s NPM_TOKEN
```

### Issue: Service containers not working

```bash
# act has limited support for service containers
# Workaround: Run services separately
docker run -d --name postgres -e POSTGRES_PASSWORD=test postgres:15
act --network host
```

## Limitations

**What act doesn't fully support:**
- Service containers (limited support)
- GitHub-hosted runner caching (actions/cache)
- Some GitHub context variables
- OIDC tokens
- Large runner tool cache
- Systemd-dependent actions

**Workarounds:**
- Mock external services
- Skip unsupported steps with `if: ${{ !env.ACT }}`
- Use smaller, focused workflows for local testing

## Recommended Workflow

### 1. Quick Validation

```bash
# List jobs to verify YAML is valid
act -l

# Dry run to see execution plan
act -n
```

### 2. Test Specific Job

```bash
# Test just the lint job
act -j lint

# Test just the test job
act -j test -s GITHUB_TOKEN="$(gh auth token)"
```

### 3. Full Workflow Test

```bash
# Run complete CI workflow
act -W .github/workflows/ci.yml

# With secrets
act -W .github/workflows/ci.yml --secret-file .secrets
```

### 4. Test PR Event

```bash
# Simulate pull request
act pull_request

# With specific inputs
act pull_request -e event.json
```

## Integration with Skill Workflow

When using this skill to generate CI/CD configs:

1. **After generating workflows**, offer to test locally:
   ```
   "Would you like to test the workflow locally with act?"
   ```

2. **Check if act is installed**:
   ```bash
   command -v act > /dev/null 2>&1
   ```

3. **If not installed, offer to install**:
   ```
   "act is not installed. Would you like to install it?
   - macOS: brew install act
   - Linux: curl install script
   - Other: See installation options"
   ```

4. **Run validation**:
   ```bash
   act -l  # Verify YAML parses
   act -n  # Dry run
   act -j lint  # Test lint job
   ```

## Example Testing Session

```bash
# 1. Check act is ready
act --version
docker info

# 2. List workflows
act -l
# Output:
# Stage  Job ID  Job name  Workflow name  Events
# 0      lint    lint      CI             push,pull_request
# 1      test    test      CI             push,pull_request
# 2      build   build     CI             push,pull_request

# 3. Dry run
act -n

# 4. Test lint job
act -j lint

# 5. Test with secrets
act -j test -s GITHUB_TOKEN="$(gh auth token)"

# 6. Full workflow
act push --secret-file .secrets
```

## References

- [Act GitHub Repository](https://github.com/nektos/act)
- [Act Documentation](https://nektosact.com)
- [Act Discussions](https://github.com/nektos/act/discussions)
- [VS Code Extension](https://marketplace.visualstudio.com/items?itemName=GitHub.vscode-github-actions)
