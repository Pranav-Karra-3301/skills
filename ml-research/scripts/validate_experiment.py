#!/usr/bin/env python3
"""
ML Experiment Validation Script

Checks for common ML mistakes and best practice violations:
- Reproducibility issues (seeds, determinism)
- Data leakage risks
- Training setup problems
- Memory/performance issues
- Logging and documentation gaps
"""

import ast
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Issue:
    """Represents a detected issue."""
    category: str
    severity: str  # "error", "warning", "info"
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    suggestion: Optional[str] = None


class ExperimentValidator:
    """Validates ML experiment code and configuration."""

    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.issues: List[Issue] = []
        self.python_files: List[Path] = []
        self.config_files: List[Path] = []

    def find_files(self):
        """Find relevant files in the project."""
        # Exclude common non-project directories
        exclude_dirs = {
            ".git", "__pycache__", ".venv", "venv", "env",
            "node_modules", ".eggs", "*.egg-info", "build", "dist",
            "outputs", "experiments", "data", "checkpoints", "wandb"
        }

        for py_file in self.project_path.rglob("*.py"):
            if not any(ex in str(py_file) for ex in exclude_dirs):
                self.python_files.append(py_file)

        for config_ext in ["*.yaml", "*.yml", "*.json", "*.toml"]:
            for config_file in self.project_path.rglob(config_ext):
                if not any(ex in str(config_file) for ex in exclude_dirs):
                    self.config_files.append(config_file)

    def check_reproducibility(self):
        """Check for reproducibility issues."""
        seed_patterns = [
            r"random\.seed\(",
            r"np\.random\.seed\(",
            r"torch\.manual_seed\(",
            r"torch\.cuda\.manual_seed",
            r"tf\.random\.set_seed\(",
            r"set_seed\(",
        ]

        deterministic_patterns = [
            r"cudnn\.deterministic\s*=\s*True",
            r"cudnn\.benchmark\s*=\s*False",
            r"use_deterministic_algorithms",
        ]

        found_seed = False
        found_deterministic = False

        for py_file in self.python_files:
            try:
                content = py_file.read_text()

                for pattern in seed_patterns:
                    if re.search(pattern, content):
                        found_seed = True
                        break

                for pattern in deterministic_patterns:
                    if re.search(pattern, content):
                        found_deterministic = True
                        break

            except Exception:
                continue

        if not found_seed:
            self.issues.append(Issue(
                category="Reproducibility",
                severity="warning",
                message="No random seed setting detected",
                suggestion="Add seed setting for random, numpy, and torch/tf at the start of training"
            ))

        # Check for .gitignore
        gitignore = self.project_path / ".gitignore"
        if not gitignore.exists():
            self.issues.append(Issue(
                category="Reproducibility",
                severity="warning",
                message="No .gitignore file found",
                suggestion="Add .gitignore to exclude data/, outputs/, checkpoints/, wandb/"
            ))

    def check_data_leakage(self):
        """Check for potential data leakage patterns."""
        leakage_patterns = [
            # Fitting on all data before split
            (r"fit_transform\([^)]*\).*train_test_split", "fit_transform called before train_test_split"),
            (r"\.fit\([^)]*\).*train_test_split", "fit() called before train_test_split"),
            # Normalizing before split
            (r"normalize\([^)]*\).*split", "Normalization before data split"),
            (r"StandardScaler.*\.fit\(.*X\)", "Scaler fit on full dataset (should fit only on train)"),
        ]

        for py_file in self.python_files:
            try:
                content = py_file.read_text()

                for pattern, message in leakage_patterns:
                    if re.search(pattern, content, re.DOTALL):
                        self.issues.append(Issue(
                            category="Data Leakage",
                            severity="error",
                            message=message,
                            file=str(py_file),
                            suggestion="Fit preprocessing only on training data, transform validation/test separately"
                        ))
                        break

            except Exception:
                continue

    def check_training_setup(self):
        """Check training loop and setup."""
        has_validation = False
        has_early_stopping = False
        has_checkpointing = False
        has_logging = False
        uses_print = False

        for py_file in self.python_files:
            try:
                content = py_file.read_text()

                # Check for validation
                if re.search(r"val_loader|valid_loader|validation|eval\(\)", content):
                    has_validation = True

                # Check for early stopping
                if re.search(r"early_stop|EarlyStopping|patience", content, re.IGNORECASE):
                    has_early_stopping = True

                # Check for checkpointing
                if re.search(r"save_checkpoint|torch\.save|\.save_pretrained|checkpoint", content, re.IGNORECASE):
                    has_checkpointing = True

                # Check for proper logging
                if re.search(r"wandb\.log|mlflow\.log|logger\.|logging\.", content):
                    has_logging = True

                # Check for print-based logging
                if re.search(r'print\s*\(\s*f?["\'].*loss|print\s*\(\s*f?["\'].*epoch', content, re.IGNORECASE):
                    uses_print = True

            except Exception:
                continue

        if not has_validation:
            self.issues.append(Issue(
                category="Training",
                severity="warning",
                message="No validation loop detected",
                suggestion="Add validation to monitor overfitting during training"
            ))

        if not has_early_stopping:
            self.issues.append(Issue(
                category="Training",
                severity="info",
                message="No early stopping detected",
                suggestion="Consider adding early stopping to prevent overfitting"
            ))

        if not has_checkpointing:
            self.issues.append(Issue(
                category="Training",
                severity="warning",
                message="No checkpointing detected",
                suggestion="Save model checkpoints to resume training and keep best model"
            ))

        if uses_print and not has_logging:
            self.issues.append(Issue(
                category="Logging",
                severity="warning",
                message="Using print() for logging instead of proper logging",
                suggestion="Use wandb, mlflow, or Python logging for experiment tracking"
            ))

    def check_memory_issues(self):
        """Check for potential memory issues."""
        memory_patterns = [
            # No gradient accumulation with small batch
            (r"batch_size\s*=\s*[12]\b", "Very small batch size detected (1 or 2)", "Consider gradient accumulation if memory-constrained"),
            # Missing mixed precision
            (r"\.backward\(\)", None, None),  # Just for detection
        ]

        has_amp = False
        has_grad_accumulation = False
        has_grad_checkpointing = False

        for py_file in self.python_files:
            try:
                content = py_file.read_text()

                if re.search(r"amp\.autocast|GradScaler|mixed_precision|float16|bf16", content, re.IGNORECASE):
                    has_amp = True

                if re.search(r"accumulation_steps|gradient_accumulation|accum_iter", content, re.IGNORECASE):
                    has_grad_accumulation = True

                if re.search(r"gradient_checkpointing|checkpoint_sequential", content, re.IGNORECASE):
                    has_grad_checkpointing = True

                # Check for small batch sizes
                match = re.search(r"batch_size\s*=\s*([0-9]+)", content)
                if match and int(match.group(1)) <= 2:
                    self.issues.append(Issue(
                        category="Memory",
                        severity="info",
                        message=f"Small batch size ({match.group(1)}) detected",
                        file=str(py_file),
                        suggestion="If memory-constrained, consider gradient accumulation instead"
                    ))

            except Exception:
                continue

        # Only suggest AMP if there's actual training happening
        training_detected = any(
            re.search(r"\.backward\(\)|loss\.backward|optimizer\.step", py_file.read_text())
            for py_file in self.python_files
            if py_file.exists()
        )

        if training_detected and not has_amp:
            self.issues.append(Issue(
                category="Memory",
                severity="info",
                message="Mixed precision (AMP) not detected",
                suggestion="Enable torch.cuda.amp.autocast() for faster training and lower memory"
            ))

    def check_code_quality(self):
        """Check for ML code quality issues."""
        for py_file in self.python_files:
            try:
                content = py_file.read_text()
                lines = content.split("\n")

                # Check for hardcoded paths
                for i, line in enumerate(lines):
                    if re.search(r'["\'][/\\]Users[/\\]|["\'][/\\]home[/\\]|["\']C:\\\\', line):
                        self.issues.append(Issue(
                            category="Code Quality",
                            severity="warning",
                            message="Hardcoded absolute path detected",
                            file=str(py_file),
                            line=i + 1,
                            suggestion="Use relative paths or environment variables"
                        ))

                # Check for magic numbers in training
                magic_number_patterns = [
                    (r"lr\s*=\s*[0-9.e-]+(?!\s*#)", "Uncommented learning rate"),
                    (r"epochs\s*=\s*\d+(?!\s*#)", "Uncommented epochs"),
                    (r"hidden_size\s*=\s*\d+(?!\s*#)", "Uncommented hidden size"),
                ]

                # Only flag if there's no config file
                has_config = len([
                    f for f in self.config_files
                    if f.name in ["config.yaml", "config.yml", "hparams.yaml"]
                ]) > 0

                if not has_config:
                    for pattern, desc in magic_number_patterns:
                        matches = re.finditer(pattern, content)
                        for match in matches:
                            line_num = content[:match.start()].count("\n") + 1
                            if line_num not in [i.line for i in self.issues if i.file == str(py_file)]:
                                self.issues.append(Issue(
                                    category="Code Quality",
                                    severity="info",
                                    message=f"{desc} - consider using config file",
                                    file=str(py_file),
                                    line=line_num,
                                    suggestion="Use Hydra or a config file for hyperparameters"
                                ))
                                break  # Only one per pattern

            except Exception:
                continue

    def check_documentation(self):
        """Check for ML project documentation."""
        claude_md = self.project_path / "CLAUDE.md"
        agents_md = self.project_path / "AGENTS.md"
        readme = self.project_path / "README.md"

        if not claude_md.exists():
            self.issues.append(Issue(
                category="Documentation",
                severity="info",
                message="No CLAUDE.md found",
                suggestion="Add CLAUDE.md with project context, architecture, and current experiment info"
            ))

        if not readme.exists():
            self.issues.append(Issue(
                category="Documentation",
                severity="info",
                message="No README.md found",
                suggestion="Add README.md with setup instructions and project overview"
            ))

    def check_experiment_tracking(self):
        """Check for experiment tracking setup."""
        has_tracking = False

        for py_file in self.python_files:
            try:
                content = py_file.read_text()

                if re.search(r"wandb\.init|mlflow\.start_run|tensorboard", content, re.IGNORECASE):
                    has_tracking = True
                    break

            except Exception:
                continue

        if not has_tracking:
            self.issues.append(Issue(
                category="Experiment Tracking",
                severity="warning",
                message="No experiment tracking detected",
                suggestion="Add W&B (wandb.init) or MLflow for reproducible experiment tracking"
            ))

    def run_all_checks(self):
        """Run all validation checks."""
        self.find_files()

        if not self.python_files:
            self.issues.append(Issue(
                category="Project",
                severity="warning",
                message="No Python files found in project",
                suggestion="Ensure you're running from the correct directory"
            ))
            return

        self.check_reproducibility()
        self.check_data_leakage()
        self.check_training_setup()
        self.check_memory_issues()
        self.check_code_quality()
        self.check_documentation()
        self.check_experiment_tracking()

    def print_report(self):
        """Print validation report."""
        print("=" * 60)
        print("ML EXPERIMENT VALIDATION REPORT")
        print("=" * 60)
        print(f"\nProject: {self.project_path.absolute()}")
        print(f"Python files scanned: {len(self.python_files)}")
        print(f"Config files found: {len(self.config_files)}")

        if not self.issues:
            print("\n[All Checks Passed]")
            print("No issues detected.")
            return

        # Group by severity
        errors = [i for i in self.issues if i.severity == "error"]
        warnings = [i for i in self.issues if i.severity == "warning"]
        infos = [i for i in self.issues if i.severity == "info"]

        print(f"\nIssues found: {len(errors)} errors, {len(warnings)} warnings, {len(infos)} suggestions")

        if errors:
            print("\n[ERRORS - Must Fix]")
            for issue in errors:
                self._print_issue(issue)

        if warnings:
            print("\n[WARNINGS - Should Address]")
            for issue in warnings:
                self._print_issue(issue)

        if infos:
            print("\n[SUGGESTIONS - Consider]")
            for issue in infos:
                self._print_issue(issue)

        print("\n" + "=" * 60)

    def _print_issue(self, issue: Issue):
        """Print a single issue."""
        location = ""
        if issue.file:
            location = f" ({issue.file}"
            if issue.line:
                location += f":{issue.line}"
            location += ")"

        print(f"\n  [{issue.category}]{location}")
        print(f"    {issue.message}")
        if issue.suggestion:
            print(f"    Suggestion: {issue.suggestion}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate ML experiment for common issues")
    parser.add_argument("path", nargs="?", default=".", help="Project path to validate")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    validator = ExperimentValidator(args.path)
    validator.run_all_checks()

    if args.json:
        import json
        issues_dict = [
            {
                "category": i.category,
                "severity": i.severity,
                "message": i.message,
                "file": i.file,
                "line": i.line,
                "suggestion": i.suggestion,
            }
            for i in validator.issues
        ]
        print(json.dumps(issues_dict, indent=2))
    else:
        validator.print_report()

    # Exit code based on errors
    errors = [i for i in validator.issues if i.severity == "error"]
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
