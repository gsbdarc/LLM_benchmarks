"""Check whether a repository workflow's local prerequisites are configured.

The checker is deliberately local and read-only: it verifies installed package
metadata, command availability, and whether required configuration names are set.
It never connects to MongoDB or a model API and never prints configuration values.
"""

import argparse
from dataclasses import dataclass
from importlib import metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Mapping, Sequence

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def load_requirements(path: Path) -> list[Requirement]:
    requirements: list[Requirement] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            requirements.append(Requirement(line))
    return requirements


def _exact_version(requirement: Requirement) -> str | None:
    specifiers = list(requirement.specifier)
    if len(specifiers) != 1:
        return None
    specifier = specifiers[0]
    if specifier.operator not in {"==", "==="} or specifier.version.endswith(".*"):
        return None
    return specifier.version


def check_lock(
    source: Sequence[Requirement], lock: Sequence[Requirement]
) -> list[Check]:
    """Verify that the generated lock is exact and covers the direct requirements."""
    lock_by_name: dict[str, Requirement] = {}
    duplicate_names: set[str] = set()
    unpinned_names: list[str] = []

    for requirement in lock:
        if requirement.marker and not requirement.marker.evaluate():
            continue
        name = canonicalize_name(requirement.name)
        if name in lock_by_name:
            duplicate_names.add(name)
        lock_by_name[name] = requirement
        if _exact_version(requirement) is None:
            unpinned_names.append(name)

    lock_problems = sorted(set(unpinned_names) | duplicate_names)
    pin_check = Check(
        "dependency lock pins",
        not lock_problems,
        (
            f"{len(lock_by_name)} packages are pinned exactly"
            if not lock_problems
            else "not exactly pinned or duplicated: " + ", ".join(lock_problems[:8])
        ),
    )

    source_problems: list[str] = []
    checked_source = 0
    for requirement in source:
        if requirement.marker and not requirement.marker.evaluate():
            continue
        checked_source += 1
        name = canonicalize_name(requirement.name)
        locked = lock_by_name.get(name)
        if locked is None:
            source_problems.append(f"{name} missing")
            continue
        version = _exact_version(locked)
        if version is None:
            source_problems.append(f"{name} not exact")
        elif requirement.specifier and version not in requirement.specifier:
            source_problems.append(
                f"{name} {version} violates {requirement.specifier}"
            )

    coverage_check = Check(
        "direct dependency coverage",
        not source_problems,
        (
            f"all {checked_source} direct dependencies are locked compatibly"
            if not source_problems
            else "; ".join(source_problems[:8])
        ),
    )
    return [pin_check, coverage_check]


def check_requirements(
    requirements: Sequence[Requirement],
    *,
    installed_version: Callable[[str], str] = metadata.version,
) -> list[Check]:
    checks: list[Check] = []
    for requirement in requirements:
        if requirement.marker and not requirement.marker.evaluate():
            continue

        name = requirement.name.lower()
        try:
            version = installed_version(requirement.name)
        except metadata.PackageNotFoundError:
            checks.append(
                Check(
                    f"package {name}",
                    False,
                    "not installed; run `python -m pip install -r requirements.txt`",
                )
            )
            continue

        if requirement.specifier and version not in requirement.specifier:
            checks.append(
                Check(
                    f"package {name}",
                    False,
                    f"installed {version}, requires {requirement.specifier}",
                )
            )
            continue

        constraint = (
            f" and satisfies {requirement.specifier}" if requirement.specifier else ""
        )
        checks.append(Check(f"package {name}", True, f"installed {version}{constraint}"))

    return checks


def check_locked_environment(
    lock: Sequence[Requirement],
    *,
    installed_version: Callable[[str], str] = metadata.version,
) -> Check:
    """Aggregate installed-version mismatches so normal output stays reviewable."""
    mismatches: list[str] = []
    checked = 0
    for requirement in lock:
        if requirement.marker and not requirement.marker.evaluate():
            continue
        checked += 1
        expected = _exact_version(requirement)
        name = canonicalize_name(requirement.name)
        if expected is None:
            mismatches.append(f"{name} is not exactly pinned")
            continue
        try:
            actual = installed_version(requirement.name)
        except metadata.PackageNotFoundError:
            mismatches.append(f"{name} is missing")
            continue
        if actual != expected:
            mismatches.append(f"{name} is {actual}, expected {expected}")

    return Check(
        "locked environment",
        not mismatches,
        (
            f"all {checked} locked packages match"
            if not mismatches
            else "; ".join(mismatches[:8])
        ),
    )


def _dotenv_values(path: Path) -> dict[str, str]:
    """Read enough dotenv syntax to check presence without importing or revealing values."""
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip():
            values[key.strip()] = value.strip().strip("'\"")
    return values


def _is_configured(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return normalized not in {
        "changeme",
        "replace-me",
        "tbd",
        "your-key-here",
        "your_mongo_password",
        "your_mongo_user",
        "your_stanford_key",
        "your_wandb_key",
    } and not normalized.startswith(("your_", "/path/to/"))


def _environment_check(name: str, values: Mapping[str, str]) -> Check:
    if _is_configured(values.get(name)):
        return Check(f"environment {name}", True, "configured")
    return Check(
        f"environment {name}",
        False,
        f"missing or placeholder; set {name} in the environment or repository .env",
    )


def check_environment(
    profile: str,
    *,
    backend: str = "playground",
    batch: bool = False,
    root: Path,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> list[Check]:
    if profile not in {"python", "inference", "agent-eval"}:
        raise ValueError(f"unknown profile: {profile}")
    if backend not in {"playground", "nim", "qwen"}:
        raise ValueError(f"unknown backend: {backend}")

    command_path = which or shutil.which
    values = _dotenv_values(root / ".env")
    values.update(os.environ if environ is None else environ)
    checks: list[Check] = []

    if profile == "inference":
        poppler = command_path("pdftoppm")
        checks.append(
            Check(
                "command pdftoppm",
                poppler is not None,
                "available" if poppler else "missing; install Poppler before converting PDFs",
            )
        )
        for name in (
            "BASE_DIR",
            "STANFORD_API_KEY",
            "MONGO_DB_USERNAME",
            "MONGO_DB_PASSWORD",
        ):
            checks.append(_environment_check(name, values))

    if profile == "agent-eval":
        for name in ("MONGO_DB_USERNAME", "MONGO_DB_PASSWORD"):
            checks.append(_environment_check(name, values))
        if backend == "playground":
            checks.append(_environment_check("STANFORD_API_KEY", values))
        elif backend == "qwen":
            checks.append(_environment_check("LOCAL_MODEL_URL", values))

    if batch:
        sbatch = command_path("sbatch")
        checks.append(
            Check(
                "command sbatch",
                sbatch is not None,
                "available" if sbatch else "missing; batch mode requires a SLURM client",
            )
        )

    return checks


def check_python_version(version: tuple[int, int] | None = None) -> Check:
    current = version or sys.version_info[:2]
    ok = current == (3, 10)
    return Check(
        "python version",
        ok,
        f"{current[0]}.{current[1]} detected; supported version is Python 3.10.x",
    )


def check_dependency_graph() -> Check:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return Check("pip dependency graph", True, "no broken requirements found")
    detail = (result.stdout or result.stderr).strip().splitlines()
    return Check(
        "pip dependency graph",
        False,
        detail[0] if detail else "`python -m pip check` failed",
    )


def render_checks(checks: Sequence[Check]) -> str:
    lines = [
        f"{'PASS' if check.ok else 'FAIL'}  {check.name}: {check.detail}"
        for check in checks
    ]
    failed = sum(not check.ok for check in checks)
    lines.append(f"Summary: {len(checks) - failed} passed, {failed} failed")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check local prerequisites without contacting external services."
    )
    parser.add_argument("profile", choices=("python", "inference", "agent-eval"))
    parser.add_argument(
        "--backend",
        choices=("playground", "nim", "qwen"),
        default="playground",
        help="agent-eval backend whose local configuration should be checked",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="also require the sbatch command used by SLURM launchers",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    source = load_requirements(root / "requirements.in")
    lock = load_requirements(root / "requirements.txt")
    checks = [check_python_version()]
    checks.extend(check_lock(source, lock))
    checks.append(check_locked_environment(lock))
    checks.append(check_dependency_graph())
    checks.extend(
        check_environment(
            args.profile,
            backend=args.backend,
            batch=args.batch,
            root=root,
        )
    )
    print(render_checks(checks))
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
