#!/usr/bin/env python3
# Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
"""Generate the bundled third-party dependency license report.

The Rust extension statically links Cargo dependencies, so their license files
are copied into one deterministic report. Python requirements are installed as
separate distributions and are listed from pyproject.toml; those distributions
remain responsible for shipping their own license files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 build environments
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "THIRD_PARTY_LICENSES.md"
PACKAGE_REPORT = ROOT / "python" / "mharmony" / REPORT.name


def _cargo_packages() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "cargo",
            "metadata",
            "--locked",
            "--offline",
            "--all-features",
            "--format-version",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(completed.stdout)
    return sorted(
        (
            package
            for package in metadata["packages"]
            if package["name"] != "mharmony" and package.get("source") is not None
        ),
        key=lambda package: (package["name"].casefold(), package["version"]),
    )


def _license_files(package: dict[str, Any]) -> list[Path]:
    package_root = Path(package["manifest_path"]).parent
    candidates = [
        path
        for path in package_root.iterdir()
        if path.is_file()
        and path.name.upper().startswith(("LICENSE", "COPYING", "UNLICENSE"))
    ]
    explicit = package.get("license_file")
    if explicit:
        candidates.append(package_root / explicit)
    return sorted(set(candidates), key=lambda path: path.name.casefold())


def _render() -> str:
    packages = _cargo_packages()
    missing_metadata = [
        f"{package['name']} {package['version']}"
        for package in packages
        if not package.get("license") and not package.get("license_file")
    ]
    if missing_metadata:
        raise RuntimeError(
            "Cargo packages without license metadata: " + ", ".join(missing_metadata)
        )

    license_documents: dict[str, dict[str, Any]] = {}
    packages_without_files: list[str] = []
    for package in packages:
        files = _license_files(package)
        if not files:
            packages_without_files.append(f"{package['name']} {package['version']}")
        for path in files:
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            document = license_documents.setdefault(
                digest,
                {
                    "content": data.decode("utf-8", errors="replace")
                    .replace("\r\n", "\n")
                    .replace("\r", "\n")
                    .rstrip(),
                    "file_names": set(),
                    "packages": set(),
                },
            )
            document["file_names"].add(path.name)
            document["packages"].add(f"{package['name']} {package['version']}")

    by_license: dict[str, list[str]] = defaultdict(list)
    for package in packages:
        expression = package.get("license") or f"LicenseRef-{package['license_file']}"
        by_license[expression].append(f"{package['name']} {package['version']}")

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    python_groups = {
        "base": project["project"].get("dependencies", []),
        **project["project"].get("optional-dependencies", {}),
    }

    lines = [
        "# Third-party licenses",
        "",
        "This report is generated from the locked Cargo dependency graph and the",
        "Python dependency declarations for `mharmony` 0.1.0. It is informational",
        "and does not replace the license terms in the referenced projects.",
        "",
        "Regenerate and verify it with:",
        "",
        "```bash",
        "python scripts/generate_third_party_licenses.py --write",
        "python scripts/generate_third_party_licenses.py --check",
        "```",
        "",
        "## Rust dependency inventory",
        "",
        "Cargo dependencies are grouped by the license expression published in",
        "their package metadata. The full license files found in the locked Cargo",
        "package sources follow this inventory.",
        "",
    ]
    for expression in sorted(by_license, key=str.casefold):
        lines.extend(
            [
                f"- `{expression}`: "
                + ", ".join(f"`{item}`" for item in by_license[expression]),
                "",
            ]
        )

    if packages_without_files:
        lines.extend(
            [
                "The following packages declared license metadata but did not expose a",
                "top-level license file in the local Cargo package source:",
                "",
                ", ".join(f"`{item}`" for item in packages_without_files),
                "",
            ]
        )

    lines.extend(
        [
            "## Python dependency declarations",
            "",
            "Python dependencies are installed as separate distributions rather than",
            "copied into the `mharmony` wheel. Their own distribution metadata and",
            "license files are authoritative.",
            "",
        ]
    )
    for group in sorted(python_groups):
        requirements = python_groups[group]
        rendered = ", ".join(f"`{requirement}`" for requirement in requirements)
        lines.extend([f"- `{group}`: {rendered or '_none_'}", ""])

    lines.extend(["## Bundled Cargo license files", ""])
    for digest, document in sorted(license_documents.items()):
        package_list = ", ".join(f"`{item}`" for item in sorted(document["packages"]))
        file_names = ", ".join(f"`{name}`" for name in sorted(document["file_names"]))
        lines.extend(
            [
                f"### SHA-256 `{digest}`",
                "",
                f"Packages: {package_list}",
                "",
                f"Source file names: {file_names}",
                "",
                "```text",
                document["content"],
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    expected = _render()
    if args.write:
        REPORT.write_text(expected)
        PACKAGE_REPORT.write_text(expected)
        return

    stale = [
        path
        for path in (REPORT, PACKAGE_REPORT)
        if not path.is_file() or path.read_text() != expected
    ]
    if stale:
        print(
            "stale third-party license report: " + ", ".join(map(str, stale)),
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
