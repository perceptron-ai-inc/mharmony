#!/usr/bin/env python3
# Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
"""Validate release archives for provenance, legal files, and path leaks."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

EXPECTED_VERSION = "0.1.0"
PUBLIC_REPOSITORY = "https://github.com/perceptron-ai-inc/mharmony"
UPSTREAM_REPOSITORY = "https://github.com/openai/harmony"
UPSTREAM_REVISION = "da2389a4b2daced4f98bda50e9174b35d37e3b1d"
MODIFIED_NOTICE = b"Modified by Perceptron AI, Inc.; derived from OpenAI Harmony."


def _archive_members(path: Path) -> Iterator[tuple[str, bytes]]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if not member.is_dir():
                    yield member.filename, archive.read(member)
        return
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    extracted = archive.extractfile(member)
                    assert extracted is not None
                    yield member.name, extracted.read()
        return
    raise ValueError(f"unsupported release artifact: {path}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _unique_suffix(members: dict[str, bytes], suffix: str) -> bytes:
    matches = [data for name, data in members.items() if name.endswith(suffix)]
    _require(len(matches) == 1, f"expected one {suffix}, found {len(matches)}")
    return matches[0]


def _validate_manifest(data: bytes) -> dict[str, Any]:
    manifest = json.loads(data)
    _require(
        set(manifest)
        == {
            "schema_version",
            "artifact",
            "release_version",
            "license",
            "source_release",
            "upstream",
            "derivative_work",
            "files",
            "inventory_scope",
            "snapshot_sha256",
        },
        "source manifest contains unexpected fields",
    )
    _require(manifest.get("schema_version") == 1, "unsupported source manifest schema")
    _require(
        manifest.get("artifact") == "mharmony", "source manifest artifact mismatch"
    )
    _require(
        manifest.get("release_version") == EXPECTED_VERSION, "release version mismatch"
    )
    _require(
        manifest.get("license") == "Apache-2.0", "source manifest license mismatch"
    )

    source_release = manifest.get("source_release", {})
    _require(
        set(source_release) == {"repository", "tag", "source_archive"},
        "source release contains unexpected fields",
    )
    _require(
        source_release.get("repository") == PUBLIC_REPOSITORY,
        "public repository mismatch",
    )
    _require(
        source_release.get("tag") == f"v{EXPECTED_VERSION}",
        "public source tag mismatch",
    )
    _require(
        source_release.get("source_archive")
        == f"{PUBLIC_REPOSITORY}/archive/refs/tags/v{EXPECTED_VERSION}.tar.gz",
        "public source archive mismatch",
    )

    upstream = manifest.get("upstream", {})
    _require(
        set(upstream)
        == {"name", "repository", "revision", "revision_url", "version", "license"},
        "upstream provenance contains unexpected fields",
    )
    _require(
        upstream.get("repository") == UPSTREAM_REPOSITORY,
        "upstream repository mismatch",
    )
    _require(
        upstream.get("revision") == UPSTREAM_REVISION, "upstream revision mismatch"
    )
    _require(
        upstream.get("revision_url")
        == f"{UPSTREAM_REPOSITORY}/tree/{UPSTREAM_REVISION}",
        "upstream revision URL mismatch",
    )
    _require(upstream.get("license") == "Apache-2.0", "upstream license mismatch")

    derivative = manifest.get("derivative_work", {})
    _require(
        set(derivative)
        == {"modified", "modification_notices", "upstream_history_included"},
        "derivative-work provenance contains unexpected fields",
    )
    _require(derivative.get("modified") is True, "derivative-work notice is missing")
    _require(
        derivative.get("modification_notices") == "MODIFICATIONS.md",
        "modification notice path mismatch",
    )
    _require(
        derivative.get("upstream_history_included") is False,
        "source manifest must describe the exported history accurately",
    )

    files = manifest.get("files")
    _require(isinstance(files, list) and files, "source inventory is empty")
    paths: list[str] = []
    for entry in files:
        _require(
            isinstance(entry, dict)
            and set(entry) == {"mode", "path", "sha256", "size"},
            "source inventory entry is malformed",
        )
        path = PurePosixPath(entry["path"])
        _require(
            not path.is_absolute() and ".." not in path.parts,
            "source inventory path is unsafe",
        )
        _require(
            path.name != "SOURCE_MANIFEST.json",
            "generated manifest entered source inventory",
        )
        _require(entry["mode"] in {"0644", "0755"}, "source inventory mode is invalid")
        _require(
            isinstance(entry["size"], int) and entry["size"] >= 0,
            "source inventory size is invalid",
        )
        digest = entry["sha256"]
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            "source inventory digest is invalid",
        )
        paths.append(entry["path"])
    _require(
        paths == sorted(paths) and len(paths) == len(set(paths)),
        "source inventory is not canonical",
    )
    _require(
        {"Cargo.lock", "Cargo.toml", "LICENSE", "NOTICE", "python/mharmony/__init__.py"}
        <= set(paths),
        "source inventory omits required release inputs",
    )
    canonical_entries = json.dumps(
        files, sort_keys=True, separators=(",", ":")
    ).encode()
    _require(
        hashlib.sha256(canonical_entries).hexdigest()
        == manifest.get("snapshot_sha256"),
        "source inventory snapshot digest mismatch",
    )
    _require(
        manifest.get("inventory_scope")
        == "allowlisted source files; generated manifests excluded",
        "source inventory scope mismatch",
    )
    return manifest


def _validate_members(members: dict[str, bytes], source_root: Path) -> None:
    _require(bool(members), "release artifact is empty")
    for name in members:
        path = PurePosixPath(name)
        _require(not path.is_absolute(), f"absolute archive member: {name}")
        _require(".." not in path.parts, f"traversing archive member: {name}")
        _require("__pycache__" not in path.parts, f"cache file packaged: {name}")
        _require(
            not name.endswith((".pyc", ".DS_Store")),
            f"development file packaged: {name}",
        )

    resolved_root = str(source_root.resolve())
    exact_source_markers = {
        resolved_root.encode(),
        resolved_root.replace("\\", "/").encode(),
        resolved_root.replace("/", "\\").encode(),
    }
    generic_path_markers = {
        b"/" + b"private" + b"/",
        b"/" + b"Users" + b"/",
        b"/" + b"home" + b"/",
        b":\\" + b"Users" + b"\\",
    }
    for name, data in members.items():
        for marker in exact_source_markers:
            if marker and marker in data:
                raise SystemExit(f"private or local source path leaked into {name}")
        # Native extensions legitimately retain compiler paths for third-party
        # Rust dependencies. CI remaps this project's exact source root above;
        # generic home-directory checks are limited to text assets so those
        # unrelated binary paths do not make every wheel a false positive.
        if b"\0" not in data:
            for marker in generic_path_markers:
                if marker in data:
                    raise SystemExit(f"private or local source path leaked into {name}")


def _validate_legal_files(members: dict[str, bytes]) -> None:
    package_manifest = _unique_suffix(members, "mharmony/SOURCE_MANIFEST.json")
    package_modifications = _unique_suffix(members, "mharmony/MODIFICATIONS.md")
    package_licenses = _unique_suffix(members, "mharmony/THIRD_PARTY_LICENSES.md")
    _validate_manifest(package_manifest)

    notice = _unique_suffix(members, "NOTICE")
    license_text = _unique_suffix(members, "LICENSE")
    _require(b"OpenAI Harmony" in notice, "NOTICE is missing upstream attribution")
    _require(b"Apache License" in license_text, "LICENSE is not Apache-2.0")
    _require(
        UPSTREAM_REVISION.encode() in package_modifications,
        "modification base is missing",
    )
    _require(
        b"## Bundled Cargo license files" in package_licenses
        and len(package_licenses) > 10_000,
        "third-party license bundle is incomplete",
    )


def _validate_public_readme(data: bytes) -> None:
    _require(
        PUBLIC_REPOSITORY.encode() in data, "README is missing the public repository"
    )
    _require(
        UPSTREAM_REPOSITORY.encode() in data,
        "README is missing public upstream attribution",
    )
    for disallowed in (b"file" + b"://", b"local" + b"host", b"127" + b".0.0.1"):
        _require(disallowed not in data, "README contains a non-public link")


def _validate_wheel(members: dict[str, bytes]) -> None:
    names = set(members)
    _require("mharmony/py.typed" in names, "wheel is missing py.typed")
    _require("mharmony/mharmony.pyi" in names, "wheel is missing native stubs")
    _require(
        any(
            PurePosixPath(name).parent == PurePosixPath("mharmony")
            and name.endswith((".so", ".pyd"))
            for name in names
        ),
        "wheel is missing the native extension",
    )
    for required in {
        "LICENSE",
        "MODIFICATIONS.md",
        "NOTICE",
        "THIRD_PARTY_LICENSES.md",
    }:
        _require(
            any(
                ".dist-info/licenses" in PurePosixPath(name).as_posix()
                and PurePosixPath(name).name == required
                for name in names
            ),
            f"wheel metadata is missing {required}",
        )
    _validate_public_readme(_unique_suffix(members, ".dist-info/METADATA"))
    for name, data in members.items():
        if name.startswith("mharmony/") and name.endswith((".py", ".pyi")):
            _require(
                MODIFIED_NOTICE in data[:512],
                f"modified-file notice is missing from {name}",
            )


def _validate_sdist(members: dict[str, bytes]) -> None:
    archive_roots = {PurePosixPath(name).parts[0] for name in members}
    _require(len(archive_roots) == 1, "sdist must contain one archive root")
    archive_root = archive_roots.pop()

    def root_member(relative_path: str) -> bytes:
        name = f"{archive_root}/{relative_path}"
        _require(name in members, f"sdist is missing {relative_path}")
        return members[name]

    root_manifest = root_member("SOURCE_MANIFEST.json")
    root_modifications = root_member("MODIFICATIONS.md")
    root_licenses = root_member("THIRD_PARTY_LICENSES.md")
    package_manifest = root_member("python/mharmony/SOURCE_MANIFEST.json")
    package_modifications = root_member("python/mharmony/MODIFICATIONS.md")
    package_licenses = root_member("python/mharmony/THIRD_PARTY_LICENSES.md")
    _validate_public_readme(root_member("README.md"))
    _require(root_manifest == package_manifest, "source manifests differ")
    _require(root_modifications == package_modifications, "modification notices differ")
    _require(root_licenses == package_licenses, "third-party license reports differ")
    _require(
        not any(name.endswith((".so", ".pyd")) for name in members),
        "sdist contains a binary",
    )

    for name, data in members.items():
        path = PurePosixPath(name)
        is_source = name.endswith((".rs", ".py", ".pyi"))
        is_project_source = any(
            part in {"src", "python", "scripts", "tests"} for part in path.parts
        )
        if is_source and is_project_source:
            _require(
                MODIFIED_NOTICE in data[:512],
                f"modified-file notice is missing from {name}",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    members = dict(_archive_members(args.artifact))
    _validate_members(members, args.source_root)
    _validate_legal_files(members)
    if args.artifact.suffix == ".whl":
        _validate_wheel(members)
    else:
        _validate_sdist(members)
    print(f"validated {args.artifact} ({len(members)} files)")


if __name__ == "__main__":
    main()
