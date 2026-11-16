"""Build the offline Windows Server bundle: Python runtime, dependencies, code, weights and dashboard.

Run with the project's Python 3.12 environment, which must already hold the versions pinned in
deploy/windows/requirements-windows.txt (the packages are copied from it, not downloaded):

    python scripts/build_windows_bundle.py --python-home PATH --output dist/windows-server-YYYYMMDD

--python-home is a standalone CPython 3.12 for Windows x64 (for example from
`uv python install 3.12`); its site-packages are not copied apart from pip.
The output folder must not exist: an earlier bundle is never overwritten or deleted.
Needs about 4 GB free. Only the override wheels (see OVERRIDES) are downloaded.
"""
import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "deploy" / "windows"
APP = "LAAS Football Analytics"
# Wheels installed in place of whatever the build environment holds. The server's virtual CPU has no
# SSE4.2/AVX, so it needs the older NumPy and the Polars runtime built without those instructions.
OVERRIDES = {"numpy": "numpy==2.3.5", "polars-runtime-compat": "polars-runtime-compat==2.0.0"}
NEEDED_FREE = 4 * 1024 ** 3


def requirements():
    lines = [line.strip() for line in (WINDOWS / "requirements-windows.txt").read_text(encoding="utf-8").splitlines()]
    return [Requirement(line) for line in lines if line and not line.startswith("#")]


def copy_dependencies(site_packages: Path):
    """Copy the pinned packages and everything they need from this environment, file by file."""
    pending, seen, versions, missing = [], set(), [], []
    for requirement in requirements():
        name = canonicalize_name(requirement.name)
        if name not in OVERRIDES and requirement.specifier and metadata.version(name) not in requirement.specifier:
            raise SystemExit(f"{name} {metadata.version(name)} is installed but the bundle pins {requirement.specifier}. "
                             "Install deploy/windows/requirements-windows.txt into this environment first.")
        pending.append(name)
    while pending:
        name = canonicalize_name(pending.pop())
        if name in seen:
            continue
        seen.add(name)
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            if name not in OVERRIDES:
                missing.append(name)  # declared by a package but absent here; fetched as a wheel instead
            continue
        pending.extend(requirement.name for spec in distribution.requires or []
                       if (requirement := Requirement(spec)).marker is None or requirement.marker.evaluate())
        if name in OVERRIDES:
            continue  # its files come from the override wheel
        versions.append(f"{name}=={distribution.version}")
        for record in distribution.files or []:
            path = Path(record)
            if ".." in path.parts or path.suffix == ".pyc":
                continue
            source = distribution.locate_file(record)
            if source.is_file():
                target = site_packages / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    return versions, missing


def install_wheels(site_packages: Path, cache: Path, extra):
    """Download and unpack the override wheels and any declared dependency the build environment lacks."""
    cache.mkdir(parents=True, exist_ok=True)
    versions = []
    for spec in [*OVERRIDES.values(), *extra]:
        subprocess.run([sys.executable, "-m", "pip", "download", spec, "--no-deps", "--only-binary=:all:",
                        "--platform", "win_amd64", "--python-version", "3.12", "--dest", str(cache)], check=True)
    for wheel in sorted(cache.glob("*.whl")):
        versions.append("==".join(wheel.name.split("-")[:2]).lower().replace("_", "-") + " (wheel)")
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.namelist():
                if ".." in Path(member).parts:
                    raise SystemExit(f"Unsafe path in {wheel.name}")
            archive.extractall(site_packages)
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-web-build", action="store_true", help="Use the existing apps/web/dist as it is")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or sys.platform != "win32":
        raise SystemExit("Build with Python 3.12 on Windows x64: compiled packages are copied from this environment.")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"{output} already exists. Choose a new folder; earlier bundles are never overwritten.")
    if not (args.python_home / "python.exe").is_file():
        raise SystemExit("--python-home must contain python.exe")
    free = shutil.disk_usage(output.parent if output.parent.exists() else ROOT).free
    if free < NEEDED_FREE:
        raise SystemExit(f"Only {free / 1024 ** 3:.1f} GB free; the build needs about {NEEDED_FREE / 1024 ** 3:.0f} GB.")
    sys.path.insert(0, str(ROOT))
    from workers.video.yolo_adapter import model_path, pitch_model_path
    weights, pitch = model_path(ROOT), pitch_model_path(ROOT)
    if not weights.is_file():
        raise SystemExit("Detector weights not found. Set YOLO_MODEL_PATH.")
    if pitch is None or not pitch.is_file():
        raise SystemExit("Pitch model not found at models/pitch-keypoints/football-pitch-detection.pt")
    if not args.skip_web_build:
        subprocess.run("npm run build", cwd=ROOT / "apps" / "web", check=True, shell=True)
    if not (ROOT / "apps" / "web" / "dist" / "index.html").is_file():
        raise SystemExit("apps/web/dist is missing; build the dashboard first.")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "apps", "workers", "deploy/windows"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip())

    payload = output / APP
    payload.mkdir(parents=True)
    print("Copying Python runtime", flush=True)
    shutil.copytree(args.python_home, payload / "python",
                    ignore=lambda folder, names: [n for n in names if n == "__pycache__" or
                                                  (Path(folder).name == "site-packages" and not n.lower().startswith("pip"))])
    site_packages = payload / "python" / "Lib" / "site-packages"
    site_packages.mkdir(parents=True, exist_ok=True)
    print("Copying dependencies", flush=True)
    versions, missing = copy_dependencies(site_packages)
    if missing:
        print("Not installed here, downloading wheels:", ", ".join(missing), flush=True)
    versions += install_wheels(site_packages, output / "wheels", missing)
    print("Copying application", flush=True)
    skip = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "workers", payload / "workers", ignore=skip)
    shutil.copytree(ROOT / "apps" / "api", payload / "apps" / "api", ignore=skip)
    shutil.copytree(ROOT / "apps" / "web" / "dist", payload / "apps" / "web" / "dist")
    (payload / "models" / "pitch-keypoints").mkdir(parents=True)
    shutil.copyfile(weights, payload / "models" / "football.pt")
    shutil.copyfile(pitch, payload / "models" / "pitch-keypoints" / pitch.name)
    for name in ("Start-LAAS.py", "smoke.py", "smoke-clip.mp4", "requirements-windows.txt", "READ-ME.txt", "LAAS-Startup.xml"):
        shutil.copyfile(WINDOWS / name, payload / name)
    (payload / "BUILD.json").write_text(json.dumps(dict(commit=commit, uncommitted_changes=dirty, packages=sorted(versions)), indent=2), encoding="utf-8")
    for name in ("READ-ME.txt", "LAAS-Startup.xml"):
        shutil.copyfile(WINDOWS / name, output / name)
    if not args.no_zip:
        print("Writing zip", flush=True)
        with zipfile.ZipFile(output / "LAAS-Windows-Server.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(payload.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(APP) / path.relative_to(payload))
    print(json.dumps(dict(output=str(output), commit=commit, uncommitted_changes=dirty, packages=len(versions))))


if __name__ == "__main__":
    main()
