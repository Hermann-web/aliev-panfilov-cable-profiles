"""Recount the released profile rises and check the release checksums.

The checker uses the standard library only. It does not integrate the cable.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY_PAGE = "https://github.com/Hermann-web/aliev-panfilov-cable-profiles"
GIT_REMOTE_SSH = "git@github.com:Hermann-web/aliev-panfilov-cable-profiles.git"
GIT_REMOTE_HTTPS = "https://github.com/Hermann-web/aliev-panfilov-cable-profiles.git"
THRESHOLD = 3.841459
FIVE_COEFFICIENTS: tuple[float, ...] = (0.20, -0.12, 0.08, -0.06, 0.05)
SCALAR_IDENTIFIABLE: tuple[str, ...] = ("a", "d", "epsilon", "k", "mu1")
NOISE_COUNTS: tuple[tuple[float, int], ...] = (
    (0.02, 5),
    (0.05, 1),
    (0.10, 0),
    (0.20, 0),
)


def fail(message: str) -> None:
    """Stop the check with one reader-facing reason."""
    raise SystemExit(message)


def require_dict(value: object, label: str) -> dict[str, object]:
    """Return a string-keyed object or stop."""
    if not isinstance(value, dict):
        fail(f"{label} is not an object")
    out: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            fail(f"{label} has a non-string key")
        out[key] = item
    return out


def require_list(value: object, label: str) -> list[object]:
    """Return a list or stop."""
    if not isinstance(value, list):
        fail(f"{label} is not a list")
    return list(value)


def require_str(value: object, label: str) -> str:
    """Return a string or stop."""
    if not isinstance(value, str):
        fail(f"{label} is not a string")
    return value


def require_int(value: object, label: str) -> int:
    """Return an integer. Booleans are rejected."""
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} is not an integer")
    return value


def require_float(value: object, label: str) -> float:
    """Return a real number. Booleans are rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not a number")
    return float(value)


def require_bool(value: object, label: str) -> bool:
    """Return a boolean or stop."""
    if not isinstance(value, bool):
        fail(f"{label} is not a boolean")
    return value


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of one file."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_json(name: str) -> dict[str, object]:
    """Load one released JSON object."""
    path = ROOT / name
    if not path.is_file():
        fail(f"missing {name}")
    value: object = json.loads(path.read_text())
    return require_dict(value, name)


def side_minimum(row: dict[str, object], label: str) -> float:
    """Return the smallest of the two side rises in one coefficient row."""
    coefficients = require_list(row.get("coefficients"), f"{label} coefficients")
    if not coefficients:
        fail(f"{label} has no coefficients")
    weakest: float | None = None
    passed = 0
    for index, item in enumerate(coefficients):
        coefficient = require_dict(item, f"{label} coefficient {index}")
        left = require_float(coefficient.get("max_rise_left"), f"{label} left {index}")
        right = require_float(coefficient.get("max_rise_right"), f"{label} right {index}")
        stored = require_bool(coefficient.get("identifiable"), f"{label} flag {index}")
        clears = left > THRESHOLD and right > THRESHOLD
        if stored != clears:
            fail(f"{label} mode flag {index + 1} does not match the two-sided rule")
        if clears:
            passed += 1
        side = left if left < right else right
        weakest = side if weakest is None else min(weakest, side)
    recorded = require_int(row.get("n_identifiable"), f"{label} n_identifiable")
    if recorded != passed:
        fail(f"{label} records {recorded} identifiable coefficients and recounts {passed}")
    if weakest is None:
        fail(f"{label} has no side minimum")
    stored_weakest = row.get("weakest_rise")
    if stored_weakest is not None:
        recorded_weakest = require_float(stored_weakest, f"{label} weakest_rise")
        if recorded_weakest != weakest:
            fail(f"{label} weakest_rise does not match the stored sides")
    return weakest


def coefficients_match(value: object, label: str, expected: tuple[float, ...]) -> None:
    """Require an exact coefficient prefix or full list."""
    raw = require_list(value, label)
    found = tuple(require_float(item, f"{label} value") for item in raw)
    if found[: len(expected)] != expected:
        fail(f"{label} does not start with the five released cosine coefficients")
    if len(found) not in (len(expected), 8):
        fail(f"{label} has an unexpected length")


def check_scalar(document: dict[str, object]) -> None:
    """Check the six-parameter profile at noise fraction 0.02."""
    if require_float(document.get("noise_fraction_of_rms"), "scalar noise") != 0.02:
        fail("scalar noise fraction is not 0.02")
    if require_int(document.get("seed"), "scalar seed") != 20260805:
        fail("scalar seed is not 20260805")
    if require_float(document.get("threshold_delta_chi_square"), "scalar threshold") != THRESHOLD:
        fail("scalar threshold is not 3.841459")
    if require_int(document.get("n_identifiable"), "scalar count") != 5:
        fail("scalar pass count is not 5")
    identifiable = tuple(
        require_str(item, "scalar name")
        for item in require_list(document.get("identifiable"), "scalar identifiable")
    )
    one_sided = tuple(
        require_str(item, "one-sided name")
        for item in require_list(document.get("one_sided"), "scalar one_sided")
    )
    non_identifiable = require_list(document.get("non_identifiable"), "scalar non_identifiable")
    if identifiable != SCALAR_IDENTIFIABLE or one_sided != ("mu2",) or non_identifiable:
        fail("scalar verdicts are not five passes with mu2 one-sided")
    profiles = require_list(document.get("profiles"), "scalar profiles")
    if len(profiles) != 6:
        fail("scalar profile list does not contain six parameters")
    for index, item in enumerate(profiles):
        profile = require_dict(item, f"scalar profile {index}")
        left = require_float(profile.get("max_rise_left"), f"scalar left {index}")
        right = require_float(profile.get("max_rise_right"), f"scalar right {index}")
        left_flag = require_bool(profile.get("bounded_left"), f"scalar bounded_left {index}")
        right_flag = require_bool(profile.get("bounded_right"), f"scalar bounded_right {index}")
        if left_flag != (left > THRESHOLD) or right_flag != (right > THRESHOLD):
            fail(f"scalar profile {index} flags do not match the threshold")
        if left_flag and right_flag:
            verdict = "identifiable"
        elif left_flag or right_flag:
            verdict = "one_sided"
        else:
            verdict = "non_identifiable"
        if require_str(profile.get("verdict"), f"scalar verdict {index}") != verdict:
            fail(f"scalar profile {index} verdict does not match the two-sided rule")


def row_by_modes(document: dict[str, object], n_modes: int, label: str) -> dict[str, object]:
    """Return the unique row with one mode count."""
    rows = require_list(document.get("rows"), f"{label} rows")
    found: list[dict[str, object]] = []
    for index, item in enumerate(rows):
        row = require_dict(item, f"{label} row {index}")
        if require_int(row.get("n_modes"), f"{label} n_modes {index}") == n_modes:
            found.append(row)
    if len(found) != 1:
        fail(f"{label} does not contain exactly one {n_modes}-mode row")
    return found[0]


def check_field_scaling(document: dict[str, object]) -> None:
    """Check modes one to five at 2 percent noise."""
    if require_float(document.get("noise_fraction_of_rms"), "field-scaling noise") != 0.02:
        fail("field-scaling noise is not 0.02")
    if require_float(document.get("threshold_delta_chi_square"), "field-scaling threshold") != THRESHOLD:
        fail("field-scaling threshold is not 3.841459")
    coefficients_match(document.get("true_coefficients"), "field-scaling coefficients", FIVE_COEFFICIENTS)
    for n_modes in (1, 2, 3, 4, 5):
        row = row_by_modes(document, n_modes, "field-scaling")
        side_minimum(row, f"field-scaling {n_modes}")
        if require_int(row.get("n_identifiable"), f"field-scaling {n_modes} count") != n_modes:
            fail(f"field-scaling mode {n_modes} is not a complete pass")
    one = row_by_modes(document, 1, "field-scaling")
    five = row_by_modes(document, 5, "field-scaling")
    if side_minimum(one, "field-scaling 1") != 7469.34695664094:
        fail("one-mode weakest side is not 7469.34695664094")
    if side_minimum(five, "field-scaling 5") != 7.199851994644462:
        fail("five-mode weakest side at 2 percent is not 7.199851994644462")


def noise_block(document: dict[str, object], fraction: float) -> dict[str, object]:
    """Return the unique five-mode noise block."""
    blocks = require_list(document.get("noise_axis"), "noise axis")
    found: list[dict[str, object]] = []
    for index, item in enumerate(blocks):
        block = require_dict(item, f"noise block {index}")
        if require_float(block.get("noise_fraction_of_rms"), f"noise fraction {index}") == fraction:
            found.append(block)
    if len(found) != 1:
        fail(f"noise fraction {fraction} is not stored once")
    return found[0]


def check_crossover(document: dict[str, object]) -> None:
    """Check the six-mode, seven-mode, and five-mode noise rows."""
    mode_axis = require_dict(document.get("mode_axis"), "mode axis")
    if require_float(mode_axis.get("noise_fraction_of_rms"), "mode-axis noise") != 0.02:
        fail("mode axis is not at noise 0.02")
    coefficients_match(mode_axis.get("true_coefficients"), "mode-axis coefficients", FIVE_COEFFICIENTS)
    six = row_by_modes(mode_axis, 6, "mode axis")
    seven = row_by_modes(mode_axis, 7, "mode axis")
    if require_int(six.get("n_identifiable"), "six-mode count") != 4:
        fail("six-mode count is not 4")
    if side_minimum(six, "six-mode") != -2.6796664115226747:
        fail("six-mode weakest side does not match the release")
    if require_int(seven.get("n_identifiable"), "seven-mode count") != 7:
        fail("seven-mode count is not 7")
    if side_minimum(seven, "seven-mode") != 5.383316658233582:
        fail("seven-mode weakest side does not match the release")
    for fraction, expected in NOISE_COUNTS:
        if fraction == 0.02:
            continue
        block = noise_block(document, fraction)
        if require_float(block.get("threshold_delta_chi_square"), f"threshold {fraction}") != THRESHOLD:
            fail(f"noise {fraction} does not use threshold 3.841459")
        coefficients_match(block.get("true_coefficients"), f"noise {fraction} coefficients", FIVE_COEFFICIENTS)
        row = row_by_modes(block, 5, f"noise {fraction}")
        side_minimum(row, f"noise {fraction}")
        if require_int(row.get("n_identifiable"), f"noise {fraction} count") != expected:
            fail(f"noise {fraction} count is not {expected}")


def check_origin() -> None:
    """When this checkout is a git repository, require the recorded origin."""
    if not (ROOT / ".git").exists():
        return
    url = subprocess.check_output(
        ["git", "-C", str(ROOT), "remote", "get-url", "origin"],
        text=True,
    ).strip()
    if url not in {GIT_REMOTE_SSH, GIT_REMOTE_HTTPS}:
        fail("origin does not match the recorded public remote")


def check_generator_settings() -> None:
    """Require the released generator to still name the run settings."""
    source = (ROOT / "practical_identifiability.py").read_text()
    required = (
        "SEED = 20260805",
        "CHI2_95_1DOF = 3.841459",
        "NOISE_FRACTION = 0.02",
        "for nz in (0.05, 0.10, 0.20)",
        "rng = np.random.default_rng(SEED + n_modes)",
        "sigma = noise * rms",
    )
    for snippet in required:
        if snippet not in source:
            fail(f"generator is missing the setting {snippet}")


def check_manifest() -> None:
    """Check every manifest digest and the checksum list."""
    manifest_path = ROOT / "PUBLIC-RELEASE-MANIFEST.json"
    sums_path = ROOT / "SHA256SUMS"
    manifest = require_dict(json.loads(manifest_path.read_text()), "manifest")
    if require_str(manifest.get("publication_status"), "publication_status") != "local_unpublished":
        fail("manifest publication status is not local_unpublished")
    if require_str(manifest.get("repository_url"), "repository_url") != REPOSITORY_PAGE:
        fail("manifest repository URL does not match the recorded remote")
    if require_str(manifest.get("git_remote_url"), "git_remote_url") != GIT_REMOTE_SSH:
        fail("manifest git remote does not match the recorded remote")
    if manifest.get("version") is not None:
        fail("manifest records a version")
    files = require_list(manifest.get("files"), "manifest files")
    seen: set[str] = set()
    for index, item in enumerate(files):
        entry = require_dict(item, f"manifest file {index}")
        name = require_str(entry.get("path"), f"manifest path {index}")
        if name in seen or name in {"PUBLIC-RELEASE-MANIFEST.json", "SHA256SUMS"}:
            fail(f"manifest path {name} is duplicated or circular")
        seen.add(name)
        rights = require_str(entry.get("rights_class"), f"manifest rights {name}")
        licence = require_str(entry.get("licence"), f"manifest licence {name}")
        require_str(entry.get("role"), f"manifest role {name}")
        if rights == "original_software" and licence != "MIT":
            fail(f"{name} is software without the MIT licence")
        if rights == "project_authored_output" and licence != "CC-BY-4.0":
            fail(f"{name} is a profile output without CC BY 4.0")
        if rights not in {"original_software", "project_authored_output"}:
            fail(f"{name} has an unknown rights class")
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            fail(f"{name} is missing or is a symlink")
        digest = require_str(entry.get("sha256"), f"manifest sha256 {name}")
        if sha256_file(path) != digest:
            fail(f"sha256 mismatch for {name}")
        size = require_int(entry.get("size_bytes"), f"manifest size {name}")
        if path.stat().st_size != size:
            fail(f"byte size mismatch for {name}")
    listed: dict[str, str] = {}
    for line in sums_path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name in listed:
            fail(f"SHA256SUMS repeats {name}")
        listed[name] = digest
    expected_names = seen | {"PUBLIC-RELEASE-MANIFEST.json"}
    if set(listed) != expected_names:
        fail("SHA256SUMS does not match the manifest file set")
    for name, digest in listed.items():
        if sha256_file(ROOT / name) != digest:
            fail(f"SHA256SUMS mismatch for {name}")


def check_rights_text() -> None:
    """Require the reader files to state the same two-licence split."""
    rights = (ROOT / "RIGHTS.md").read_text()
    readme = (ROOT / "README.md").read_text()
    notice = (ROOT / "LICENSE").read_text()
    software = (ROOT / "LICENSE-MIT.txt").read_text()
    for label, text in (("RIGHTS.md", rights), ("README.md", readme)):
        if "CC BY 4.0" not in text or "MIT" not in text:
            fail(f"{label} does not name both licences")
    if "Fondation Ecole Centrale Casablanca" not in notice or "CC BY 4.0" not in notice:
        fail("LICENSE does not state the dataset notice")
    if not software.startswith("MIT License\n") or "Fondation Ecole Centrale Casablanca" not in software:
        fail("LICENSE-MIT.txt is not the software notice")
    if "unresolved" in rights.lower():
        fail("RIGHTS.md still leaves a licence unresolved")


def check_no_private_markers() -> None:
    """Reject local paths and internal working notes in the projection."""
    forbidden = ("/home/", "reviews-round1", "engineering-plan", "identifiability-note")
    for path in sorted(ROOT.iterdir()):
        if path.name == "verify_profiles.py" or not path.is_file() or path.is_symlink():
            continue
        text = path.read_text(errors="replace")
        for marker in forbidden:
            if marker in text:
                fail(f"{path.name} contains {marker}")


def main() -> int:
    """Run the count recount and the checksum check."""
    check_scalar(load_json("practical-identifiability.json"))
    check_field_scaling(load_json("identifiability-field-scaling.json"))
    check_crossover(load_json("identifiability-crossover.json"))
    check_generator_settings()
    check_manifest()
    check_origin()
    check_rights_text()
    check_no_private_markers()
    print("five-mode cosine field: 5/5, 1/5, 0/5, 0/5 at 2%, 5%, 10%, 20% RMS")
    print("scalar preset at 2% RMS: 5 of 6 pass; mu2 is one-sided")
    print("checksums and two-sided rises match the released JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
