"""
tools/system/package_manager.py
================================
Implements system.install_app and system.update_app intents.

Install strategy (in order):
  1. Check KNOWN_APPS map — maps friendly names like "whatsapp" to
     the correct package name + preferred source (apt / snap / flatpak).
  2. Try Flatpak into the custom THIRAN_INSTALL_DIR on New Volume (preferred).
  3. If Flatpak is unavailable or fails, fall back to Snap.
  4. If Snap also fails, fall back to the system package manager (apt/dnf/…).
  5. Report success or failure via ToolResult.

Install location:
  All user-installed apps are stored in THIRAN_INSTALL_DIR so that
  they reside on /media/sriram/New Volume rather than the system root.

Security:
  - Package names are sanitised against a strict allowlist regex.
  - sudo password is fed via stdin (sudo -S); never logged.
  - Flatpak and Snap do NOT need sudo — they run as the current user.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path
from typing import Any

from core.models import ToolResult
from tools.base import BaseTool, SlotField

# ---------------------------------------------------------------------------
# Custom install location — all apps go here instead of system root
# ---------------------------------------------------------------------------

#: Base directory on New Volume where Thiran installs all apps.
THIRAN_INSTALL_DIR: str = os.environ.get(
    "THIRAN_INSTALL_DIR",
    "/media/sriram/New Volume/Apps",
)

#: Flatpak uses a named installation called "thiran" inside THIRAN_INSTALL_DIR.
_FLATPAK_INSTALL_NAME = "thiran"
_FLATPAK_INSTALL_PATH = str(Path(THIRAN_INSTALL_DIR) / "flatpak")


def _ensure_flatpak_installation() -> None:
    """
    Create the directory and register a custom Flatpak installation
    named 'thiran' pointing at THIRAN_INSTALL_DIR/flatpak.

    This is idempotent — running it multiple times is safe.
    Requires flatpak >= 1.0 and that the user has write access to the path.
    """
    install_path = Path(_FLATPAK_INSTALL_PATH)
    install_path.mkdir(parents=True, exist_ok=True)

    conf_dir = Path.home() / ".local" / "share" / "flatpak" / "installations.d"
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_file = conf_dir / "thiran.conf"

    if not conf_file.exists():
        conf_file.write_text(
            f"[Installation \"{_FLATPAK_INSTALL_NAME}\"]\n"
            f"Path={_FLATPAK_INSTALL_PATH}\n"
            f"DisplayName=Thiran (New Volume)\n"
            f"StorageType=harddisk\n"
        )

# ---------------------------------------------------------------------------
# Sudo password helper
# ---------------------------------------------------------------------------

_FALLBACK_SUDO_PASS = "340125"


def _get_sudo_pass() -> str:
    """Read sudo password from env, falling back to the configured default."""
    return os.environ.get("SUDO_PASS", _FALLBACK_SUDO_PASS)


# ---------------------------------------------------------------------------
# Known-apps map
# ---------------------------------------------------------------------------
# Maps a friendly/spoken name → (apt_pkg, snap_pkg, flatpak_id)
# Use None to skip a source.  Order of preference: apt → snap → flatpak.
#
# Entries here cover popular apps that are NOT in standard apt repos or have
# a different package name than what the user typically says.

_KNOWN_APPS: dict[str, tuple[str | None, str | None, str | None]] = {
    # Messaging
    "whatsapp":        (None,             "whatsapp-for-linux", "com.ktechpit.whatsie"),
    "telegram":        ("telegram-desktop","telegram-desktop",   "org.telegram.desktop"),
    "signal":          (None,             "signal-desktop",     "org.signal.Signal"),
    "discord":         (None,             "discord",            "com.discordapp.Discord"),
    "slack":           (None,             "slack",              "com.slack.Slack"),
    "zoom":            (None,             "zoom-client",        "us.zoom.Zoom"),
    "teams":           (None,             "teams",              "com.microsoft.Teams"),
    "skype":           (None,             "skype",              "com.skype.Skype"),

    # Browsers
    "chrome":          (None,             "chromium",           "org.chromium.Chromium"),
    "chromium":        ("chromium-browser","chromium",          "org.chromium.Chromium"),
    "firefox":         ("firefox",        "firefox",            "org.mozilla.firefox"),
    "brave":           (None,             "brave",              "com.brave.Browser"),
    "opera":           (None,             "opera",              "com.opera.Opera"),

    # Media
    "vlc":             ("vlc",            "vlc",                "org.videolan.VLC"),
    "spotify":         (None,             "spotify",            "com.spotify.Client"),
    "mpv":             ("mpv",            "mpv",                "io.mpv.Mpv"),

    # Dev tools
    "vscode":          (None,             "code",               "com.visualstudio.code"),
    "vs code":         (None,             "code",               "com.visualstudio.code"),
    "visual studio code": (None,          "code",               "com.visualstudio.code"),
    "sublime":         (None,             "sublime-text",       None),
    "pycharm":         (None,             "pycharm-community",  "com.jetbrains.PyCharm-Community"),
    "gimp":            ("gimp",           "gimp",               "org.gimp.GIMP"),
    "obsidian":        (None,             "obsidian",           "md.obsidian.Obsidian"),
    "postman":         (None,             "postman",            None),

    # Utilities
    "libreoffice":     ("libreoffice",    "libreoffice",        "org.libreoffice.LibreOffice"),
    "thunderbird":     ("thunderbird",    "thunderbird",        "org.mozilla.Thunderbird"),
    "bitwarden":       (None,             "bitwarden",          "com.bitwarden.desktop"),
    "steam":           ("steam",          "steam",              "com.valvesoftware.Steam"),
    "handbrake":       (None,             "handbrake",          "fr.handbrake.ghb"),
    "audacity":        ("audacity",       "audacity",           "org.audacityteam.Audacity"),
}


def _resolve_known_app(
    name: str,
) -> tuple[str | None, str | None, str | None] | None:
    """
    Look up a friendly app name in the known-apps map.
    Returns (apt_pkg, snap_pkg, flatpak_id) or None if not found.
    """
    key = name.strip().lower()
    return _KNOWN_APPS.get(key)


# ---------------------------------------------------------------------------
# System (apt/dnf/pacman) package manager registry
# ---------------------------------------------------------------------------

# (binary, install_cmd, update_pkg_cmd, update_all_cmd)
_SYSTEM_MANAGERS = [
    (
        "apt-get",
        "sudo -S apt-get install -y {pkg}",
        "sudo -S apt-get install --only-upgrade -y {pkg}",
        "sudo -S sh -c 'apt-get update && apt-get upgrade -y'",
    ),
    (
        "apt",
        "sudo -S apt install -y {pkg}",
        "sudo -S apt install --only-upgrade -y {pkg}",
        "sudo -S sh -c 'apt update && apt upgrade -y'",
    ),
    (
        "dnf",
        "sudo -S dnf install -y {pkg}",
        "sudo -S dnf upgrade -y {pkg}",
        "sudo -S dnf upgrade -y",
    ),
    (
        "yum",
        "sudo -S yum install -y {pkg}",
        "sudo -S yum update -y {pkg}",
        "sudo -S yum update -y",
    ),
    (
        "pacman",
        "sudo -S pacman -S --noconfirm {pkg}",
        "sudo -S pacman -S --noconfirm {pkg}",
        "sudo -S pacman -Syu --noconfirm",
    ),
    (
        "zypper",
        "sudo -S zypper install -y {pkg}",
        "sudo -S zypper update -y {pkg}",
        "sudo -S zypper update -y",
    ),
]

_INSTALL_TIMEOUT = 180
_UPDATE_TIMEOUT  = 240

_SAFE_PKG_RE = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9._+\-]*[a-zA-Z0-9])?$")
# Flatpak IDs allow dots and more segments
_SAFE_FLATPAK_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._\-]*$")


def _detect_system_pm() -> tuple[str, str, str, str] | None:
    for entry in _SYSTEM_MANAGERS:
        if shutil.which(entry[0]):
            return entry  # type: ignore[return-value]
    return None


def _sanitise(raw: str, pattern: re.Pattern = _SAFE_PKG_RE) -> str | None:
    cleaned = raw.strip().lower()
    if not cleaned or not pattern.match(cleaned):
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Shell runner
# ---------------------------------------------------------------------------


async def _run(cmd: str, timeout: int, needs_sudo: bool = True) -> tuple[int, str]:
    """
    Run *cmd* in a shell.
    If needs_sudo is True, feed the password via stdin for sudo -S.
    """
    stdin_bytes = (((_get_sudo_pass()) + "\n").encode()) if needs_sudo else b""

    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout_b, _ = await asyncio.wait_for(
            proc.communicate(input=stdin_bytes), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return -1, "Timed out."

    output = stdout_b.decode(errors="replace").strip() if stdout_b else ""
    rc = proc.returncode if proc.returncode is not None else -1
    return rc, output


# ---------------------------------------------------------------------------
# Source-specific install helpers
# ---------------------------------------------------------------------------


async def _install_apt(pkg: str, pm_entry: tuple) -> tuple[int, str, str]:
    """Returns (rc, output, label)."""
    binary, install_tpl, *_ = pm_entry
    cmd = install_tpl.format(pkg=pkg)
    rc, out = await _run(cmd, _INSTALL_TIMEOUT, needs_sudo=True)
    return rc, out, binary


async def _install_snap(pkg: str) -> tuple[int, str, str]:
    """Returns (rc, output, label)."""
    if not shutil.which("snap"):
        return -1, "snap not available", "snap"
    cmd = f"sudo -S snap install {pkg}"
    rc, out = await _run(cmd, _INSTALL_TIMEOUT, needs_sudo=True)
    return rc, out, "snap"


async def _install_flatpak(flatpak_id: str) -> tuple[int, str, str]:
    """
    Install *flatpak_id* from Flathub into the custom Thiran installation
    on New Volume (THIRAN_INSTALL_DIR/flatpak).

    Falls back to the default user installation if the custom path
    registration fails for any reason.
    """
    if not shutil.which("flatpak"):
        return -1, "flatpak not available", "flatpak"

    # Ensure the custom installation directory and config are in place.
    try:
        _ensure_flatpak_installation()
        install_flag = f"--installation={_FLATPAK_INSTALL_NAME}"
        label = f"flatpak ({_FLATPAK_INSTALL_PATH})"
    except Exception:
        install_flag = "--user"
        label = "flatpak (user)"

    cmd = (
        f"flatpak install -y --noninteractive "
        f"{install_flag} flathub {flatpak_id}"
    )
    rc, out = await _run(cmd, _INSTALL_TIMEOUT, needs_sudo=False)
    return rc, out, label


# ---------------------------------------------------------------------------
# Install tool
# ---------------------------------------------------------------------------


class SystemInstallAppTool(BaseTool):

    @property
    def name(self) -> str:
        return "system.install_app"

    @property
    def description(self) -> str:
        return (
            "Install a new application or package on the system. "
            "Handles popular apps like WhatsApp, Spotify, Discord, VLC, etc. "
            "Tries apt first, then Snap, then Flatpak automatically."
        )

    @property
    def slot_fields(self) -> list[SlotField]:
        return [
            SlotField(
                name="package_name",
                type_hint="string",
                required=True,
                description=(
                    "The name of the application to install, "
                    "e.g. 'vlc', 'whatsapp', 'spotify', 'vscode', 'firefox'."
                ),
            )
        ]

    async def _execute_impl(self, slots: dict[str, Any], request_id: str) -> ToolResult:
        log = self._log(request_id)
        raw: str = str(slots.get("package_name", "")).strip()

        # ── Resolve friendly name ──────────────────────────────────────
        known = _resolve_known_app(raw)

        # ── Attempt installation ───────────────────────────────────────
        # Priority: Flatpak (custom New Volume path) → Snap → apt/system
        # Flatpak is preferred because it installs into THIRAN_INSTALL_DIR
        # so all apps are stored on /media/sriram/New Volume, not system root.
        pm = _detect_system_pm()
        attempts: list[str] = []  # record tried sources for error message

        log.info(
            "system.install_app.start",
            app=raw,
            install_dir=THIRAN_INSTALL_DIR,
        )

        # 1. Flatpak (preferred — installs to New Volume)
        flatpak_id = known[2] if known else None
        if not flatpak_id:
            flatpak_id = _sanitise(raw, _SAFE_FLATPAK_RE)
        if flatpak_id:
            rc, out, label = await _install_flatpak(flatpak_id)
            attempts.append(label)
            if rc == 0:
                log.info("system.install_app.success", pkg=flatpak_id, via=label)
                return ToolResult(
                    request_id=request_id, tool_name=self.name, success=True,
                    message=(
                        f"Successfully installed {raw.title()} via Flatpak. "
                        f"It's stored in {THIRAN_INSTALL_DIR}."
                    ),
                    data={
                        "package": flatpak_id,
                        "source": label,
                        "install_dir": THIRAN_INSTALL_DIR,
                    },
                    duration_ms=0,
                )
            log.warning("system.install_app.flatpak_failed", pkg=flatpak_id, rc=rc)

        # 2. Snap fallback
        snap_pkg = known[1] if known else _sanitise(raw)
        if snap_pkg:
            rc, out, label = await _install_snap(snap_pkg)
            attempts.append(label)
            if rc == 0:
                log.info("system.install_app.success", pkg=snap_pkg, via=label)
                return ToolResult(
                    request_id=request_id, tool_name=self.name, success=True,
                    message=f"Successfully installed {raw.title()} via Snap.",
                    data={"package": snap_pkg, "source": label},
                    duration_ms=0,
                )
            log.warning("system.install_app.snap_failed", pkg=snap_pkg, rc=rc)

        # 3. apt / system package manager (last resort — installs system-wide)
        apt_pkg = known[0] if known else _sanitise(raw)
        if apt_pkg and pm:
            rc, out, label = await _install_apt(apt_pkg, pm)
            attempts.append(label)
            if rc == 0:
                log.info("system.install_app.success", pkg=apt_pkg, via=label)
                return ToolResult(
                    request_id=request_id, tool_name=self.name, success=True,
                    message=f"Successfully installed {raw.title()} via {label}.",
                    data={"package": apt_pkg, "source": label},
                    duration_ms=0,
                )
            log.warning("system.install_app.apt_failed", pkg=apt_pkg, rc=rc)

        tried = ", ".join(dict.fromkeys(attempts)) or "any package manager"
        log.error("system.install_app.all_failed", app=raw, tried=tried)
        return ToolResult(
            request_id=request_id, tool_name=self.name, success=False,
            message=(
                f"I couldn't install {raw.title()}. "
                f"I tried {tried} but none of them worked. "
                "It may not be available or may need manual installation."
            ),
            error_code="INSTALL_FAILED",
            data={"app": raw, "tried": tried},
            duration_ms=0,
        )


# ---------------------------------------------------------------------------
# Update tool
# ---------------------------------------------------------------------------


class SystemUpdateAppTool(BaseTool):

    @property
    def name(self) -> str:
        return "system.update_app"

    @property
    def description(self) -> str:
        return (
            "Update an installed application or all packages. "
            "Uses apt, Snap, and Flatpak. Leave package_name empty to update everything."
        )

    @property
    def slot_fields(self) -> list[SlotField]:
        return [
            SlotField(
                name="package_name",
                type_hint="string",
                required=False,
                default=None,
                description=(
                    "App to update, e.g. 'firefox', 'whatsapp'. "
                    "Omit or say 'all' to update all installed packages."
                ),
            )
        ]

    async def _execute_impl(self, slots: dict[str, Any], request_id: str) -> ToolResult:
        log = self._log(request_id)
        raw: str | None = slots.get("package_name")  # type: ignore[assignment]
        update_all = not raw or str(raw).strip().lower() in ("all", "everything", "system", "")

        pm = _detect_system_pm()

        if update_all:
            results: list[str] = []
            errors: list[str] = []

            if pm:
                _, _, _, update_all_cmd = pm
                rc, out = await _run(update_all_cmd, _UPDATE_TIMEOUT, needs_sudo=True)
                if rc == 0:
                    results.append("system packages")
                else:
                    errors.append(f"{pm[0]} (rc={rc})")

            if shutil.which("snap"):
                rc, out = await _run("sudo -S snap refresh", _UPDATE_TIMEOUT, needs_sudo=True)
                if rc == 0:
                    results.append("Snap apps")
                else:
                    errors.append(f"snap (rc={rc})")

            if shutil.which("flatpak"):
                rc, out = await _run(
                    "flatpak update -y --noninteractive", _UPDATE_TIMEOUT, needs_sudo=False
                )
                if rc == 0:
                    results.append("Flatpak apps")
                else:
                    errors.append(f"flatpak (rc={rc})")

            if results:
                updated = ", ".join(results)
                log.info("system.update_app.all_done", updated=updated)
                return ToolResult(
                    request_id=request_id, tool_name=self.name, success=True,
                    message=f"Update complete! Updated {updated}.",
                    data={"updated": results, "errors": errors},
                    duration_ms=0,
                )
            else:
                return ToolResult(
                    request_id=request_id, tool_name=self.name, success=False,
                    message="I couldn't update any packages. Please check your internet connection.",
                    error_code="UPDATE_FAILED",
                    data={"errors": errors},
                    duration_ms=0,
                )

        # ── Single-package update ──────────────────────────────────────
        raw_str = str(raw).strip()
        known = _resolve_known_app(raw_str)

        # Try apt first
        apt_pkg = known[0] if known else _sanitise(raw_str)
        if apt_pkg and pm:
            _, _, update_pkg_tpl, _ = pm
            rc, out = await _run(update_pkg_tpl.format(pkg=apt_pkg), _UPDATE_TIMEOUT, needs_sudo=True)
            if rc == 0:
                return ToolResult(
                    request_id=request_id, tool_name=self.name, success=True,
                    message=f"Updated {raw_str.title()} successfully.",
                    data={"package": apt_pkg, "source": pm[0]},
                    duration_ms=0,
                )

        # Try snap
        snap_pkg = known[1] if known else _sanitise(raw_str)
        if snap_pkg and shutil.which("snap"):
            rc, out = await _run(f"sudo -S snap refresh {snap_pkg}", _UPDATE_TIMEOUT, needs_sudo=True)
            if rc == 0:
                return ToolResult(
                    request_id=request_id, tool_name=self.name, success=True,
                    message=f"Updated {raw_str.title()} via Snap.",
                    data={"package": snap_pkg, "source": "snap"},
                    duration_ms=0,
                )

        # Try flatpak
        flatpak_id = known[2] if known else None
        if flatpak_id and shutil.which("flatpak"):
            rc, out = await _run(
                f"flatpak update -y --noninteractive {flatpak_id}",
                _UPDATE_TIMEOUT, needs_sudo=False,
            )
            if rc == 0:
                return ToolResult(
                    request_id=request_id, tool_name=self.name, success=True,
                    message=f"Updated {raw_str.title()} via Flatpak.",
                    data={"package": flatpak_id, "source": "flatpak"},
                    duration_ms=0,
                )

        return ToolResult(
            request_id=request_id, tool_name=self.name, success=False,
            message=(
                f"I couldn't update {raw_str.title()}. "
                "It may not be installed or the name might be different."
            ),
            error_code="UPDATE_FAILED",
            data={"app": raw_str},
            duration_ms=0,
        )
