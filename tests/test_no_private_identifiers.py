"""
Guard against publishing infrastructure identifiers (a GCP project ID and a VM's public IP once shipped in the docs).
Scans every tracked text file for public IPv4 addresses and GCP project IDs.
"""
import ipaddress
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
PROJECT = re.compile(r"--project[= ]\S+|\b[a-z]+-[a-z]+-\d{6}(?:-[a-z0-9]{1,4})?\b")
ALLOWED_NETS = [ipaddress.ip_network("35.235.240.0/20")]  # Google IAP's published range, used in the firewall rule


def _tracked_text_files():
    try:
        names = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    for name in names:
        path = ROOT / name
        if path.suffix in {".png", ".jpg", ".ico", ".gguf", ".so", ".bin", ".pdf"} or not path.is_file():
            continue
        try:
            yield name, path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


def _is_public(text):
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return False  # version strings like 1.2.300.4
    return ip.is_global and not any(ip in net for net in ALLOWED_NETS)


def test_no_public_ips_or_gcp_project_ids_in_tracked_files():
    leaks = []
    for name, text in _tracked_text_files():
        for n, line in enumerate(text.splitlines(), 1):
            hits = [m for m in IPV4.findall(line) if _is_public(m)] + PROJECT.findall(line)
            leaks += [f"{name}:{n}: {h}" for h in hits]
    assert not leaks, "infrastructure identifiers in tracked files:\n" + "\n".join(leaks)
