"""Install the official Linux x86_64 Trivy release; lock version and SHA once."""
import hashlib
import io
import json
import platform
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "ci" / "trivy-lock.json"


def fetch(url):
    if not url.startswith(("https://api.github.com/", "https://github.com/")):
        raise ValueError("Only official GitHub HTTPS URLs are accepted")
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
    raise SystemExit("This installer targets Linux x86_64 (VMware / ubuntu-24.04 runner)")

if "--lock" in sys.argv:
    if LOCK.exists():
        raise SystemExit("Trivy lock already exists; refusing to replace it automatically")
    release = json.loads(fetch("https://api.github.com/repos/aquasecurity/trivy/releases/latest"))
    version = release["tag_name"].removeprefix("v")
    name = f"trivy_{version}_Linux-64bit.tar.gz"
    base = f"https://github.com/aquasecurity/trivy/releases/download/v{version}/"
    checksums = fetch(base + f"trivy_{version}_checksums.txt").decode()
    digest = next(line.split()[0] for line in checksums.splitlines() if line.split()[-1] == name)
    LOCK.write_text(json.dumps({"version": version, "url": base + name, "sha256": digest}, indent=2) + "\n")

lock = json.loads(LOCK.read_text())
payload = fetch(lock["url"])
if hashlib.sha256(payload).hexdigest() != lock["sha256"]:
    raise SystemExit("Trivy SHA256 mismatch")
with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
    member = archive.getmember("trivy")
    if not member.isfile():
        raise SystemExit("Invalid binary member")
    binary = archive.extractfile(member).read()
path = ROOT / ".tools" / "trivy"
path.parent.mkdir(exist_ok=True)
path.write_bytes(binary)
path.chmod(0o755)
print("Installed official Trivy", lock["version"], "with verified checksum")
