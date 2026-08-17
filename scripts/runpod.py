"""Minimal RunPod lifecycle helper (REST v1). API key from ~/.runpod/api_key or $RUNPOD_API_KEY.

    python scripts/runpod.py list
    python scripts/runpod.py create NAME [--gpu "NVIDIA GeForce RTX 4090"] [--disk 60] [--spot]
    python scripts/runpod.py wait POD_ID          # until publicIp + ssh port
    python scripts/runpod.py ssh POD_ID           # prints: ssh -p PORT root@IP
    python scripts/runpod.py stop POD_ID | terminate POD_ID
    python scripts/runpod.py cost                 # running pods, $/h
Every create/terminate is appended to ~/.runpod/ledger.log.
"""
from __future__ import annotations
import argparse, json, os, sys, time, urllib.request, urllib.error

KEY = os.environ.get("RUNPOD_API_KEY") or open(os.path.expanduser("~/.runpod/api_key")).read().strip()
BASE = "https://rest.runpod.io/v1"
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "User-Agent": "stickdance/0.1"}
LEDGER = os.path.expanduser("~/.runpod/ledger.log")
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def req(method, path, body=None):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None, headers=H, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            t = resp.read()
            return json.loads(t) if t else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} -> HTTP {e.code}: {e.read().decode()[:400]}")


def ledger(msg):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    open(LEDGER, "a").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def create(name, gpu, disk, spot, image):
    body = dict(name=name, imageName=image, cloudType="SECURE", computeType="GPU", gpuTypeIds=[gpu], gpuCount=1,
                containerDiskInGb=disk, volumeInGb=20, ports=["22/tcp"], interruptible=spot,
                env={"JUPYTER_DISABLE": "1"}, allowedCudaVersions=["12.4", "12.5", "12.6", "12.8"])
    p = req("POST", "/pods", body)
    ledger(f"CREATE {p['id']} {name} {gpu} spot={spot} costPerHr={p.get('costPerHr')}")
    return p


def wait(pid, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = req("GET", f"/pods/{pid}")
        ip, pm = p.get("publicIp"), p.get("portMappings") or {}
        if ip and "22" in pm:
            return ip, pm["22"], p
        time.sleep(10)
    raise SystemExit("timeout waiting for pod network")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("arg", nargs="?")
    ap.add_argument("--gpu", default="NVIDIA GeForce RTX 4090"); ap.add_argument("--disk", type=int, default=60)
    ap.add_argument("--spot", action="store_true"); ap.add_argument("--image", default=IMAGE)
    a = ap.parse_args()
    if a.cmd == "list":
        for p in req("GET", "/pods"):
            print(p["id"], p["name"], p.get("desiredStatus"), p.get("publicIp"), (p.get("portMappings") or {}).get("22"), f"${p.get('costPerHr', 0):.2f}/h", (p.get("gpu") or {}).get("displayName", ""))
    elif a.cmd == "create":
        p = create(a.arg, a.gpu, a.disk, a.spot, a.image); print(p["id"], f"${p.get('costPerHr')}/h")
    elif a.cmd == "wait":
        ip, port, _ = wait(a.arg); print(f"ssh -p {port} root@{ip}")
    elif a.cmd == "ssh":
        p = req("GET", f"/pods/{a.arg}"); print(f"ssh -p {(p.get('portMappings') or {}).get('22')} root@{p.get('publicIp')}")
    elif a.cmd == "stop":
        req("POST", f"/pods/{a.arg}/stop"); ledger(f"STOP {a.arg}"); print("stopped")
    elif a.cmd == "terminate":
        req("DELETE", f"/pods/{a.arg}"); ledger(f"TERMINATE {a.arg}"); print("terminated")
    elif a.cmd == "cost":
        tot = 0
        for p in req("GET", "/pods"):
            if p.get("desiredStatus") == "RUNNING": tot += p.get("costPerHr", 0); print(p["id"], p["name"], f"${p.get('costPerHr'):.2f}/h")
        print(f"total ${tot:.2f}/h")
    else:
        sys.exit("unknown cmd")


if __name__ == "__main__":
    main()
