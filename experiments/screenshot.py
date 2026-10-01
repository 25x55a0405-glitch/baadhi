"""Screenshot the dashboard with headless Chrome once the map has really finished drawing.

usage: python experiments/screenshot.py "<url>" out.png [--size 1440x900] [--wait "<js condition>"] [--js "<js to run first>"]

Drives Chrome over its DevTools protocol: loads the page, runs optional JS, waits until the map reports
loaded (and the optional condition holds), then captures the viewport. Used for checks and report figures.
"""
import argparse
import base64
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websockets.sync.client as ws

CHROME = r"C:/Program Files/Google/Chrome/Application/chrome.exe"
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--size", default="1440x900")
    ap.add_argument("--wait", default="true", help="extra JS condition to wait for")
    ap.add_argument("--js", default="", help="JS to run once the page has loaded (e.g. click a button)")
    ap.add_argument("--timeout", type=float, default=90)
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.split("x"))
    import socket
    with socket.socket() as sock:                  # a free port, so back-to-back runs never collide
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    prof = ROOT / ".work" / f"chrome-cdp-{port}"
    shutil.rmtree(prof, ignore_errors=True)
    proc = subprocess.Popen([CHROME, "--headless=new", "--enable-unsafe-swiftshader", "--use-angle=swiftshader", f"--window-size={w},{h}",
                             "--hide-scrollbars", f"--remote-debugging-port={port}", f"--user-data-dir={prof}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2).read())
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:  # noqa: BLE001 — Chrome still starting
                time.sleep(0.5)
        else:
            sys.exit("Chrome did not start")
        with ws.connect(page["webSocketDebuggerUrl"], max_size=None) as c:
            n = [0]

            def call(method, **params):
                n[0] += 1
                c.send(json.dumps({"id": n[0], "method": method, "params": params}))
                while True:
                    msg = json.loads(c.recv())
                    if msg.get("id") == n[0]:
                        if "error" in msg:
                            raise RuntimeError(msg["error"])
                        return msg.get("result", {})

            def js(expr):
                r = call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
                return r.get("result", {}).get("value")

            call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=False)
            call("Page.enable")
            call("Page.navigate", url=a.url)
            t0 = time.time()
            ready = f"(() => {{ const m = window.baadhiMap; return !!(document.readyState === 'complete' && m && m.loaded() && ({a.wait})); }})()"
            ran = False
            while time.time() - t0 < a.timeout:
                if not ran and a.js and js("document.readyState") == "complete" and js("!!window.baadhiMap"):
                    js(a.js)
                    ran = True
                if js(ready) and (ran or not a.js):
                    time.sleep(1.5)          # let the last frame settle
                    if js(ready):
                        break
                time.sleep(0.5)
            else:
                print("warning: timed out waiting; capturing anyway", file=sys.stderr)
            err = js("window.baadhiLastError || null")
            if err:
                print("page error:", err, file=sys.stderr)
            shot = call("Page.captureScreenshot", format="png")
            Path(a.out).write_bytes(base64.b64decode(shot["data"]))
            print(f"saved {a.out} after {time.time() - t0:.1f}s")
    finally:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        shutil.rmtree(prof, ignore_errors=True)


if __name__ == "__main__":
    main()
