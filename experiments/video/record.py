"""Record the dashboard with headless Chrome (DevTools screencast) while a script drives it.

    rec = Recorder(1920, 1080); rec.open("http://127.0.0.1:8000/#run=…")
    rec.js("document.querySelector('[data-view=s2_after]').click()"); rec.hold(3)
    rec.save_clip("scene.mp4", speed=1.0); rec.close()

Frames arrive only when the page repaints; each frame is held until the next one, so the clip keeps
real time (or a chosen speed-up for long waits such as a live analysis).
"""
from __future__ import annotations

import base64
import json
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import websockets.sync.client as ws

CHROME = r"C:/Program Files/Google/Chrome/Application/chrome.exe"
ROOT = Path(__file__).resolve().parents[2]


class Recorder:
    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 30):
        self.w, self.h, self.fps = width, height, fps
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.prof = ROOT / ".work" / f"chrome-rec-{self.port}"
        shutil.rmtree(self.prof, ignore_errors=True)
        self.proc = subprocess.Popen([CHROME, "--headless=new", "--enable-unsafe-swiftshader", "--use-angle=swiftshader",
                                      f"--window-size={width},{height}", "--hide-scrollbars", "--force-device-scale-factor=1",
                                      f"--remote-debugging-port={self.port}", f"--user-data-dir={self.prof}", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(80):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json", timeout=2).read())
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:  # noqa: BLE001 — Chrome still starting
                time.sleep(0.25)
        self.c = ws.connect(page["webSocketDebuggerUrl"], max_size=None)
        self.n, self.lock = 0, threading.Lock()
        self.pending: dict[int, dict] = {}
        self.frames: list[tuple[float, bytes]] = []
        self.recording = False
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)
        self.call("Page.enable")

    # ---------------------------------------------------------------- DevTools plumbing
    def _read(self):
        while True:
            try:
                msg = json.loads(self.c.recv())
            except Exception:  # noqa: BLE001 — socket closed
                return
            if msg.get("method") == "Page.screencastFrame":
                p = msg["params"]
                if self.recording:
                    self.frames.append((time.time(), base64.b64decode(p["data"])))
                self._send("Page.screencastFrameAck", sessionId=p["sessionId"])
            elif "id" in msg:
                with self.lock:
                    self.pending[msg["id"]] = msg

    def _send(self, method, **params):
        with self.lock:
            self.n += 1
            i = self.n
        self.c.send(json.dumps({"id": i, "method": method, "params": params}))
        return i

    def call(self, method, **params):
        i = self._send(method, **params)
        for _ in range(6000):
            with self.lock:
                if i in self.pending:
                    msg = self.pending.pop(i)
                    if "error" in msg:
                        raise RuntimeError(msg["error"])
                    return msg.get("result", {})
            time.sleep(0.01)
        raise TimeoutError(method)

    def js(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    # ---------------------------------------------------------------- recording
    def open(self, url: str, ready: str = "window.baadhiMap && window.baadhiMap.loaded()", timeout: float = 90):
        self.call("Page.navigate", url=url)
        self.wait(ready, timeout)

    def wait(self, cond: str, timeout: float = 600, poll: float = 0.5) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.js(f"!!({cond})"):
                    return True
            except Exception:  # noqa: BLE001 — page navigating
                pass
            time.sleep(poll)
        return False

    def start(self):
        self.frames = []
        self.recording = True
        self.call("Page.startScreencast", format="jpeg", quality=88, maxWidth=self.w, maxHeight=self.h, everyNthFrame=1)
        self.t_start = time.time()

    def hold(self, seconds: float):
        time.sleep(seconds)

    def stop(self):
        self.t_end = time.time()
        self.recording = False
        try:
            self.call("Page.stopScreencast")
        except Exception:  # noqa: BLE001
            pass

    def save_clip(self, out: Path, speed: float = 1.0, min_seconds: float = 0.0):
        """Frames → constant-rate MP4; each frame is held until the next (real time ÷ speed)."""
        out = Path(out)
        tmp = out.parent / f"{out.stem}_frames"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        frames = self.frames or []
        if not frames:
            raise RuntimeError("no frames recorded")
        lines = []
        end = max(self.t_end, frames[-1][0] + 0.1)
        for k, (t, data) in enumerate(frames):
            (tmp / f"{k:06d}.jpg").write_bytes(data)
            nxt = frames[k + 1][0] if k + 1 < len(frames) else end
            lines.append(f"file '{k:06d}.jpg'\nduration {max(nxt - t, 1 / self.fps) / speed:.4f}")
        total = (end - frames[0][0]) / speed
        if total < min_seconds:
            lines[-1] = lines[-1].rsplit("duration", 1)[0] + f"duration {float(lines[-1].rsplit('duration', 1)[1]) + min_seconds - total:.4f}"
        lines.append(f"file '{len(frames) - 1:06d}.jpg'")            # concat demuxer: repeat the last frame
        (tmp / "list.txt").write_text("\n".join(lines), encoding="utf8")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
                        "-vf", f"fps={self.fps},scale={self.w}:{self.h}:flags=lanczos,format=yuv420p", "-c:v", "libx264",
                        "-preset", "medium", "-crf", "18", str(out)], check=True)
        shutil.rmtree(tmp, ignore_errors=True)
        return out

    def screenshot(self, out: Path):
        Path(out).write_bytes(base64.b64decode(self.call("Page.captureScreenshot", format="png")["data"]))

    def close(self):
        try:
            self.c.close()
        except Exception:  # noqa: BLE001
            pass
        self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        shutil.rmtree(self.prof, ignore_errors=True)
