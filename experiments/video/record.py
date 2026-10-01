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
    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 30, scale: float = 1.0):
        """width x height = size of the video frames; the page itself is laid out at (width/scale) x (height/scale)
        CSS pixels, so scale=1.5 gives a 1280x720 layout with 1.5x bigger text and controls in a 1080p video."""
        self.w, self.h, self.fps, self.scale = width, height, fps, scale
        cw, ch = int(round(width / scale)), int(round(height / scale))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.prof = ROOT / ".work" / f"chrome-rec-{self.port}"
        shutil.rmtree(self.prof, ignore_errors=True)
        self.proc = subprocess.Popen([CHROME, "--headless=new", "--enable-unsafe-swiftshader", "--use-angle=swiftshader",
                                      f"--window-size={cw},{ch}", "--hide-scrollbars", f"--force-device-scale-factor={scale}",
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
        self.call("Emulation.setDeviceMetricsOverride", width=cw, height=ch, deviceScaleFactor=scale, mobile=False)
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

    def click_xy(self, x: float, y: float):
        for typ in ("mouseMoved", "mousePressed", "mouseReleased"):
            self.call("Input.dispatchMouseEvent", type=typ, x=x, y=y, button="left", clickCount=1)

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

    def save_clip(self, out: Path, speed: float = 1.0, min_seconds: float = 0.0, segments=None):
        """Frames -> constant-rate MP4: each frame is held until the next one (real time / speed).
        `segments` = [(from_s, to_s, speed)], seconds since start(): parts played faster (e.g. a long wait).
        The JPEG frames are streamed into ffmpeg through a pipe — no temporary image files (deleting
        .jpg/.mp4 files from Python misbehaves on this machine)."""
        out = Path(out)
        frames = self.frames or []
        if not frames:
            raise RuntimeError("no frames recorded")
        t = self.t_start if segments else frames[0][0]
        end = max(self.t_end, frames[-1][0] + 0.1)

        def speed_at(t_abs):
            r = t_abs - self.t_start
            for a, b, sp in segments or []:
                if a <= r < b:
                    return sp
            return speed

        ticks = []
        while t < end:
            ticks.append(t)
            t += speed_at(t) / self.fps
        while len(ticks) < int(round(min_seconds * self.fps)):
            ticks.append(end)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(self.fps), "-c:v", "mjpeg", "-i", "-",
               "-vf", f"scale={self.w}:{self.h}:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset", "medium",
               "-crf", "18", "-r", str(self.fps), str(out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        idx = 0
        for tick in ticks:
            while idx + 1 < len(frames) and frames[idx + 1][0] <= tick:
                idx += 1
            proc.stdin.write(frames[idx][1])
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode the clip")
        return out

    def screenshot(self, out: Path):
        Path(out).write_bytes(base64.b64decode(self.call("Page.captureScreenshot", format="png")["data"]))

    def close(self):
        try:
            self.c.close()
        except Exception:  # noqa: BLE001
            pass
        # kill the whole tree (renderer, GPU and crashpad helpers outlive the main process)
        subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       stdin=subprocess.DEVNULL)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        # the throw-away Chrome profile (.work/chrome-rec-*) is left for `rm -rf` from the shell
