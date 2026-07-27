"""Serve the data collection app on the local network so a phone can open it.

    python collect_app/serve.py [--port 8810]

Prints every address the phone can use. Bookmark it, or use the browser's
"Add to Home Screen" so it opens like an app.
"""
from __future__ import annotations
import argparse, socket
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


class H(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def lan_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ips.add(s.getsockname()[0]); s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    return sorted(ips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8810)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), partial(H, directory=str(HERE)))
    print(f"MNKAI-III data collection app serving on port {a.port}")
    print(f"  on this computer : http://localhost:{a.port}/")
    for ip in lan_ips():
        print(f"  on your phone    : http://{ip}:{a.port}/     <- bookmark this")
    print("\n  (phone must be on the same Wi-Fi. Ctrl-C to stop.)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
