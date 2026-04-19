#!/usr/bin/env python3
"""
UI Module — Antigravity AI Standalone Web Server
==================================================
Serves frontend/index.html (pure HTML/CSS/JS) and exposes three REST
endpoints that the JavaScript frontend calls:

  GET  /api/status  → JSON system status
  POST /api/chat    → { message } → { response }
  POST /api/clear   → clears conversation history

No Gradio dependency required.
To change the UI, edit: frontend/index.html
"""

import asyncio
import json
import logging
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ─── Keep this flag True so unified_ai_agent.py's GRADIO_AVAILABLE check
# still works without any changes to that file.
GRADIO_AVAILABLE = True

# Absolute path to the HTML file
_FRONTEND = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "frontend", "index.html"
)


def _run_async(coro):
    """Run an async coroutine synchronously (creates a dedicated event loop)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────────────────────
class _Handler(BaseHTTPRequestHandler):
    """Minimal HTTP request handler for the Antigravity AI backend."""

    # Injected by GradioInterface before the server starts
    orchestrator = None

    # ── Logging ──────────────────────────────────────────────────────────────
    def log_message(self, fmt, *args):           # suppress default output
        logger.debug("%s — %s", self.address_string(), fmt % args)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type",   "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _html(self):
        try:
            with open(_FRONTEND, "rb") as fh:
                body = fh.read()
            self.send_response(200)
            self.send_header("Content-Type",   "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_error(404, "frontend/index.html not found")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    # ── CORS preflight ────────────────────────────────────────────────────────
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ── GET ───────────────────────────────────────────────────────────────────
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            self._json(self.orchestrator.get_status())
        else:
            self._html()         # serve the SPA for every other path

    # ── POST ──────────────────────────────────────────────────────────────────
    def do_POST(self):
        path = urlparse(self.path).path
        body = self._body()

        if path == "/api/chat":
            message = body.get("message", "").strip()
            if not message:
                self._json({"error": "Empty message"}, 400)
                return
            try:
                response = _run_async(self.orchestrator.process(message))
                self._json({"response": response})
            except Exception as exc:
                logger.exception("Chat processing error")
                self._json({"error": str(exc)}, 500)

        elif path == "/api/clear":
            self.orchestrator.clear_history()
            self._json({"success": True})

        else:
            self._json({"error": "Not found"}, 404)


# ─────────────────────────────────────────────────────────────────────────────
class GradioInterface:
    """
    Drop-in replacement for the old Gradio-based interface.

    Starts a pure-Python HTTP server that:
      • serves  frontend/index.html  (the Antigravity-themed UI)
      • exposes /api/chat, /api/status, /api/clear REST endpoints

    Usage (same call-site as before):
        ui = GradioInterface(orchestrator)
        ui.launch(server_name="127.0.0.1", server_port=7860)
    """

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        _Handler.orchestrator = orchestrator

    def launch(
        self,
        server_name: str = "127.0.0.1",
        server_port: int = 7860,
        inbrowser:   bool = True,
        share:       bool = False,   # ignored — no public tunnel
        **_ignored,
    ):
        """Start the HTTP server and (optionally) open the browser."""
        url = f"http://{server_name}:{server_port}"

        if not os.path.isfile(_FRONTEND):
            logger.error(
                "Frontend file not found: %s\n"
                "Make sure frontend/index.html exists.", _FRONTEND
            )
            return

        server = HTTPServer((server_name, server_port), _Handler)

        print(f"\n  🚀  Antigravity AI  ›  {url}")
        print(  "      Press Ctrl+C to stop\n")

        if inbrowser:
            # Open the browser slightly after the server is listening
            threading.Timer(0.9, lambda: webbrowser.open(url)).start()

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n\n  ⏹  Server stopped.")
        finally:
            server.shutdown()
