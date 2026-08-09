"""Web server for Trail REPL — FastAPI-like REST API with HTTP server."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from trail.repl.session import ReplSession, Result


# ── Session store ─────────────────────────────────────────────────────

@dataclass
class SessionRecord:
    id: str
    created_at: str
    last_active: str
    history: list[dict[str, Any]] = field(default_factory=list)
    session: ReplSession = field(default_factory=lambda: ReplSession(), repr=False)

    def touch(self) -> None:
        self.last_active = datetime.utcnow().isoformat()


_sessions: dict[str, SessionRecord] = {}


def create_session() -> str:
    sid = uuid.uuid4().hex[:8]
    _sessions[sid] = SessionRecord(
        id=sid,
        created_at=datetime.utcnow().isoformat(),
        last_active=datetime.utcnow().isoformat(),
    )
    return sid


def get_session(sid: str) -> SessionRecord:
    rec = _sessions.get(sid)
    if rec is None:
        raise ValueError(f"Session {sid} not found")
    return rec


def delete_session(sid: str) -> None:
    _sessions.pop(sid, None)


def list_sessions() -> list[SessionRecord]:
    return list(_sessions.values())


# ── REST API ───────────────────────────────────────────────────────────

class APIServer:
    """API layer for REPL session management."""

    def __init__(self) -> None:
        self.sessions: dict[str, SessionRecord] = _sessions

    def create_session(self) -> dict[str, str]:
        sid = create_session()
        return {"session_id": sid}

    def execute(self, sid: str, input_text: str) -> dict[str, Any]:
        rec = get_session(sid)
        rec.touch()
        result = rec.session.process_input(input_text)

        entry: dict[str, Any] = {"input": input_text}

        if result.is_error:
            entry["output"] = None
            entry["error"] = result.message
            entry["type"] = "error"
        elif result.is_result:
            df = result.value
            entry["error"] = None
            entry["type"] = "result"
            if df is not None and len(df) > 0:
                records = df.to_dicts()
                entry["output"] = {
                    "rows": len(records),
                    "columns": df.columns,
                    "preview": records[:20],
                }
            else:
                entry["output"] = None
        else:
            entry["error"] = None
            entry["output"] = None
            entry["type"] = "noop"

        rec.history.append(entry)
        return entry

    def history(self, sid: str) -> list[dict[str, Any]]:
        get_session(sid)
        return list(get_session(sid).history)

    def info(self, sid: str) -> dict[str, Any]:
        rec = get_session(sid)
        return {
            "session_id": rec.id,
            "created_at": rec.created_at,
            "last_active": rec.last_active,
            "history_count": len(rec.history),
            "definitions": list(rec.session.definitions.keys()),
        }

    def delete(self, sid: str) -> dict[str, str]:
        delete_session(sid)
        return {"status": "deleted"}

    def session_list(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": rec.id,
                "created_at": rec.created_at,
                "last_active": rec.last_active,
                "history_count": len(rec.history),
                "definitions": list(rec.session.definitions.keys()),
            }
            for rec in self.sessions.values()
        ]


# ── HTTP Handlers ─────────────────────────────────────────────────────

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse


class REPLHandler(BaseHTTPRequestHandler):
    """HTTP handler for the REPL web API and UI."""

    api = APIServer()

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> str:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length).decode("utf-8")

    def _do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/sessions":
            self._send_json(self.api.session_list())
        elif path.startswith("/api/sessions/") and path.endswith("/history"):
            sid = path.split("/")[3]
            self._send_json(self.api.history(sid))
        elif path.startswith("/api/sessions/") and path.endswith("/info"):
            sid = path.split("/")[3]
            self._send_json(self.api.info(sid))
        elif path.startswith("/api/sessions/") and path.endswith("/delete"):
            sid = path.split("/")[3]
            self._send_json(self.api.delete(sid))
        elif path == "/":
            self._serve_ui()
        elif path == "/static/app.js":
            self._serve_static("app.js", "text/javascript")
        elif path == "/static/style.css":
            self._serve_static("style.css", "text/css")
        else:
            self._send_json({"error": "not found"}, 404)

    def _do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/sessions":
            body = self._read_body()
            if body == "" or body.strip() == "{}":
                self._send_json(self.api.create_session())
            return

        if path.startswith("/api/sessions/") and path.endswith("/execute"):
            sid = path.split("/")[3]
            body = json.loads(self._read_body())
            input_text = body.get("input", "").strip()
            if not input_text:
                self._send_json({"error": "empty input"}, 400)
                return
            result = self.api.execute(sid, input_text)
            self._send_json(result)
            return

        self._send_json({"error": "not found"}, 404)

    def _serve_ui(self) -> None:
        html_path = Path(__file__).parent / "templates" / "index.html"
        if html_path.exists():
            html = html_path.read_text(encoding="utf-8")
        else:
            html = self._fallback_html()

        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, filename: str, content_type: str) -> None:
        static_dir = Path(__file__).parent / "templates" / "static"
        filepath = static_dir / filename
        if filepath.exists():
            data = filepath.read_bytes()
        else:
            data = b""
        self.send_response(200 if data else 404)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _fallback_html(self) -> str:
        return '<!DOCTYPE html><html><head><title>Trail REPL</title></head>' \
               '<body><h1>Trail REPL Web Console</h1>' \
               '<p>Template not found — check trail/web/templates/</p></body></html>'


def start_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Start the HTTP server on the given host:port."""
    server = HTTPServer((host, port), REPLHandler)
    print(f"Trail REPL Web Console running on http://{host}:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()
