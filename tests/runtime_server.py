"""Filho HTTP sintético exclusivo dos testes de QProcess; não é um runtime LLM."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
from threading import Thread
import time


parser = argparse.ArgumentParser()
parser.add_argument("--host", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--alias", required=True)
parser.add_argument("--model", required=True)
args, _ = parser.parse_known_args()
assert args.host == "127.0.0.1"
settings_path = Path(os.environ["OPENFRAME_TEST_CONFIG"])
requests_path = settings_path.with_suffix(".requests")


def settings():
    return json.loads(settings_path.read_text(encoding="utf-8"))


if settings().get("exit_early"):
    sys.exit(2)
time.sleep(settings().get("bind_delay", 0))
health_count = 0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *unused):
        pass

    def do_GET(self):
        global health_count
        options = settings()
        authenticated = self.headers.get("Authorization") == "Bearer " + os.environ["LLAMA_API_KEY"]
        with requests_path.open("a", encoding="utf-8") as record:
            record.write(json.dumps({"path": self.path, "authenticated": authenticated}) + "\n")
        if options.get("redirect"):
            self.respond(302, {}, {"Location": "http://example.invalid/proibido"})
        elif self.path == "/health":
            time.sleep(options.get("health_delay", 0))
            health_count += 1
            if options.get("always_loading") or health_count <= options.get("loading_requests", 0):
                self.respond(503, {"error": {"message": "Loading model"}})
            else:
                self.respond(200, options.get("health_body", {"status": "ok"}))
        elif self.path == "/v1/models":
            if not authenticated and not options.get("no_auth"):
                self.respond(401, {"error": "Unauthorized"})
            else:
                self.respond(200, {"data": [{"id": options.get("model_id", args.alias)}]})
        else:
            self.respond(404, {})

    def do_POST(self):
        options = settings()
        authenticated = self.headers.get("Authorization") == "Bearer " + os.environ["LLAMA_API_KEY"]
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with requests_path.open("a", encoding="utf-8") as record:
            record.write(json.dumps({"path": self.path, "authenticated": authenticated}) + "\n")
        time.sleep(options.get("generation_delay", 0))
        if not authenticated:
            self.respond(401, {})
        else:
            self.respond(200, {"choices": [{"message": {"content": options.get("summary", "Resumo sintético do documento de teste.")}, "finish_reason": "stop"}]})

    def respond(self, code, body, headers=None):
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


server = ThreadingHTTPServer((args.host, args.port), Handler)
server.daemon_threads = True


def stop_from_test():
    for line in sys.stdin:
        if line.strip() == "stop":
            server.shutdown()
            return


Thread(target=stop_from_test, daemon=True).start()
try:
    server.serve_forever(poll_interval=0.01)
finally:
    server.server_close()
