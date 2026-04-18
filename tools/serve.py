"""Dev server for lossy tools — serves project files + scan API."""

import http.server
import json
import os
import socketserver
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
except ValueError:
    PORT = 8000
SELECTIONS_PATH = os.path.join(PROJECT_ROOT, "docs", "blog", "selections.json")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blog_selections  # noqa: E402


def scan_project():
    """Find available source videos and their reconstructed strategies.

    Groups all output runs by their source media file. Each source gets a
    flat list of strategies (labelled as "run/strategy") so the comparator
    only needs one dropdown.
    """
    output_dir = os.path.join(PROJECT_ROOT, "output")

    # Collect all output runs with their strategies
    # keyed by source media path -> list of {name, path, output_path, has_prompts, has_speech}
    by_source = {}

    if os.path.isdir(output_dir):
        for name in sorted(os.listdir(output_dir)):
            dirpath = os.path.join(output_dir, name)
            if not os.path.isdir(dirpath):
                continue
            manifest_path = os.path.join(dirpath, "manifest.json")
            if not os.path.isfile(manifest_path):
                continue

            # Read source from manifest
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                source_info = manifest.get("source", {}) if isinstance(manifest, dict) else {}
                source_file = source_info.get("file", "unknown")
                source_path = source_info.get("path", f"media/{source_file}")
            except (json.JSONDecodeError, KeyError):
                source_file = "unknown"
                source_path = "unknown"

            has_prompts = os.path.isfile(os.path.join(dirpath, "prompts.json"))
            has_speech = os.path.isfile(os.path.join(dirpath, "speech_track.wav"))

            # Collect strategies from reconstructed videos and/or decode progress files
            seen_strategies = set()

            def _add_strategy(strategy_name, recon_path=None):
                if strategy_name in seen_strategies:
                    return
                seen_strategies.add(strategy_name)
                by_source.setdefault(source_file, {
                    "source_file": source_file,
                    "source_path": f"/{source_path}",
                    "strategies": [],
                })
                video_strat = strategy_name.split("+")[0] if "+" in strategy_name else strategy_name
                adj_dir = os.path.join(dirpath, "adjusted", video_strat)
                raw_clips_dir = os.path.join(dirpath, "clips", video_strat)
                if os.path.isdir(adj_dir):
                    clips_path = f"/output/{name}/adjusted/{video_strat}"
                elif os.path.isdir(raw_clips_dir):
                    clips_path = f"/output/{name}/clips/{video_strat}"
                else:
                    clips_path = None
                path = recon_path or clips_path or f"/output/{name}"
                by_source[source_file]["strategies"].append({
                    "name": strategy_name,
                    "label": f"{name} / {strategy_name}",
                    "path": path,
                    "clips_path": clips_path,
                    "output_path": f"/output/{name}",
                    "has_prompts": has_prompts,
                    "has_speech": has_speech,
                })

            # Strategies with stitched reconstructed videos
            # Filename format: {film}_reconstructed_{strategy}[+audio].mp4
            marker = "_reconstructed_"
            for fname in sorted(os.listdir(dirpath)):
                if marker in fname and fname.endswith(".mp4"):
                    strategy_name = fname[fname.index(marker) + len(marker):-len(".mp4")]
                    _add_strategy(strategy_name, recon_path=f"/output/{name}/{fname}")

            # Strategies with only a decode progress file (not yet stitched)
            for fname in sorted(os.listdir(dirpath)):
                if fname.startswith("decode_progress_") and fname.endswith(".json"):
                    strategy_name = fname[len("decode_progress_"):-len(".json")]
                    _add_strategy(strategy_name)

    sources = list(by_source.values())
    return {"sources": sources, "project_root": PROJECT_ROOT}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PROJECT_ROOT, **kwargs)

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw.decode() or "{}")

    def do_GET(self):
        if self.path == "/api/blog_selections":
            try:
                data = blog_selections.load_selections(SELECTIONS_PATH)
            except ValueError as e:
                self._send_json(500, {"error": str(e)})
                return
            self._send_json(200, data)
            return

        if self.path == "/api/scan":
            data = scan_project()
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
            return

        # Range request support for video seeking
        if "Range" in self.headers:
            self.handle_range_request()
            return

        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/blog_selections":
            try:
                entry = self._read_json_body()
                blog_selections.validate_entry(entry)
            except (ValueError, json.JSONDecodeError) as e:
                self._send_json(400, {"error": str(e)})
                return
            data = blog_selections.load_selections(SELECTIONS_PATH)
            blog_selections.upsert_entry(data, entry)
            blog_selections.save_selections(SELECTIONS_PATH, data)
            self._send_json(200, data)
            return
        self.send_error(404)

    def do_DELETE(self):
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        if parsed.path == "/api/blog_selections":
            q = parse_qs(parsed.query)
            film = (q.get("film") or [None])[0]
            shot_idx_raw = (q.get("shot_idx") or [None])[0]
            if not film or shot_idx_raw is None:
                self._send_json(400, {"error": "film and shot_idx query params required"})
                return
            try:
                shot_idx = int(shot_idx_raw)
            except ValueError:
                self._send_json(400, {"error": "shot_idx must be an int"})
                return
            data = blog_selections.load_selections(SELECTIONS_PATH)
            removed = blog_selections.delete_entry(data, film, shot_idx)
            if not removed:
                self._send_json(404, {"error": "no matching entry"})
                return
            blog_selections.save_selections(SELECTIONS_PATH, data)
            self._send_json(200, data)
            return
        self.send_error(404)

    def handle_range_request(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404)
            return

        file_size = os.path.getsize(path)
        range_header = self.headers["Range"]
        # Parse "bytes=start-end"
        try:
            range_spec = range_header.replace("bytes=", "")
            parts = range_spec.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1
        except (ValueError, IndexError):
            self.send_error(416)
            return

        if start >= file_size:
            self.send_error(416)
            return
        end = min(end, file_size - 1)
        length = end - start + 1

        self.send_response(206)
        ctype = self.guess_type(path)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", length)
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(remaining, 64 * 1024))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, format, *args):
        # Quieter logging — skip noisy asset requests
        parts = str(args[0]).split() if args else []
        path = parts[1] if len(parts) > 1 else ""
        if path.startswith("/output/") and "/keyframes/" in path:
            return
        super().log_message(format, *args)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    server = ThreadedHTTPServer(("", PORT), Handler)
    print(f"lossy dev server on http://localhost:{PORT} (threaded)")
    print(f"  comparator: http://localhost:{PORT}/tools/compare.html")
    print(f"  shot verify: http://localhost:{PORT}/tools/verify_shots.html")
    print(f"  project root: {PROJECT_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
