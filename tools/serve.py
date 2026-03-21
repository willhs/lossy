"""Dev server for lossy tools — serves project files + scan API."""

import http.server
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


def scan_project():
    """Find available output directories and source videos."""
    outputs = []
    output_dir = os.path.join(PROJECT_ROOT, "output")
    if os.path.isdir(output_dir):
        for name in sorted(os.listdir(output_dir)):
            dirpath = os.path.join(output_dir, name)
            if not os.path.isdir(dirpath):
                continue
            manifest = os.path.join(dirpath, "manifest.json")
            if not os.path.isfile(manifest):
                continue
            entry = {"name": name, "path": f"/output/{name}"}
            # Find all reconstructed videos (per-strategy and legacy)
            strategies = []
            for fname in sorted(os.listdir(dirpath)):
                if fname.startswith("reconstructed") and fname.endswith(".mp4"):
                    if fname == "reconstructed.mp4":
                        strategy_name = "unknown"
                    else:
                        # reconstructed_fal-seedance.mp4 -> fal-seedance
                        strategy_name = fname[len("reconstructed_"):-len(".mp4")]
                    strategies.append({
                        "name": strategy_name,
                        "path": f"/output/{name}/{fname}",
                    })
            if strategies:
                entry["strategies"] = strategies
                entry["reconstructed"] = strategies[0]["path"]
            # Check for prompts
            if os.path.isfile(os.path.join(dirpath, "prompts.json")):
                entry["has_prompts"] = True
            # Check for speech track
            if os.path.isfile(os.path.join(dirpath, "speech_track.wav")):
                entry["has_speech"] = True
            outputs.append(entry)

    sources = []
    media_dir = os.path.join(PROJECT_ROOT, "media")
    if os.path.isdir(media_dir):
        for name in sorted(os.listdir(media_dir)):
            if name.endswith((".mp4", ".mkv", ".mov", ".avi")):
                sources.append({"name": name, "path": f"/media/{name}"})

    return {"outputs": outputs, "sources": sources}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PROJECT_ROOT, **kwargs)

    def do_GET(self):
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
        path = args[0].split()[1] if args else ""
        if path.startswith("/output/") and "/keyframes/" in path:
            return
        super().log_message(format, *args)


if __name__ == "__main__":
    server = http.server.HTTPServer(("", PORT), Handler)
    print(f"lossy dev server on http://localhost:{PORT}")
    print(f"  comparator: http://localhost:{PORT}/tools/compare.html")
    print(f"  shot verify: http://localhost:{PORT}/tools/verify_shots.html")
    print(f"  project root: {PROJECT_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
