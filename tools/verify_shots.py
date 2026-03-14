#!/usr/bin/env python3
"""Tiny server to verify shot boundaries against the source video."""

import http.server
import json
import os
import socketserver
import sys
import urllib.parse

PORT = 8787
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_DIR, "output/star_wars_iv")
MEDIA_DIR = os.path.join(PROJECT_DIR, "media")


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            with open(os.path.join(SCRIPT_DIR, "verify_shots.html"), "rb") as f:
                self.wfile.write(f.read())
        elif path == "/manifest.json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            with open(os.path.join(OUTPUT_DIR, "manifest.json"), "rb") as f:
                self.wfile.write(f.read())
        elif path.startswith("/keyframes/"):
            filepath = os.path.join(OUTPUT_DIR, path[1:])
            if os.path.exists(filepath):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        elif path.startswith("/media/"):
            filepath = os.path.join(PROJECT_DIR, path[1:])
            if os.path.exists(filepath):
                self._serve_video(filepath)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def _serve_video(self, filepath):
        """Serve video with Range request support for seeking."""
        file_size = os.path.getsize(filepath)
        range_header = self.headers.get("Range")

        if range_header:
            range_spec = range_header.replace("bytes=", "")
            start_str, end_str = range_spec.split("-")
            start = int(start_str)
            end = int(end_str) if end_str else min(start + 2 * 1024 * 1024, file_size - 1)
            end = min(end, file_size - 1)
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(filepath, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(filepath, "rb") as f:
                while chunk := f.read(64 * 1024):
                    self.wfile.write(chunk)

    def log_message(self, format, *args):
        # quieter logs - only show non-200 or non-206
        status = args[1] if len(args) > 1 else ""
        if str(status) not in ("200", "206"):
            super().log_message(format, *args)


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Shot boundary viewer: http://localhost:{PORT}")
        print(f"Serving manifest from: {OUTPUT_DIR}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
