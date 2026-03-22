"""Shared RunPod pod lifecycle management for ComfyUI strategies."""

import atexit
import json
import os
import signal
import subprocess
import sys
import time


# Default GPU preferences (cheapest first)
GPU_TYPES = [
    ("NVIDIA GeForce RTX 4090", 0.34),
    ("NVIDIA RTX 4000 Ada Generation", 0.34),
    ("NVIDIA RTX A6000", 0.52),
    ("NVIDIA L40S", 0.54),
]

DOCKER_IMAGE = "runpod/comfyui:latest"
CONTAINER_DISK_GB = 50
COMFYUI_PORT = 8188
POD_READY_TIMEOUT = 600
COMFYUI_READY_TIMEOUT = 600
COMFYUI_DIR = "/workspace/runpod-slim/ComfyUI"


class RunPodSession:
    """Manages a RunPod ComfyUI pod with state file persistence.

    Can create a new pod or reconnect to an existing one via a state file.
    Handles cleanup on exit (terminate or keep-alive depending on flags).
    """

    def __init__(self, output_dir: str, keep_pod: bool = False):
        self.output_dir = output_dir
        self.keep_pod = keep_pod
        self.pod_id: str | None = None
        self.base_url: str | None = None
        self.ssh_host: str | None = None
        self.ssh_port: int | None = None
        self.gpu_hourly_rate: float = GPU_TYPES[0][1]
        self.pod_start_time: float | None = None
        self._clean_exit = False
        self._cleanup_registered = False

    def _register_cleanup(self):
        """Register atexit and signal handlers to terminate pod on exit."""
        if self._cleanup_registered:
            return
        self._cleanup_registered = True

        atexit.register(self._cleanup)

        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _handler(signum, frame):
            self._cleanup()
            if signum == signal.SIGINT and callable(original_sigint):
                original_sigint(signum, frame)
            elif signum == signal.SIGTERM and callable(original_sigterm):
                original_sigterm(signum, frame)
            else:
                sys.exit(1)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _cleanup(self):
        """Terminate or preserve pod depending on keep_pod flag and exit status."""
        if self.pod_id is None:
            return
        if self.keep_pod and self._clean_exit:
            # Normal exit with --keep-pod: write state file, don't terminate
            self.write_state()
            print(f"  Pod {self.pod_id} kept alive (state written to {self.state_path})")
        else:
            self.terminate()

    @property
    def state_path(self) -> str:
        return os.path.join(self.output_dir, "runpod_pod.json")

    def mark_clean_exit(self):
        """Call before normal return to signal that pod should be kept."""
        self._clean_exit = True

    # -- State file I/O --

    def write_state(self):
        """Write pod connection info to state file (atomic)."""
        state = {
            "pod_id": self.pod_id,
            "base_url": self.base_url,
            "ssh_host": self.ssh_host,
            "ssh_port": self.ssh_port,
            "gpu_hourly_rate": self.gpu_hourly_rate,
            "start_time": self.pod_start_time,
        }
        os.makedirs(self.output_dir, exist_ok=True)
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, self.state_path)

    def read_state(self) -> bool:
        """Read pod state file. Returns True if state was loaded."""
        if not os.path.exists(self.state_path):
            return False
        with open(self.state_path) as f:
            state = json.load(f)
        self.pod_id = state["pod_id"]
        self.base_url = state["base_url"]
        self.ssh_host = state.get("ssh_host")
        self.ssh_port = state.get("ssh_port")
        self.gpu_hourly_rate = state.get("gpu_hourly_rate", GPU_TYPES[0][1])
        self.pod_start_time = state.get("start_time")
        return True

    def remove_state(self):
        """Remove the pod state file."""
        if os.path.exists(self.state_path):
            os.remove(self.state_path)

    # -- Pod lifecycle --

    def create_pod(self):
        """Create a new RunPod pod and wait for it to be ready."""
        import runpod

        runpod.api_key = os.environ.get("RUNPOD_API_KEY")
        if not runpod.api_key:
            print("Error: RUNPOD_API_KEY not set in .env or environment.")
            sys.exit(1)

        pod = None
        gpu_rate = GPU_TYPES[0][1]
        for gpu_type, rate in GPU_TYPES:
            print(f"  Trying {gpu_type}...")
            try:
                ssh_pubkey = ""
                pubkey_path = os.path.expanduser("~/.ssh/id_ed25519.pub")
                if not os.path.exists(pubkey_path):
                    pubkey_path = os.path.expanduser("~/.ssh/id_rsa.pub")
                if os.path.exists(pubkey_path):
                    with open(pubkey_path) as f:
                        ssh_pubkey = f.read().strip()

                pod = runpod.create_pod(
                    name="lossy-comfyui",
                    image_name=DOCKER_IMAGE,
                    gpu_type_id=gpu_type,
                    cloud_type="COMMUNITY",
                    gpu_count=1,
                    container_disk_in_gb=CONTAINER_DISK_GB,
                    ports=f"{COMFYUI_PORT}/http,22/tcp",
                    support_public_ip=True,
                    start_ssh=True,
                    env={"PUBLIC_KEY": ssh_pubkey} if ssh_pubkey else None,
                )
                gpu_rate = rate
                print(f"  Got {gpu_type} @ ${rate}/hr")
                break
            except Exception as e:
                print(f"  {gpu_type} unavailable: {e}")
                continue

        if pod is None:
            print("Error: No GPU available. Try again later.")
            sys.exit(1)

        self.pod_id = pod["id"]
        self.gpu_hourly_rate = gpu_rate
        self.pod_start_time = time.time()
        self._register_cleanup()
        print(f"  Pod created: {self.pod_id}")

        # Wait for runtime
        print("  Waiting for pod to start...")
        start = time.time()
        while time.time() - start < POD_READY_TIMEOUT:
            status = runpod.get_pod(self.pod_id)
            runtime = status.get("runtime")
            if runtime is not None and runtime.get("ports"):
                break
            time.sleep(5)
        else:
            print(f"  Error: Pod did not start within {POD_READY_TIMEOUT}s")
            self.terminate()
            sys.exit(1)

        self.base_url = f"https://{self.pod_id}-{COMFYUI_PORT}.proxy.runpod.net"
        print(f"  Pod ready: {self.base_url}")

        # Get SSH info
        pod_info = runpod.get_pod(self.pod_id)
        for port_info in pod_info.get("runtime", {}).get("ports", []):
            if port_info["privatePort"] == 22 and port_info["isIpPublic"]:
                self.ssh_host = port_info["ip"]
                self.ssh_port = port_info["publicPort"]
                break

    def reconnect(self) -> bool:
        """Reconnect to an existing pod via state file. Returns True if successful."""
        if not self.read_state():
            return False

        self._register_cleanup()
        print(f"  Reconnecting to pod {self.pod_id}...")

        # Verify pod is still running
        import runpod
        runpod.api_key = os.environ.get("RUNPOD_API_KEY")
        if not runpod.api_key:
            print("Error: RUNPOD_API_KEY not set in .env or environment.")
            sys.exit(1)

        try:
            status = runpod.get_pod(self.pod_id)
            runtime = status.get("runtime")
            if runtime is None or not runtime.get("ports"):
                print(f"  Pod {self.pod_id} is no longer running.")
                self.remove_state()
                return False
        except Exception as e:
            print(f"  Failed to check pod {self.pod_id}: {e}")
            self.remove_state()
            return False

        print(f"  Reconnected: {self.base_url}")
        return True

    def ensure_pod(self):
        """Reconnect to existing pod or create a new one."""
        if self.pod_id is not None:
            return
        if not self.reconnect():
            self.create_pod()

    def terminate(self):
        """Terminate the pod and remove state file."""
        if self.pod_id is None:
            return

        import runpod

        pod_id = self.pod_id
        self.pod_id = None  # Prevent double-terminate

        elapsed_h = (time.time() - self.pod_start_time) / 3600 if self.pod_start_time else 0
        estimated_cost = elapsed_h * self.gpu_hourly_rate

        try:
            runpod.api_key = os.environ.get("RUNPOD_API_KEY")
            runpod.terminate_pod(pod_id)
            print(f"  Pod {pod_id} terminated. Session cost: ~${estimated_cost:.2f} ({elapsed_h:.1f}hrs)")
        except Exception as e:
            print(f"  Warning: Failed to terminate pod {pod_id}: {e}")
            print(f"  Manually terminate at https://www.runpod.io/console/pods")

        self.remove_state()

    # -- SSH helpers --

    def ssh_cmd(self, cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
        """Run a command on the pod via SSH."""
        return subprocess.run(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-o", "ServerAliveInterval=30",
                "-p", str(self.ssh_port),
                f"root@{self.ssh_host}",
                cmd,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def ssh_bg(self, cmd: str):
        """Run a command on the pod via SSH in the background (detached)."""
        p = subprocess.Popen(
            [
                "ssh", "-f",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-p", str(self.ssh_port),
                f"root@{self.ssh_host}",
                cmd,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()

    # -- ComfyUI helpers --

    def wait_for_comfyui(self):
        """Poll ComfyUI until it responds to /system_stats."""
        import httpx

        print("  Waiting for ComfyUI...")
        start = time.time()
        while time.time() - start < COMFYUI_READY_TIMEOUT:
            try:
                resp = httpx.get(f"{self.base_url}/system_stats", timeout=10)
                if resp.status_code == 200:
                    print("  ComfyUI is ready.")
                    return
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
                pass
            elapsed = int(time.time() - start)
            print(f"  Waiting for ComfyUI... ({elapsed}s)", end="\r")
            time.sleep(5)

        print(f"\n  Error: ComfyUI did not become ready within {COMFYUI_READY_TIMEOUT}s")
        if self.ssh_host:
            try:
                result = self.ssh_cmd("tail -50 /tmp/comfyui.log", timeout=10)
                if result.stdout:
                    print(f"  ComfyUI log:\n{result.stdout}")
            except Exception:
                pass
        self.terminate()
        sys.exit(1)

    def restart_comfyui(self):
        """Stop ComfyUI, then start it again. Used after installing custom nodes."""
        print("  Restarting ComfyUI...")
        self.ssh_cmd('pkill -f "python main.py" || true', timeout=10)
        time.sleep(2)

        find_python = self.ssh_cmd(
            f"if [ -x {COMFYUI_DIR}/.venv/bin/python ]; then echo .venv/bin/python; "
            f"elif command -v python3 >/dev/null; then echo python3; "
            f"else echo python; fi",
            timeout=10,
        )
        python_bin = find_python.stdout.strip() if find_python.returncode == 0 else "python3"
        self.ssh_bg(
            f"cd {COMFYUI_DIR} && {python_bin} main.py --listen 0.0.0.0 --port 8188 "
            f"</dev/null >/tmp/comfyui.log 2>&1"
        )
        self.wait_for_comfyui()

    def download_models(self, models: list[tuple[str, str]]):
        """Download model files to ComfyUI models dir via SSH.

        Args:
            models: List of (relative_dest_path, url) tuples.
                    e.g. ("vae/wan_2.1_vae.safetensors", "https://...")
        """
        models_dir = f"{COMFYUI_DIR}/models"
        print(f"  Downloading models via SSH ({self.ssh_host}:{self.ssh_port})...")

        for dest_path, url in models:
            full_path = f"{models_dir}/{dest_path}"
            filename = os.path.basename(dest_path)

            check = self.ssh_cmd(f'[ -s {full_path} ] && echo exists || echo missing')
            if check.returncode == 0 and "exists" in check.stdout:
                print(f"    {filename}: already exists")
                continue

            print(f"    {filename}: downloading...")
            dir_path = os.path.dirname(full_path)
            result = self.ssh_cmd(
                f"mkdir -p {dir_path} && wget -q -O {full_path} '{url}' && echo OK",
                timeout=600,
            )
            if result.returncode == 0 and "OK" in result.stdout:
                print(f"    {filename}: done")
            else:
                print(f"    {filename}: FAILED (exit {result.returncode})")
                if result.stderr:
                    print(f"      {result.stderr[:200]}")

    def submit_workflow(self, workflow: dict, timeout: int = 300) -> dict | None:
        """Submit a ComfyUI workflow and wait for completion.

        Returns the history entry for the prompt, or None on failure.
        """
        import httpx

        resp = httpx.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow},
            timeout=30,
        )
        if resp.status_code != 200:
            error_detail = resp.text[:300] if resp.text else "no body"
            print(f"  ComfyUI rejected workflow ({resp.status_code}): {error_detail}")
            return None

        prompt_id = resp.json()["prompt_id"]

        start = time.time()
        while time.time() - start < timeout:
            resp = httpx.get(
                f"{self.base_url}/history/{prompt_id}",
                timeout=10,
            )
            history = resp.json()
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("status_str") != "success":
                    print(f"  ComfyUI error: {status}")
                    return None
                return history[prompt_id]
            time.sleep(2)

        print(f"  Timeout waiting for workflow ({timeout}s)")
        return None

    def download_output(self, output_file: dict) -> bytes | None:
        """Download an output file from ComfyUI's /view endpoint."""
        import httpx

        try:
            resp = httpx.get(
                f"{self.base_url}/view",
                params={
                    "filename": output_file["filename"],
                    "subfolder": output_file.get("subfolder", ""),
                    "type": output_file.get("type", "output"),
                },
                timeout=60,
                follow_redirects=True,
            )
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            print(f"  Error downloading output: {e}")
            return None

    def free_vram(self):
        """Free cached VRAM from last generation (keep models loaded)."""
        import httpx
        try:
            httpx.post(f"{self.base_url}/free", json={"free_memory": True}, timeout=10)
        except Exception:
            pass
