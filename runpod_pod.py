"""Shared RunPod pod lifecycle management for ComfyUI strategies."""

import atexit
import json
import os
import signal
import subprocess
import sys
import time
import uuid


# Default GPU preferences (cheapest first, with VRAM headroom preference)
GPU_TYPES = [
    ("NVIDIA GeForce RTX 4090", 0.34),   # 24 GB, fastest consumer GPU
    ("NVIDIA L40S", 0.54),               # 48 GB, Ada Lovelace
    ("NVIDIA RTX A6000", 0.33),          # 48 GB, Ampere (slower inference)
    ("NVIDIA RTX 4000 Ada Generation", 0.34),
]

# Community cloud is cheapest but its stock genuinely runs out -- a rehearsal
# slice found no capacity across all five 48GB types for ~15 minutes. Secure
# cloud is RunPod's own datacentres: pricier, far more reliably available. Try
# community first and fall back rather than failing the run.
CLOUD_TYPES = [t for t in os.environ.get("LOSSY_RUNPOD_CLOUD", "COMMUNITY,SECURE").split(",") if t]
CLOUD_TYPE = CLOUD_TYPES[0]  # back-compat for anything reading the old name
DOCKER_IMAGE = "runpod/comfyui:latest"

# Every pod this project creates is named POD_NAME_PREFIX + "-" + a unique
# suffix. The prefix makes lossy's pods findable; the suffix makes *this
# session's* pod distinguishable from a concurrent run's, which is what lets
# the stray sweep below terminate its own leftovers without shooting down
# somebody else's render. Pods created before unique naming are all called
# exactly POD_NAME_PREFIX -- matched, but never claimed as ours.
POD_NAME_PREFIX = "lossy-comfyui"
CONTAINER_DISK_GB = 50
COMFYUI_PORT = 8188
POD_READY_TIMEOUT = 600
COMFYUI_READY_TIMEOUT = 600
COMFYUI_DIR = "/workspace/runpod-slim/ComfyUI"

# How long a kept-alive pod survives with nobody using it. A pod bills until
# it is explicitly terminated, and once the local process exits there is
# nothing of ours left running to notice it went idle -- a dress-rehearsal
# render finished in 2.5hrs and then billed for another 10.5hrs ($3.40, more
# than the render) purely because the next stage was never started. So the
# deadline is armed *on the pod itself*, where it survives the process
# exiting, the session ending, and the machine sleeping.
KEEP_POD_DEFAULT_MINUTES = 30
REAPER_PID_FILE = "/workspace/.lossy_reaper.pid"

# Pod provisioning fails often enough that a long run cannot treat it as
# fatal. Across ten attempts in one rehearsal session, three pods died on
# CUDA init before ComfyUI started and one found no capacity on any GPU type
# -- each stalled the run until a human reissued the command.
POD_SETUP_ATTEMPTS = 4
POD_SETUP_BACKOFF_S = 30


class PodSetupError(RuntimeError):
    """A pod could not be provisioned or brought up. Retryable on a new pod."""


class RunPodSession:
    """Manages a RunPod ComfyUI pod with state file persistence.

    Can create a new pod or reconnect to an existing one via a state file.
    Handles cleanup on exit (terminate or keep-alive depending on flags).
    """

    def __init__(self, output_dir: str, keep_pod: bool | int = False,
                 gpu_types: list[tuple[str, float]] | None = None):
        self.output_dir = output_dir
        # keep_pod is a duration, not a switch: True means the default TTL,
        # an int means that many minutes. There is deliberately no way to say
        # "keep forever" -- that contract is what produced a 10.5hr idle bill.
        self.keep_pod = bool(keep_pod)
        self.keep_pod_minutes = (
            KEEP_POD_DEFAULT_MINUTES if keep_pod is True
            else int(keep_pod) if keep_pod else 0
        )
        # Cheapest-first GPU fallback list to try when creating a pod. Defaults
        # to the module-level GPU_TYPES; strategies with stricter VRAM needs
        # (e.g. a model that OOMs on a 24GB card) can pass a narrower list.
        self.gpu_types = gpu_types or GPU_TYPES
        # Unique per session, decided up front so it is stable across a
        # create/terminate cycle and recorded in the state file for reconnects.
        self.pod_name = f"{POD_NAME_PREFIX}-{uuid.uuid4().hex[:8]}"
        self.pod_id: str | None = None
        self.base_url: str | None = None
        self.ssh_host: str | None = None
        self.ssh_port: int | None = None
        self.gpu_hourly_rate: float = self.gpu_types[0][1]
        self.pod_start_time: float | None = None
        self._clean_exit = False
        self._cleanup_registered = False
        self._swept = False
        # Pod ids this session provisioned or adopted. Belt-and-braces
        # alongside the name check: either one identifying a pod as ours is
        # enough to terminate it.
        self._owned_pod_ids: set[str] = set()

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
        keeping = bool(self.keep_pod and self._clean_exit and self.pod_id is not None)

        if self.pod_id is not None:
            if keeping:
                # Normal exit with --keep-pod: write state file, don't terminate --
                # but arm a self-destruct on the pod first so "kept alive" cannot
                # silently mean "billing all night".
                self.write_state()
                armed = self.arm_self_destruct(self.keep_pod_minutes)
                note = (f"self-destructs in {self.keep_pod_minutes}min" if armed
                        else "WARNING: self-destruct could NOT be armed -- terminate it yourself")
                print(f"  Pod {self.pod_id} kept alive ({note}); state at {self.state_path}")
            else:
                self.terminate()

        # A deliberately kept-alive pod is not a stray -- don't sweep it away.
        if keeping or self._swept:
            return
        self._swept = True
        self.sweep_stray_pods()

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
            "pod_name": self.pod_name,
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
        # Pods created before unique naming have no name in their state file;
        # fall back to the bare prefix they were actually created with.
        self.pod_name = state.get("pod_name") or POD_NAME_PREFIX
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
        gpu_rate = self.gpu_types[0][1]
        # Cheapest cloud first, then each GPU within it; only escalate to the
        # pricier cloud once the cheap one is genuinely out of everything.
        attempts = [(c, g, r) for c in CLOUD_TYPES for g, r in self.gpu_types]
        for cloud_type, gpu_type, rate in attempts:
            label = gpu_type if cloud_type == CLOUD_TYPES[0] else f"{gpu_type} [{cloud_type}]"
            print(f"  Trying {label}...")
            try:
                ssh_pubkey = ""
                pubkey_path = os.path.expanduser("~/.ssh/id_ed25519.pub")
                if not os.path.exists(pubkey_path):
                    pubkey_path = os.path.expanduser("~/.ssh/id_rsa.pub")
                if os.path.exists(pubkey_path):
                    with open(pubkey_path) as f:
                        ssh_pubkey = f.read().strip()

                pod = runpod.create_pod(
                    name=self.pod_name,
                    image_name=DOCKER_IMAGE,
                    gpu_type_id=gpu_type,
                    cloud_type=cloud_type,
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
                print(f"  {label} unavailable: {e}")
                continue

        if pod is None:
            raise PodSetupError(
                f"no capacity on any of {len(self.gpu_types)} GPU types "
                f"across {len(CLOUD_TYPES)} cloud(s)"
            )

        pod_id = str(pod["id"])
        self.pod_id = pod_id
        self._owned_pod_ids.add(pod_id)
        self.gpu_hourly_rate = gpu_rate
        self.pod_start_time = time.time()
        self._register_cleanup()
        print(f"  Pod created: {self.pod_id}")

        # Wait for runtime
        print("  Waiting for pod to start...")
        start = time.time()
        while time.time() - start < POD_READY_TIMEOUT:
            status = runpod.get_pod(self.pod_id)
            if status is None:
                time.sleep(5)
                continue
            runtime = status.get("runtime")
            if runtime is not None and runtime.get("ports"):
                break
            time.sleep(5)
        else:
            print(f"  Error: Pod did not start within {POD_READY_TIMEOUT}s")
            self.terminate()
            # Retryable: a host that never comes up is the same class of
            # problem as one whose CUDA fails -- a fresh pod usually works.
            raise PodSetupError(f"pod did not start within {POD_READY_TIMEOUT}s")

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
        if self.pod_id:
            self._owned_pod_ids.add(self.pod_id)
        # This pod was left alive with a deadline armed; clear it so it does
        # not terminate underneath the run that just picked it up.
        self.disarm_self_destruct()
        return True

    def with_setup_retry(self, setup_fn, attempts: int = POD_SETUP_ATTEMPTS) -> None:
        """Run a pod-setup routine, re-provisioning onto a fresh pod on failure.

        Provisioning is unreliable enough that a long run cannot treat it as
        fatal: in one rehearsal session three of ten pods died on CUDA init
        before ComfyUI started and one attempt found no capacity at all, and
        every one of them stalled the run until a human reissued the command.

        Each retry starts from scratch -- the failed pod is already terminated
        by the time PodSetupError is raised, and the stale state file is
        cleared so the next attempt provisions rather than reconnects.
        """
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                setup_fn()
                return
            except PodSetupError as e:
                last = e
                self.pod_id = None
                self.base_url = None
                self.ssh_host = None
                self.remove_state()
                if attempt < attempts:
                    wait = POD_SETUP_BACKOFF_S * attempt
                    print(f"  Pod setup failed ({e}). "
                          f"Retrying on a fresh pod in {wait}s "
                          f"[attempt {attempt + 1}/{attempts}]...")
                    time.sleep(wait)
        print(f"Error: pod setup failed {attempts} times; giving up.")
        raise last if last else PodSetupError("pod setup failed")

    def arm_self_destruct(self, minutes: int) -> bool:
        """Schedule the pod to terminate itself after `minutes`. Returns success.

        Runs entirely on the pod: a detached sleep-then-terminate, using the
        RUNPOD_POD_ID that RunPod injects, so the pod removes only itself and
        no account API key is ever copied onto a community-cloud host we do
        not control.

        This is the only cleanup that survives the local process exiting, so
        it is the backstop for every other mechanism -- atexit handlers,
        signal handlers and session watchdogs all die with their process.
        """
        if not self.ssh_host or minutes <= 0:
            return False
        self.disarm_self_destruct()  # never stack two reapers
        script = (
            f"nohup setsid sh -c 'sleep {minutes * 60}; "
            f"runpodctl remove pod $RUNPOD_POD_ID' "
            f">/tmp/lossy_reaper.log 2>&1 & echo $! > {REAPER_PID_FILE}"
        )
        try:
            r = self.ssh_cmd(script, timeout=30)
            return r.returncode == 0
        except Exception as e:
            print(f"  Failed to arm pod self-destruct: {e}")
            return False

    def disarm_self_destruct(self) -> None:
        """Cancel a pending self-destruct, e.g. because a new run reconnected.

        Without this, resuming onto a kept-alive pod would inherit the old
        deadline and the pod would vanish mid-render.
        """
        if not self.ssh_host:
            return
        try:
            self.ssh_cmd(
                f"[ -f {REAPER_PID_FILE} ] && kill $(cat {REAPER_PID_FILE}) 2>/dev/null; "
                f"rm -f {REAPER_PID_FILE}; true",
                timeout=30,
            )
        except Exception:
            pass  # a stale reaper is worth a warning, not a crash

    def ensure_pod(self):
        """Reconnect to existing pod or create a new one."""
        if self.pod_id is not None:
            return
        if not self.reconnect():
            self.create_pod()

    def _is_ours(self, pod: dict) -> bool:
        """Whether a pod belongs to this session.

        The name is the load-bearing test, not the id: the failure this whole
        sweep exists for is precisely the recorded id going stale, so an
        id-only check would miss the case it is meant to catch.
        """
        return pod.get("name") == self.pod_name or pod.get("id") in self._owned_pod_ids

    def sweep_stray_pods(self) -> list[str]:
        """Terminate this session's leftover pods; report anyone else's.

        terminate() removes the pod id it recorded and warns if that call
        fails -- but a run has been observed reporting "pod not found to
        terminate" while a pod under a *different* id was still up and
        billing. Nothing catches that today: a pod bills until something
        explicitly removes it, and once this process exits there is nobody
        left to notice. So before exiting, ask what is actually still running.

        Pods whose name matches this session are terminated outright. Anything
        else carrying the lossy prefix is another run's (or an older orphan's)
        and is only reported -- killing it would take down a render that is
        very likely someone's active work. Set LOSSY_RUNPOD_SWEEP=1 to
        terminate those too.

        Returns the ids of the stray pods found, for tests and callers.
        """
        if not self._owned_pod_ids:
            return []  # never provisioned anything, so nothing can have escaped
        if not os.environ.get("RUNPOD_API_KEY"):
            return []

        try:
            import runpod
            runpod.api_key = os.environ["RUNPOD_API_KEY"]
            pods = runpod.get_pods() or []
        except Exception as e:
            print(f"  Stray-pod check failed ({e}) -- "
                  f"verify at https://www.runpod.io/console/pods")
            return []

        strays = [
            p for p in pods
            if str(p.get("name", "")).startswith(POD_NAME_PREFIX)
            and p.get("desiredStatus", "RUNNING") != "EXITED"
            and p.get("id")
        ]
        if not strays:
            return []

        force = os.environ.get("LOSSY_RUNPOD_SWEEP", "").lower() in ("1", "true", "yes")
        for pod in strays:
            pod_id = pod["id"]
            ours = self._is_ours(pod)
            if ours or force:
                why = "ours" if ours else "LOSSY_RUNPOD_SWEEP=1"
                print(f"  Sweep: terminating stray pod {pod_id} ({why})")
                try:
                    runpod.terminate_pod(pod_id)
                except Exception as e:
                    print(f"  Sweep: FAILED to terminate {pod_id} ({e}) -- "
                          f"terminate it at https://www.runpod.io/console/pods")
            else:
                print(f"  NOTE: pod {pod_id} ({pod.get('name')}) is still running and "
                      f"billing, but is not this session's.")
                print(f"        Probably a concurrent run. If it is orphaned, terminate it "
                      f"at https://www.runpod.io/console/pods (or re-run with "
                      f"LOSSY_RUNPOD_SWEEP=1).")
        return [p["id"] for p in strays]

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

    def _ssh_opts(self) -> list[str]:
        """Common SSH options for all remote commands."""
        return [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
        ]

    def ssh_cmd(self, cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
        """Run a command on the pod via SSH."""
        return subprocess.run(
            ["ssh", *self._ssh_opts(), "-o", "ServerAliveInterval=30",
             "-p", str(self.ssh_port), f"root@{self.ssh_host}", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def ssh_bg(self, cmd: str):
        """Run a command on the pod via SSH in the background (detached)."""
        p = subprocess.Popen(
            ["ssh", "-f", *self._ssh_opts(),
             "-p", str(self.ssh_port), f"root@{self.ssh_host}", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()

    def scp_to(self, local_paths: list[str], remote_dir: str, timeout: int = 60) -> subprocess.CompletedProcess:
        """Copy local files to a directory on the pod via scp."""
        return subprocess.run(
            ["scp", *self._ssh_opts(),
             "-P", str(self.ssh_port),
             *local_paths,
             f"root@{self.ssh_host}:{remote_dir}/"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

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
        # Retryable: this is overwhelmingly a bad host (CUDA failing to
        # initialise before ComfyUI starts), and a fresh pod usually works.
        raise PodSetupError("ComfyUI did not come up on this pod")

    def restart_comfyui(self):
        """Stop ComfyUI, then start it again. Used after installing custom nodes."""
        print("  Restarting ComfyUI...")
        self.ssh_cmd('pkill -f "main.py" || true', timeout=10)
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

    def download_models(self, models: list[tuple[str, str]], timeout: int = 600):
        """Download model files to ComfyUI models dir via SSH.

        Args:
            models: List of (relative_dest_path, url) tuples.
                    e.g. ("vae/wan_2.1_vae.safetensors", "https://...")
            timeout: Per-file SSH command timeout in seconds. Default 600s covers
                     the ~10GB Wan model files; larger checkpoints (e.g. LTX-2's
                     22B FP8, ~22GB) need a longer allowance.
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
                timeout=timeout,
            )
            if result.returncode == 0 and "OK" in result.stdout:
                print(f"    {filename}: done")
            else:
                print(f"    {filename}: FAILED (exit {result.returncode})")
                if result.stderr:
                    print(f"      {result.stderr[:200]}")

    def submit_workflow(self, workflow: dict, timeout: int = 600) -> dict | None:
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
