"""OS-level isolation for headless scene-agent sessions (bubblewrap).

The agent may run any shell command, so permission rules alone cannot keep it from reading the project (docs,
configs, other runs, test scenes, eval code, Claude memory and transcripts). Inside the sandbox the filesystem holds
only what the session needs:
    the run dir (read-write, at its real path, so the Claude session key is unchanged), with its _driver/ hidden
    the tool code, copied into a runtime dir and mounted at `mount` (only the listed files; nothing else of the repo)
    the robot meshes and other read-only `binds` (python env, tracker checkpoint, ...)
    the system (/usr, /etc), and a private /tmp
    ~/.claude with its credentials and settings, but with projects/ (per-run), history, sessions, skills, ... hidden
    ~/.claude.json copied per run without its per-project history
Network and everything else are left as they are.

    python -m humanoid_datagen.datagen_pipelines.agents.sandbox   # print the prefix, run a probe
"""
from __future__ import annotations

import json
import os
import shutil

from humanoid_datagen.utils.paths import repo_path

HIDE_CLAUDE = ["file-history", "paste-cache", "session-env", "sessions", "shell-snapshots", "backups", "debug",
               "skills", "ide", "plans", "todos", "downloads"]


def build_runtime(iso: dict) -> str:
    """Copy the listed repo files into a fresh runtime dir (rebuilt every batch, so it never goes stale)."""
    rt = repo_path(iso["runtime_dir"])
    if os.path.isdir(rt):
        shutil.rmtree(rt)
    for f in iso["code"]:
        dst = os.path.join(rt, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(repo_path(f), dst)
    for f in iso.get("empty") or []:  # package __init__ files whose repo docstrings describe the project
        dst = os.path.join(rt, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "w").close()
    if iso.get("library_index"):  # asset library index (uids, categories, sizes, CLIP features)
        from humanoid_datagen.datagen_pipelines.shared.library import export_index
        excl = set()
        if iso.get("library_exclude"):  # the test set's held-out assets (and their alternates) never reach the agents
            split = json.load(open(repo_path(iso["library_exclude"])))
            excl = {a["uid"] for k in ("held_out", "alternates") for t in split.get(k, {}).values() for a in t}
            print(f"[sandbox] library: {len(excl)} held-out assets excluded ({iso['library_exclude']})")
        export_index(repo_path(iso["library_index"]), os.path.join(rt, "library"), exclude=excl,
                     assets_root=repo_path(iso["library_assets"]) if iso.get("library_assets") else None)
    for dst in (iso.get("binds") or {}).values():  # mount points inside the runtime dir
        if dst and dst.startswith(iso["mount"] + "/"):
            os.makedirs(os.path.join(rt, os.path.relpath(dst, iso["mount"])), exist_ok=True)
    return rt


SETTINGS_MOUNT = "/opt/agent_settings.json"  # the session's --settings file (its source sits in the hidden _driver/)


def prefix(iso: dict, run_dir: str, drv: str, settings_path: str | None = None) -> list[str]:
    """bwrap argv to put in front of the session command."""
    home = os.path.expanduser("~")
    proj = os.path.join(drv, "claude_projects")
    os.makedirs(proj, exist_ok=True)
    cj = os.path.join(drv, "claude.json")
    try:
        with open(os.path.join(home, ".claude.json")) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        d = {}
    for k in ("projects", "githubRepoPaths"):
        d.pop(k, None)
    with open(cj, "w") as f:
        json.dump(d, f)
    a = ["bwrap", "--die-with-parent", "--unshare-pid",
         "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
         "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
         "--ro-bind", "/etc", "/etc", "--dev", "/dev", "--dev-bind-try", "/dev/dri", "/dev/dri", "--proc", "/proc",
         "--tmpfs", "/tmp", "--tmpfs", "/run",
         "--tmpfs", home]
    for p in iso["home_ro"]:
        a += ["--ro-bind", os.path.join(home, p), os.path.join(home, p)]
    c = os.path.join(home, ".claude")
    a += ["--bind", c, c, "--bind", proj, os.path.join(c, "projects"), "--bind", cj, os.path.join(home, ".claude.json"),
          "--ro-bind-try", "/dev/null", os.path.join(c, "history.jsonl")]
    for h in HIDE_CLAUDE:
        if os.path.isdir(os.path.join(c, h)):
            a += ["--tmpfs", os.path.join(c, h)]
    a += ["--ro-bind", repo_path(iso["runtime_dir"]), iso["mount"]]
    for src, dst in (iso.get("binds") or {}).items():  # dst null: same path inside the sandbox
        a += ["--ro-bind", repo_path(src), dst or repo_path(src)]
    # the run dir, minus the driver's own files (launch command, policy, arm): mounted over with an empty dir
    a += ["--bind", run_dir, run_dir, "--tmpfs", drv, "--chdir", run_dir]
    if settings_path:
        a += ["--ro-bind", settings_path, SETTINGS_MOUNT]
    for k, v in (iso.get("env") or {}).items():
        a += ["--setenv", k, str(v)]
    return a + ["--"]


def main():
    import subprocess
    import tempfile
    from omegaconf import OmegaConf
    from humanoid_datagen.utils.config import load_config
    cfg = OmegaConf.to_container(load_config("datagen/push/agents/scenes").stage, resolve=True)
    iso = cfg["claude"]["isolate"]
    build_runtime(iso)
    run = tempfile.mkdtemp(prefix="sandbox_probe_", dir=repo_path("tmp"))
    drv = os.path.join(run, "_driver")
    os.makedirs(drv)
    probe = ("set -x; ls /; ls ~; ls ~/.claude; ls ~/.claude/projects ~/.claude/skills; "
             f"ls {repo_path('.')}; cat {repo_path('docs/push_box.md')} | head -2; "
             "python -c 'import humanoid_datagen.datagen_pipelines.agents.cli as c, mujoco, onnxruntime; print(\"tools ok\", c.__file__)'")
    subprocess.run(prefix(iso, run, drv) + ["bash", "-c", probe])
    print("probe run dir:", run)


if __name__ == "__main__":
    main()
