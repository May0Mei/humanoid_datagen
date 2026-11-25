"""Headless push-scene batch: one fresh `claude -p "/push-scene ..."` session per prior, same flags for both arms.

    python humanoid_datagen/scripts/datagen.py --config_name datagen/push/agents/scenes        (video prior)
    python humanoid_datagen/scripts/datagen.py --config_name datagen/push/agents_text/scenes   (text prior)

Per run <out_root>/<run_id>/ (cwd of the session):
    prior/                 video.mp4 (push arm only) + wan.txt + kimodo.txt: the only input; motions/*.csv
    .claude/skills/push-scene/    snapshot of the skill (claude.copy_skill, claude.skill_name)
    _driver/               policy.json (arm, seed, read policy for the tools), settings.json, command.sh,
                           stdout.json, stderr.log, transcript.jsonl
    builds/ renders/ ...   written by the agent's tools; result.json is written by the `finalize` tool
Resumable: runs with a valid result.json are skipped; a launched run without one is archived to
<run_id>.attempt<k> and relaunched (up to max_attempts). The manifest is <out_root>/manifest.json; each finished
session is also logged to the cost ledger (utils/cost.py). (The walk-to build-scene arms, text briefs and
video generations, are in cache/walk_to/.)
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed


from humanoid_datagen.utils.paths import REPO_ROOT, repo_path
from humanoid_datagen.datagen_pipelines.shared.base import Stage

POLICY_ENV = "HDG_BUILD_SCENE_POLICY"
_lock = threading.Lock()


# ----------------------------------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------------------------------


def make_jobs(cfg: dict) -> list[dict]:
    arm, jc = cfg["arm"], cfg["jobs"]
    priors = []
    if arm in ("push", "push_text"):
        # push-scene arm: outputs/push_box/priors/<id>/ = Wan video + prompts + object card + Kimodo motions
        pc = jc["push"]
        root = repo_path(pc["root"])
        ids = pc.get("prior_ids") or sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        for pid in ids:
            d = os.path.join(root, pid)
            vids = sorted(f for f in os.listdir(d) if f.startswith(pc["video_prefix"]) and f.endswith(".mp4"))
            mot = sorted(f for f in os.listdir(os.path.join(d, "motions")) if f.endswith(".csv")) \
                if os.path.isdir(os.path.join(d, "motions")) else []
            with_video = pc.get("with_video", True)
            req = pc.get("require_file")  # e.g. motion_select.json: motions/ comes from select_motions, not stale
            if (with_video and not vids) or not mot or (req and not os.path.isfile(os.path.join(d, req))):
                print(f"[skip] {pid}: video {len(vids)}, motions {len(mot)}"
                      + (f", no {req}" if req and not os.path.isfile(os.path.join(d, req)) else ""))
                continue
            files = ([(os.path.join(d, vids[0]), "video.mp4")] if with_video else []) + \
                [(os.path.join(d, f), f) for f in pc["prompt_files"]]
            copy = [(os.path.join(d, "motions", f), os.path.join("motions", f)) for f in mot]
            priors.append({"prior_id": pid, "files": files, "copy": copy,
                           "meta": {"prior": pid, "video": vids[0] if with_video and vids else None}})
    else:
        raise SystemExit(f"arm must be push | push_text, not {arm!r}")
    jobs = [dict(p, arm=arm, seed=int(s), run_id=f"{p['prior_id']}_s{int(s)}") for p in priors for s in jc["seeds"]]
    return jobs[: jc["limit"]] if jc.get("limit") else jobs


# ----------------------------------------------------------------------------------------------
# run dir + command
# ----------------------------------------------------------------------------------------------


def _strip(p: str) -> str:
    return os.path.abspath(p).lstrip("/")


def result_ok(run_dir: str) -> dict | None:
    p = os.path.join(run_dir, "result.json")
    try:
        with open(p) as f:
            r = json.load(f)
        return r if r.get("finalized_by") == "cli finalize" else None
    except (OSError, json.JSONDecodeError):
        return None


def rules(cfg: dict, run_dir: str) -> tuple[list[str], list[str]]:
    arm = cfg["arm"]
    other = [a for a in cfg["arms"] if a != arm]
    fill = {"run_dir": _strip(run_dir), "repo": _strip(REPO_ROOT), "home": _strip(os.path.expanduser("~")),
            "other_arm_root": _strip(repo_path(cfg["arms"][other[0]]["out_root"])) if other else "nonexistent"}
    cc = cfg["claude"]
    allow = [r.format(**fill) for r in cc["allow"]]
    deny = [r.format(**fill) for r in cc["deny_both"]]
    deny += [r.format(**fill) for r in cc.get("deny_extra") or []]
    return allow, deny


def child_env(cfg: dict, policy_path: str) -> tuple[dict, list[str], dict]:
    env = dict(os.environ)
    unset = []
    if not cfg["env"]["use_api_key"]:
        for k in cfg["env"]["unset_when_no_key"]:
            if k in env:
                unset.append(k)
            env.pop(k, None)
    sets = {k: str(v) for k, v in (cfg["env"]["set"] or {}).items()}
    sets[POLICY_ENV] = policy_path
    env.update(sets)
    return env, unset, sets


def command(cfg: dict, run_dir: str, session_id: str, settings_path: str, resume: bool = False) -> list[str]:
    cc = cfg["claude"]
    allow, deny = rules(cfg, run_dir)
    prompt = (cfg["resume"]["prompt"] if resume else cc["prompt"]).format(arm=cfg["arm"], run_dir=run_dir)
    cmd = [cfg["claude_bin"], "-p", prompt, "--model", str(cc["model"]), "--effort", str(cc["effort"]),
           "--max-turns", str(cc["max_turns"]), "--max-budget-usd", str(cc["max_budget_usd"]),
           "--output-format", cc["output_format"], "--resume" if resume else "--session-id", session_id,
           "--permission-mode", cc["permission_mode"], "--tools", ",".join(cc["tools"]),
           "--settings", settings_path, "--allowedTools", *allow, "--disallowedTools", *deny]
    cmd += [str(a) for a in cc.get("extra_args") or []]
    iso = cc.get("isolate") or {}
    if iso.get("enabled"):  # _driver/ is hidden in the sandbox: the settings file is mounted read-only elsewhere
        from humanoid_datagen.datagen_pipelines.agents.sandbox import SETTINGS_MOUNT
        cmd[cmd.index(settings_path)] = SETTINGS_MOUNT
    if iso.get("enabled"):  # OS-level sandbox: only the run dir + tool runtime are visible (agent_sandbox.py)
        from humanoid_datagen.datagen_pipelines.agents.sandbox import prefix
        cmd = prefix(iso, run_dir, os.path.join(run_dir, "_driver"), settings_path) + cmd
    return cmd


def shell_line(cmd: list[str], cwd: str, unset: list[str], sets: dict) -> str:
    env = " ".join([f"-u {k}" for k in unset] + [f"{k}={shlex.quote(v)}" for k, v in sets.items()])
    return f"cd {shlex.quote(cwd)} && env {env} " + " ".join(shlex.quote(c) for c in cmd)


def prepare(job: dict, cfg: dict, attempt: int, resume_sid: str | None = None) -> dict:
    """Create / reset the run dir and return everything needed to launch it. With `resume_sid`, keep the run dir
    and continue that session (an attempt cut off by a usage limit) instead of starting over."""
    root = repo_path(cfg["arms"][job["arm"]]["out_root"])
    run_dir = os.path.join(root, job["run_id"])
    drv = os.path.join(run_dir, "_driver")
    if resume_sid:
        for fn in ("stdout.json", "stderr.log"):  # keep the cut-off attempt's driver output next to the resumed one
            src = os.path.join(drv, fn)
            if os.path.isfile(src):
                k = 1
                while os.path.exists(f"{src}.part{k}"):
                    k += 1
                os.replace(src, f"{src}.part{k}")
    elif os.path.isfile(os.path.join(drv, "launched")):  # an earlier attempt that never finalized: archive it
        k = 1
        while os.path.exists(f"{run_dir}.attempt{k}"):
            k += 1
        os.replace(run_dir, f"{run_dir}.attempt{k}")
    os.makedirs(os.path.join(run_dir, "prior"), exist_ok=True)
    os.makedirs(drv, exist_ok=True)
    for src, dst in job.get("files") or []:
        shutil.copyfile(src, os.path.join(run_dir, "prior", dst))
    if not resume_sid:  # fresh attempt: drop earlier copies (e.g. a dry run's motions/ from an older motion set)
        for sub in {os.path.dirname(dst) for _, dst in job.get("copy") or [] if os.path.dirname(dst)}:
            shutil.rmtree(os.path.join(run_dir, sub), ignore_errors=True)
    for src, dst in job.get("copy") or []:  # inputs outside prior/ (push arm: motions/)
        os.makedirs(os.path.dirname(os.path.join(run_dir, dst)), exist_ok=True)
        shutil.copyfile(src, os.path.join(run_dir, dst))
    cc = cfg["claude"]
    if cc.get("copy_skill"):
        dst = os.path.join(run_dir, ".claude", "skills", cc.get("skill_name") or os.path.basename(cc["copy_skill"].rstrip("/")))
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(repo_path(cc["copy_skill"]), dst)
    if cc.get("isolate_git_root") and not os.path.isdir(os.path.join(run_dir, ".git")):
        subprocess.run(["git", "init", "-q", run_dir], check=False, capture_output=True)
    # deterministic per (batch tag, arm, run, attempt); a new `session_tag` gives fresh ids when re-running the same jobs
    tag = cfg.get("session_tag") or ""
    session_id = resume_sid or str(uuid.uuid5(uuid.NAMESPACE_URL,
                                              f"build-scene/{tag}{job['arm']}/{job['run_id']}/attempt{attempt}"))
    scene_id = f"{cfg['arms'][job['arm']]['source']}_{job['run_id']}"
    policy = {"arm": job["arm"], "source": cfg["arms"][job["arm"]]["source"], "run_id": job["run_id"],
              "scene_id": scene_id, "run_dir": run_dir, "seed": job["seed"], "allow_video": job["arm"] == "video",
              "deny_read": [repo_path("outputs/generations"), repo_path("outputs/prompts"), repo_path("outputs/eval"),
                            repo_path("sim_scene"), repo_path("humanoid_datagen/scenes/specs")],
              "prior_id": job["prior_id"], "prior_meta": job.get("meta"), "session_id": session_id, "attempt": attempt}
    policy_path = os.path.join(drv, "policy.json")
    with open(policy_path, "w") as f:
        json.dump(policy, f, indent=1)
    settings_path = os.path.join(drv, "settings.json")
    with open(settings_path, "w") as f:
        json.dump(cc["settings"], f, indent=1)
    cmd = command(cfg, run_dir, session_id, settings_path, resume=bool(resume_sid))
    env, unset, sets = child_env(cfg, policy_path)
    line = shell_line(cmd, run_dir, unset, sets)
    with open(os.path.join(drv, "command.sh"), "w") as f:
        f.write("#!/bin/bash\n# exact launch command of this run (written by datagen_pipelines/agents/batch.py)\n" + line + "\n")
    return {"run_dir": run_dir, "cmd": cmd, "env": env, "line": line, "session_id": session_id, "scene_id": scene_id,
            "resumed": bool(resume_sid)}


# ----------------------------------------------------------------------------------------------
# launch + bookkeeping
# ----------------------------------------------------------------------------------------------


def parse_stdout(text: str) -> dict:
    text = text.strip()
    for cand in (text, text.splitlines()[-1] if text else ""):
        try:
            d = json.loads(cand)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass
    return {}


def copy_transcript(session_id: str, dst_dir: str) -> str | None:
    # sandboxed sessions keep their Claude projects dir in _driver/claude_projects
    for base in (os.path.join(dst_dir, "claude_projects"), os.path.expanduser("~/.claude/projects")):
        if not os.path.isdir(base):
            continue
        for d in os.listdir(base):
            p = os.path.join(base, d, f"{session_id}.jsonl")
            if os.path.isfile(p):
                dst = os.path.join(dst_dir, "transcript.jsonl")
                shutil.copyfile(p, dst)
                return dst
    return None


def launch(prep: dict, cfg: dict) -> dict:
    drv = os.path.join(prep["run_dir"], "_driver")
    with open(os.path.join(drv, "launched"), "w") as f:
        f.write(str(time.time()))
    t0 = time.time()
    with open(os.path.join(drv, "stdout.json"), "w") as fo, open(os.path.join(drv, "stderr.log"), "w") as fe:
        p = subprocess.Popen(prep["cmd"], cwd=prep["run_dir"], env=prep["env"], stdout=fo, stderr=fe,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        timed_out = False
        try:
            rc = p.wait(timeout=60 * float(cfg["timeout_min"]))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(p.pid, signal.SIGTERM)
            try:
                rc = p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                rc = p.wait()
    with open(os.path.join(drv, "stdout.json")) as f:
        out = parse_stdout(f.read())
    tr = copy_transcript(prep["session_id"], drv) if cfg.get("copy_transcript") else None
    res = result_ok(prep["run_dir"])
    denials = out.get("permission_denials") or []
    return {"returncode": rc, "timed_out": timed_out, "wall_s": round(time.time() - t0, 1),
            "total_cost_usd": out.get("total_cost_usd"), "num_turns": out.get("num_turns"),
            "is_error": out.get("is_error"), "subtype": out.get("subtype"), "session_id": out.get("session_id") or prep["session_id"],
            "duration_ms": out.get("duration_ms"), "n_permission_denials": len(denials),
            "permission_denials": [{"tool": d.get("tool_name"), "input": str(d.get("tool_input"))[:200]} for d in denials[:10]],
            "result_text": str(out.get("result") or "")[:600], "transcript": tr,
            "status": "done" if res else ("timeout" if timed_out else "no_result"),
            "result_status": (res or {}).get("status"),
            "spec": os.path.join(prep["run_dir"], res["spec"]) if res and res.get("spec") else None,
            "iterations": (res or {}).get("iterations")}


class Manifest:
    def __init__(self, path: str, cfg: dict):
        self.path = path
        self.data = {"arm": cfg["arm"], "runs": {}}
        if os.path.isfile(path):
            with open(path) as f:
                self.data = json.load(f)
        self.data["claude"] = {k: cfg["claude"][k] for k in ("model", "effort", "max_turns", "max_budget_usd", "tools",
                                                             "permission_mode")}

    def get(self, rid: str) -> dict:
        return self.data["runs"].get(rid) or {}

    def put(self, rid: str, entry: dict):
        with _lock:
            self.data["runs"][rid] = entry
            self.data["updated"] = time.time()
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.data, f, indent=1)
            os.replace(tmp, self.path)

    def n_done(self) -> int:
        return sum(1 for e in self.data["runs"].values() if e.get("status") == "done")


class AgentScenes(Stage):
    """Datagen stage `scenes`: one headless push-scene session per prior. The arm follows the config's `prior`
    (video -> push, text -> push_text); the task's config reaches the sandboxed tools as $HDG_TASK (JSON)."""

    def run(self) -> None:
        cfg, tk = self.cfg, self.task
        cfg = dict(cfg, arm=cfg.get("arm") or {"video": "push", "text": "push_text"}[cfg["prior"]])
        task_json = json.dumps(tk.to_config())          # the tools rebuild the task from it (tasks/load.py)
        cfg["env"]["set"]["HDG_TASK"] = task_json
        if (cfg["claude"].get("isolate") or {}).get("env") is not None:
            cfg["claude"]["isolate"]["env"]["HDG_TASK"] = task_json
        arm = cfg["arm"]
        root = repo_path(cfg["arms"][arm]["out_root"])
        man = Manifest(os.path.join(root, cfg["manifest"]), cfg)
        jobs = make_jobs(cfg)
        iso = cfg["claude"].get("isolate") or {}
        if iso.get("enabled"):
            from humanoid_datagen.datagen_pipelines.agents.sandbox import build_runtime
            if shutil.which("bwrap") is None:
                raise SystemExit("claude.isolate.enabled needs bubblewrap (bwrap) on the PATH")
            print(f"[sandbox] tool runtime -> {build_runtime(iso)}")
        if not cfg["dry_run"] and shutil.which(cfg["claude_bin"]) is None:
            raise SystemExit(f"claude_bin {cfg['claude_bin']!r} not found")
        rc = cfg.get("resume") or {}
        limit_hit = threading.Event()  # a session stopped on a usage limit: launch nothing new until the reset
        last_limit = {"text": ""}

        def limited(e: dict) -> bool:
            return e.get("status") == "no_result" and \
                any(m.lower() in str(e.get("result_text") or "").lower() for m in rc.get("limit_markers") or [])

        for n_wait in range(int(rc.get("max_waits", 0)) + 1):
            if n_wait:
                s_ = _until_reset(last_limit["text"], float(rc.get("fallback_wait_min", 60)))
                print(f"[usage limit] waiting {s_ / 60:.0f} min for the reset ({n_wait}/{rc.get('max_waits')})", flush=True)
                time.sleep(s_)
                limit_hit.clear()
            _batch_pass(cfg, arm, root, man, jobs, rc, limit_hit, last_limit, limited)
            if cfg["dry_run"] or not limit_hit.is_set():
                break
        runs = man.data["runs"]
        print(f"done: {sum(e.get('status') == 'done' for e in runs.values())}/{len(runs)}; "
              f"cost ${sum((e.get('cost_usd_all_attempts') or e.get('total_cost_usd') or 0) for e in runs.values()):.2f} "
              f"(all attempts); manifest -> {man.path}")


def _until_reset(text: str, fallback_min: float) -> float:
    """Seconds until the reset time in a usage-limit message ("... resets 2:20am (America/New_York)"), + 3 min."""
    import datetime as dt
    import re
    from zoneinfo import ZoneInfo
    m = re.search(r"resets\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*\(([^)]+)\)", text or "", re.I)
    if not m:
        return fallback_min * 60
    try:
        tz = ZoneInfo(m.group(4))
    except Exception:
        return fallback_min * 60
    h = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
    now = dt.datetime.now(tz)
    t = now.replace(hour=h, minute=int(m.group(2) or 0), second=0, microsecond=0)
    if t <= now:
        t += dt.timedelta(days=1)
    return (t - now).total_seconds() + 180


def _batch_pass(cfg, arm, root, man, jobs, rc, limit_hit, last_limit, limited):
    todo, n_skip = [], 0
    for job in jobs:
        rd = os.path.join(root, job["run_id"])
        prev = man.get(job["run_id"])
        if result_ok(rd):
            n_skip += 1
            if prev.get("status") != "done":
                r = result_ok(rd)
                man.put(job["run_id"], dict(prev, run_id=job["run_id"], status="done", result_status=r.get("status")))
            continue
        attempts = int(prev.get("attempts") or 0)
        if attempts >= int(cfg["max_attempts"]) and not cfg["dry_run"]:
            print(f"[skip] {job['run_id']}: {attempts} attempts without a result")
            continue
        todo.append((job, attempts + 1))
    print(f"arm={arm}: {len(jobs)} jobs, {n_skip} done, {len(todo)} to run (max {cfg['max_concurrent']} concurrent)"
          + ("  [DRY RUN]" if cfg["dry_run"] else ""))

    def resume_sid(job) -> str | None:
        """Session to continue: the last attempt stopped on a usage limit (not a real failure) and left its run dir."""
        prev = man.get(job["run_id"])
        if not rc.get("on_usage_limit") or not limited(prev) or not prev.get("session_id"):
            return None
        return prev["session_id"] if os.path.isdir(os.path.join(root, job["run_id"])) else None

    def one(job, attempt):
        if cfg["dry_run"] and os.path.isfile(os.path.join(root, job["run_id"], "_driver", "launched")):
            # read-only for runs that already exist: never archive / rewrite a real run dir in a dry run
            sid = resume_sid(job)
            print(f"[dry-run] {job['run_id']}: would " + (f"resume session {sid}" if sid else
                  f"archive the run dir and start attempt {attempt}"))
            return {"run_id": job["run_id"]}, None
        if limit_hit.is_set() and not cfg["dry_run"]:  # queued behind a usage limit: leave it for the next pass
            return None, None
        sid = None if cfg["dry_run"] else resume_sid(job)
        prep = prepare(job, cfg, attempt, resume_sid=sid)
        if sid:
            print(f"[resume] {job['run_id']}: continuing session {sid} (previous attempt hit a usage limit)")
        base = {"run_id": job["run_id"], "arm": arm, "prior_id": job["prior_id"], "seed": job["seed"],
                "run_dir": prep["run_dir"], "scene_id": prep["scene_id"], "session_id": prep["session_id"],
                "command": prep["line"], "attempts": attempt - (1 if cfg["dry_run"] else 0)}
        if cfg["dry_run"]:
            if not man.get(job["run_id"]):  # never clobber a real run's record (session id, costs) with a dry run
                man.put(job["run_id"], dict(base, status="dry_run"))
            return base, None
        prev = man.get(job["run_id"])
        hist = list(prev.get("previous_attempts") or [])
        if prev.get("status") in ("no_result", "timeout", "driver_error", "running"):  # keep the cost of failures
            hist.append({k: prev.get(k) for k in ("attempts", "status", "subtype", "total_cost_usd", "num_turns",
                                                  "wall_s", "session_id")})
        base["previous_attempts"] = hist
        man.put(job["run_id"], dict(base, status="running", started=time.time()))
        r = launch(prep, cfg)
        entry = dict(base, **r, finished=time.time())
        if limited(entry):  # stopped by the subscription limit, not a failure: no attempt used, resumed after the reset
            entry["attempts"] = attempt - 1
            last_limit["text"] = str(entry.get("result_text") or "")
            limit_hit.set()
        # a resumed session reports its cumulative cost: don't add the attempts it continued (same session id)
        prior = [h for h in hist if not (prep.get("resumed") and h.get("session_id") == prep["session_id"])]
        entry["resumed"] = bool(prep.get("resumed"))
        entry["cost_usd_all_attempts"] = round((r["total_cost_usd"] or 0) + sum((h.get("total_cost_usd") or 0) for h in prior), 4)
        man.put(job["run_id"], entry)
        if cfg["cost"]["ledger"]:
            from humanoid_datagen.utils.cost import CostLedger
            CostLedger(repo_path(cfg["cost"]["ledger"])).log(
                cfg["arms"][arm]["source"], cfg["cost"]["stage"], gpu_hours=0.0, human_hours=0.0, n_scenes=man.n_done(),
                note=f"build-scene {arm}/{job['run_id']} attempt {attempt}: ${(r['total_cost_usd'] or 0):.4f} (claude -p "
                     f"total_cost_usd), {r['num_turns']} turns, {r['wall_s']} s, {r['status']}/{r['result_status']}, "
                     f"session {r['session_id']}")
        return base, entry

    if cfg["dry_run"]:
        for job, att in todo:
            base, _ = one(job, att)
            if "command" in base:
                print(f"\n# {base['run_id']}\n{base['command']}")
        print(f"\nmanifest -> {man.path}")
        return
    with ThreadPoolExecutor(max_workers=int(cfg["max_concurrent"])) as ex:
        futs = {ex.submit(one, j, a): j["run_id"] for j, a in todo}
        for fu in as_completed(futs):
            rid = futs[fu]
            try:
                _, e = fu.result()
                if e is None:
                    continue
                print(f"[{e['status']}] {rid}: result={e['result_status']} cost=${e['total_cost_usd']} "
                      f"turns={e['num_turns']} subtype={e['subtype']} denials={e['n_permission_denials']} {e['wall_s']}s",
                      flush=True)
            except Exception as ex_:  # keep the batch going
                print(f"[error] {rid}: {type(ex_).__name__}: {ex_}", flush=True)
                man.put(rid, dict(man.get(rid), status="driver_error", error=f"{type(ex_).__name__}: {ex_}"))

