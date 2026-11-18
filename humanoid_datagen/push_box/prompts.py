"""Push-task priors from the written prompt list: one prior folder per prompt.

    python -m humanoid_datagen.push_box.prompts --config_name push_box/prompts

The prompts themselves are written, not generated: `prompts_file` (configs/push_box/prompts_1000.jsonl) holds one
record per prior, each a hand-written Wan2.2 prompt of a person pushing a household object in a different home:
    object       object type (one of the 18 training types; the instruction "Push the <type> forward." and the
                 per-category grouping use it)
    object_text  the instance as described in the prompt
    height_m     the instance's height (picks the Kimodo phrasing and the palm-height filter of select_motions)
    camera       third | ego
    push_m       push distance named in the prompt
    room_type    short room label
    wan          the Wan2.2 prompt
The rooms are generic indoor homes written from general knowledge, never modelled on the test homes.

Each prior <out_root>/p<NNN>/ gets wan.txt, kimodo.txt (by height, below) and meta.json (the record without the
prompt). Prior ids interleave object types and cameras, so any prefix (a pilot of the first N) stays balanced.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict

# Kimodo-G1 phrasing by the object's height H, from what each phrasing gives on the G1 (palm height at first contact,
# tracked by SONIC). The palms must meet the object >= 5 cm below its top, so:
#   H < 0.45  deep crouch (palms 0.2-0.47)        0.45-0.7  half squat (0.38-0.56)
#   0.7-0.9   hands low, arms reaching down (0.66-0.83)   >= 0.9  waist / chest (0.77-1.06)
# 'Waist height' and 'bends forward' do not work on the G1: palms at 0.8-1.0 m, and the bent-over pushes fall.
# Kimodo is oversampled and filtered per prior (push_box/select_motions.py).
KIMODO_BANDS = [
    (0.45, ["a person crouches slightly and pushes a low object forward along the floor with both hands, taking small steps.",
            "a person squats down low and pushes a small box forward along the floor with both hands, shuffling forward."]),
    (0.70, ["a person walks forward in a half squat, pushing a low object along the floor with both hands at knee height.",
            "a person crouches slightly and pushes a low object forward along the floor with both hands, taking small steps."]),
    (0.90, ["a person leans forward a little and pushes a knee-high box forward with both arms reaching down, walking slowly.",
            "a person pushes a cart forward with both hands low in front of their hips, elbows straight, walking slowly.",
            "a person pushes a table forward with both hands resting on its edge just below their waist, walking slowly."]),
    (9.00, ["a person pushes an object forward with both hands at waist height while walking slowly.",
            "a person walks slowly while pushing a shopping cart forward with both hands.",
            "a person stands, places both hands on a large box at chest height and pushes it forward while taking small steps."]),
]
FIELDS = ("object", "object_text", "height_m", "camera", "push_m", "room_type", "wan")


def kimodo_prompt(height: float, prior_id: str, seed: int) -> str:
    prompts = next(ps for top, ps in KIMODO_BANDS if height < top)
    return random.Random(f"{seed}:{prior_id}:kimodo").choice(prompts)


def load_prompts(path: str) -> list[dict]:
    recs = [json.loads(l) for l in open(path) if l.strip()]
    for i, r in enumerate(recs):
        missing = [k for k in FIELDS if k not in r]
        if missing or r["camera"] not in ("third", "ego"):
            raise SystemExit(f"{path}:{i + 1}: bad record (missing {missing}, camera {r.get('camera')!r})")
    return recs


def order(recs: list[dict], seed: int) -> list[dict]:
    """Round-robin over object types (each type's prompts shuffled, cameras alternating), so prefixes stay balanced."""
    rng = random.Random(seed)
    by = defaultdict(lambda: {"third": [], "ego": []})
    for r in recs:
        by[r["object"]][r["camera"]].append(r)
    queues = {}
    for k, name in enumerate(sorted(by)):
        for cam in ("third", "ego"):
            rng.shuffle(by[name][cam])
        t, e, q = by[name]["third"], by[name]["ego"], []
        if k % 2:  # odd types start with ego, so every prefix of whole rounds is half third, half ego
            t, e = e, t
        while t or e:  # strict alternation; the longer list's leftover goes last
            for lst in (t, e):
                if lst:
                    q.append(lst.pop())
        queues[name] = q
    out = []
    while any(queues.values()):
        for name in sorted(queues):
            if queues[name]:
                out.append(queues[name].pop(0))
    return out


def write_priors(recs: list[dict], out_root: str, seed: int):
    width = max(3, len(str(len(recs) - 1)))
    os.makedirs(out_root, exist_ok=True)
    index = []
    for i, r in enumerate(recs):
        pid = f"p{i:0{width}d}"
        d = os.path.join(out_root, pid)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "wan.txt"), "w").write(r["wan"].strip() + "\n")
        open(os.path.join(d, "kimodo.txt"), "w").write(kimodo_prompt(float(r["height_m"]), pid, seed) + "\n")
        meta = {"prior_id": pid} | {k: r[k] for k in FIELDS if k != "wan"}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        index.append({k: meta[k] for k in ("prior_id", "object", "camera", "height_m")})
    with open(os.path.join(out_root, "index.jsonl"), "w") as f:
        for row in index:
            f.write(json.dumps(row) + "\n")


def main():
    from collections import Counter
    from humanoid_datagen.utils.config import parse_config
    cfg = parse_config("push_box/prompts")
    recs = order(load_prompts(cfg.prompts_file), int(cfg.seed))
    write_priors(recs, cfg.out_root, int(cfg.seed))
    print(len(recs), "priors ->", cfg.out_root)
    print("objects", dict(Counter(r["object"] for r in recs)))
    print("cameras", dict(Counter(r["camera"] for r in recs)))


if __name__ == "__main__":
    main()
