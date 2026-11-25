"""Repository-relative paths: config paths are relative to the repo root unless absolute; `~` and $VARS are expanded.

Third-party repos (kimodo, ml-halo, Wan2.2, infinigen) live under $THIRD_PARTY_ROOT, default <repo>/utilities/repos;
configs refer to them as $THIRD_PARTY_ROOT/<repo>/..., code as third_party("<repo>", ...).
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("THIRD_PARTY_ROOT", os.path.join(REPO_ROOT, "utilities", "repos"))


def repo_path(p: str | None) -> str | None:
    if p is None:
        return None
    p = os.path.expandvars(os.path.expanduser(str(p)))
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(REPO_ROOT, p))


def third_party(*parts: str) -> str:
    """A path inside a third-party repo: $THIRD_PARTY_ROOT/<parts>."""
    return os.path.join(os.environ["THIRD_PARTY_ROOT"], *parts)
