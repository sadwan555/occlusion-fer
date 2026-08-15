# Server Quick Start

The formal project spans three immutable Git commits. Do not run the 112/v1
configs visible on `main` or `private-final-eval-v2` as current paper training.

```bash
export FER_ROOT="${HOME}/anson-fer"
export FER_REPO="${FER_ROOT}/repo/occlusion-fer"
export FER_DATA="${FER_ROOT}/data/fer2013.csv"
export FER_RESULTS="${FER_ROOT}/results"

git clone git@github.com:sadwan555/occlusion-fer.git "${FER_REPO}"
cd "${FER_REPO}"
git worktree add "${FER_ROOT}/worktrees/e7-clean" 4cb1e0ffe4b55efc090a45cfed560b28f50b9509
git worktree add "${FER_ROOT}/worktrees/stage8" c1c9187aa2ddf7dd84906c7f139ad9a750ef202d
git worktree add "${FER_ROOT}/worktrees/private-final" 7e154aca1e95ef78ea7e3bc8767bcb21ca769335
```

Install and test the selected worktree:

```bash
python3 -m venv "${FER_ROOT}/envs/occlusion-fer"
source "${FER_ROOT}/envs/occlusion-fer/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[test,paper]"
PYTHONPATH=src pytest -q
python3 -m compileall -q src tests
```

The completed formal results should normally be verified from their archived
manifests and SHA-256 values, not retrained. For the exact stage ordering,
identity checks, and failure handling, see
[`docs/server_runbook.md`](docs/server_runbook.md).
