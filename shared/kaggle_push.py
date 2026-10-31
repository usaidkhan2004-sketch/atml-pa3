"""Push a repo script to Kaggle as a private script kernel.

Kaggle script kernels run a single file without command-line arguments, so the script's
`KAGGLE_ARGS = []` line is filled in with the given arguments in a temporary copy.

  python shared/kaggle_push.py task1/train.py atml-pa3-task1-benchmark --gpu \
      --source dentusayd/atml-pa3-task1-prepare-data -- --benchmark

`--with FILE` (repeatable) bundles other repo files: a prelude writes them, and an unmodified
copy of the script itself, under /tmp/pa3_src/<repo path> and sets PA3_BUNDLE, so the script
can import repo modules and start worker processes from its own file.
"""

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

USER = "dentusayd"
REPO = Path(__file__).resolve().parents[1]
BUNDLE_DIR = "/tmp/pa3_src"

PRELUDE = '''# --- added by shared/kaggle_push.py: repo files bundled with this kernel
import base64 as _b64, os as _os, sys as _sys
from pathlib import Path as _Path
for _rel, _data in {files}.items():
    _p = _Path("{bundle}") / _rel
    _p.parent.mkdir(parents=True, exist_ok=True)
    _p.write_bytes(_b64.b64decode(_data))
    if _p.suffix == ".py" and str(_p.parent) not in _sys.path:
        _sys.path.insert(0, str(_p.parent))
_os.environ["PA3_BUNDLE"] = "{bundle}"
# --- end of bundle prelude
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("script", type=Path)
    p.add_argument("slug")
    p.add_argument("--gpu", action="store_true", help="T4 accelerator")
    p.add_argument("--source", action="append", default=[], help="kernel output to mount as input")
    p.add_argument("--dataset", action="append", default=[], help="Kaggle dataset to mount as input")
    p.add_argument("--with", dest="bundle", action="append", default=[], type=Path,
                   help="repo file to bundle with the kernel")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    a = p.parse_args(argv[:split])
    args = argv[split + 1:]

    code = a.script.read_text()
    marker = "KAGGLE_ARGS = []"
    if args:
        assert code.count(marker) == 1, f"{a.script} has no `{marker}` line"
        code = code.replace(marker, f"KAGGLE_ARGS = {json.dumps(args)}")
    if a.bundle:
        files = {str(f.resolve().relative_to(REPO)): base64.b64encode(f.read_bytes()).decode()
                 for f in [a.script, *a.bundle]}
        code = PRELUDE.format(files=json.dumps(files), bundle=BUNDLE_DIR) + code

    with tempfile.TemporaryDirectory() as d:
        (Path(d) / a.script.name).write_text(code)
        meta = {
            "id": f"{USER}/{a.slug}", "title": a.slug, "code_file": a.script.name,
            "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": "true" if a.gpu else "false", "enable_tpu": "false",
            "enable_internet": "true", "dataset_sources": a.dataset, "competition_sources": [],
            "kernel_sources": a.source, "model_sources": [],
        }
        (Path(d) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        cmd = ["kaggle", "kernels", "push", "-p", d]
        if a.gpu:
            cmd += ["--accelerator", "NvidiaTeslaT4"]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
