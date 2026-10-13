# Kernel Kaggle sinh tự động bởi kaggle_jobs/build.py — chạy Lab Day 10 trên GPU T4 x2.
# Mã nguồn repo (scripts/, lab/, configs/...) được nhúng dạng tar.gz base64 ở PAYLOAD.
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

STAGE = "__STAGE__"
LANES = json.loads(r'''__LANES__''')
PREV_KERNELS = json.loads(r'''__PREV__''')
PAYLOAD = "__PAYLOAD__"

OUT = Path("/kaggle/working")
REPO = Path("/tmp/repo")
LOGS = OUT / "logs"
T0 = time.time()


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def stamp():
    return f"[{(time.time() - T0) / 60:6.1f}m]"


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    if REPO.exists():
        shutil.rmtree(REPO)
    REPO.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tf:
        tf.extractall(REPO)
    os.chdir(REPO)
    (OUT / "work").mkdir(exist_ok=True)
    os.symlink(OUT / "work", REPO / "work")  # mọi thứ trong work/ được Kaggle lưu làm output

    sh("nvidia-smi -L")
    sh("pip install -q transformers==5.19.0 peft==0.21.2 trl==1.9.2 accelerate==1.15.0 "
       "bitsandbytes==0.50.2 2>&1 | grep -v -i 'warning\\|notice' | tail -5")
    sh("python -c \"import torch,transformers,peft,trl,bitsandbytes as b;"
       "print('versions', torch.__version__, transformers.__version__, peft.__version__, trl.__version__, b.__version__)\"")

    src = None
    for p in sorted(Path("/kaggle/input").rglob("labels-YOLO")):
        if (p.parent / "images").is_dir():
            src = p.parent
            break
    if src is None:
        sys.exit("không tìm thấy dataset trong /kaggle/input")
    sh(f"python scripts/prepare_data.py --source {src} --work /tmp/prep")
    shutil.copy(REPO / "data" / "splits" / "meta.json", OUT / "work" / "meta.json")

    for slug in PREV_KERNELS:  # output của stage trước (adapter, projector, run.json, dự đoán)
        name = slug.split("/")[-1]
        cands = [Path("/kaggle/input") / name, *Path("/kaggle/input").rglob(name)]
        base = next((c for c in cands if (c / "work").is_dir()), None)
        if base is None:
            print(f"cảnh báo: không thấy output của {slug}", flush=True)
            continue
        sh(f"cp -rn {base / 'work'}/. {OUT / 'work'}/")
    sh("python -m pytest -q 2>&1 | tail -2")

    from huggingface_hub import snapshot_download
    t = time.time()
    snapshot_download("Qwen/Qwen3-VL-4B-Instruct")
    print(f"{stamp()} tải model xong sau {(time.time() - t) / 60:.1f} phút", flush=True)

    status = {}

    def run_lane(i, cmds):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTHONUNBUFFERED="1",
                   TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        with open(LOGS / f"{STAGE}_lane{i}.log", "a", encoding="utf-8") as log:
            for c in cmds:
                log.write(f"\n{stamp()} $ {c}\n")
                log.flush()
                t = time.time()
                rc = subprocess.run(c, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                msg = f"{stamp()} lane{i} rc={rc} ({(time.time() - t) / 60:.1f}m): {c}"
                log.write(msg + "\n")
                print(msg, flush=True)
                status.setdefault(i, []).append({"cmd": c, "rc": rc, "minutes": (time.time() - t) / 60})
                if rc != 0:
                    break

    threads = [threading.Thread(target=run_lane, args=(i, cmds)) for i, cmds in enumerate(LANES)]
    for th in threads:
        th.start()
    while any(th.is_alive() for th in threads):
        time.sleep(30)
    (OUT / "work" / f"status_{STAGE}.json").write_text(json.dumps(status, indent=2))
    for f in sorted(LOGS.glob(f"{STAGE}_lane*.log")):
        print(f"\n===== {f.name} (cuối) =====")
        print("\n".join(f.read_text(encoding="utf-8", errors="replace").splitlines()[-150:]))
    print(f"{stamp()} xong stage {STAGE}", flush=True)


if __name__ == "__main__":
    main()
