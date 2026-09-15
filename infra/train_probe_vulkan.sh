#!/usr/bin/env bash
# =============================================================================
# 集群探针：彻底查清「训练任务容器」有没有可用 NVIDIA Vulkan / modeset
# （探索机 notebook 的结论不能代替集群；请用本脚本交 ModelArts 训练任务）
#
# 检查项：
#   1) hostname / 卡数 / 平台注入的 NVIDIA_DRIVER_CAPABILITIES
#   2) ls /dev/nvidia* + 主次设备号
#   3) cgroup devices.list 是否含 c 195:254（modeset）
#   4) open(/dev/nvidia-modeset, O_RDWR)
#   5) mknod 补节点是否可行（通常 EACCES）
#   6) 强制 export graphics 后设备是否出现（通常不会）
#   7) nvidia-smi
#   8) 若 modeset 可用：试 NVIDIA ICD + SapienRenderer
#      若不可用：明确结论 = 无 NVIDIA Vulkan，只能 lavapipe
#
# 提交：算法包 cwr_wulan_algorithm，入口 train_probe_vulkan.sh
#       1 机 × 1~2 卡（或 8 卡也行）；超时 30min；挂载 wulann + wulann4
# 结果（共享盘，探索机可直接看）：
#   .../cwr_dataset_wulann4/aha-wam-runs/robotwin/vulkan_probe_<ts>/
#     forensics.json  forensics.txt  CONCLUSION.md  nvidia-smi.txt
# =============================================================================
set -euo pipefail

echo "train_probe_vulkan.sh revision: 2026-09-09-probe-v3-cluster"
echo "==== host / image ===="
date -Is || true
hostname || true
uname -a || true
echo "NVIDIA_DRIVER_CAPABILITIES(raw)=${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
echo "MA_NUM_GPUS=${MA_NUM_GPUS:-<unset>} VC_TASK_INDEX=${VC_TASK_INDEX:-<unset>}"
echo "VC_WORKER_HOSTS=${VC_WORKER_HOSTS:-<unset>}"

if [[ -d "/opt/huawei/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp" ]]; then
    export DATA="/opt/huawei/dataset"
    echo "ENV=cluster (/opt/huawei/dataset)"
else
    export DATA="/home/ma-user/work/dataset"
    echo "ENV=explore (/home/ma-user/work/dataset) — 仅供本地试跑；请再交集群任务"
fi

TS="$(date +%Y%m%d_%H%M%S)"
HOST_SAFE="$(hostname 2>/dev/null | tr -c 'A-Za-z0-9._-' '_' || echo host)"
PROBE_OUT="${DATA}/cwr_dataset_wulann4/aha-wam-runs/robotwin/vulkan_probe_${TS}_${HOST_SAFE}"
mkdir -p "${PROBE_OUT}"
export PROBE_OUT_DIR="${PROBE_OUT}"
echo "PROBE_OUT=${PROBE_OUT}"

# latest pointer (overwrite)
LATEST="${DATA}/cwr_dataset_wulann4/aha-wam-runs/robotwin/vulkan_probe_LATEST"
rm -f "${LATEST}" 2>/dev/null || true
ln -sfn "${PROBE_OUT}" "${LATEST}" 2>/dev/null || echo "${PROBE_OUT}" > "${LATEST}.path"
echo "LATEST -> ${PROBE_OUT}"

PY310="${DATA}/cwr_dataset_wulann4/envs/ahawam/bin/python"
if [[ ! -x "${PY310}" ]]; then
    echo "ERROR: missing ${PY310}"
    echo "PROBE_RESULT modeset=0 reason=no_py310" | tee "${PROBE_OUT}/CONCLUSION.md"
    exit 2
fi
export PYTHON="${PY310}"
echo "Using python: ${PYTHON} ($("${PYTHON}" -V))"

# nvidia-smi dump
echo "==== nvidia-smi ===="
(nvidia-smi 2>&1 || true) | tee "${PROBE_OUT}/nvidia-smi.txt"
(nvidia-smi -L 2>&1 || true) | tee -a "${PROBE_OUT}/nvidia-smi.txt"

# ---------- forensics ----------
"${PYTHON}" - <<'PY' | tee "${PROBE_OUT_DIR}/forensics.txt"
import json, os, errno, stat, glob, socket, subprocess
from datetime import datetime
from pathlib import Path

out = Path(os.environ["PROBE_OUT_DIR"])
data = Path(os.environ.get("DATA", "/opt/huawei/dataset"))
report = {
    "ts": datetime.now().isoformat(timespec="seconds"),
    "hostname": socket.gethostname(),
    "env_hint": "cluster" if str(data).startswith("/opt/huawei") else "explore",
    "data_root": str(data),
    "nvidia_driver_capabilities_raw": os.environ.get("NVIDIA_DRIVER_CAPABILITIES"),
    "ma_num_gpus": os.environ.get("MA_NUM_GPUS"),
    "vc_task_index": os.environ.get("VC_TASK_INDEX"),
    "devices": {},
    "cgroup_195": [],
    "cgroup_has_195_254_modeset": False,
    "modeset": {},
    "mknod_attempt": {},
    "nvidia_icd_files": [],
    "conclusion": {},
}

devs = sorted(glob.glob("/dev/nvidia*"))
report["nvidia_dev_paths"] = devs
print("==== /dev/nvidia* ====")
for p in devs:
    try:
        st = os.stat(p)
        info = {
            "exists": True,
            "mode": oct(st.st_mode),
            "maj": os.major(st.st_rdev) if stat.S_ISCHR(st.st_mode) else None,
            "min": os.minor(st.st_rdev) if stat.S_ISCHR(st.st_mode) else None,
        }
    except OSError as e:
        info = {"exists": True, "stat_error": str(e)}
    report["devices"][p] = info
    print(p, info)

cg = Path("/sys/fs/cgroup/devices/devices.list")
print("==== cgroup devices.list (195:*) ====")
if cg.is_file():
    lines = cg.read_text().splitlines()
    for line in lines:
        if "195:" in line:
            report["cgroup_195"].append(line.strip())
            print(line.strip())
    report["cgroup_has_195_254_modeset"] = any(
        l.strip().startswith("c 195:254") for l in lines
    )
    print("has c 195:254 (modeset)?", report["cgroup_has_195_254_modeset"])
    print("legend: GPU=195:0..N  nvidiactl=195:255  modeset=195:254")
else:
    print("devices.list missing:", cg)
    report["cgroup_note"] = f"missing {cg}"

p = "/dev/nvidia-modeset"
ms = {"path": p, "exists_before": os.path.exists(p), "open_rdwr": None, "open_error": None}
print("==== modeset open ====")
print("exists:", ms["exists_before"])
if ms["exists_before"]:
    try:
        st = os.stat(p)
        ms["maj"] = os.major(st.st_rdev)
        ms["min"] = os.minor(st.st_rdev)
        fd = os.open(p, os.O_RDWR)
        os.close(fd)
        ms["open_rdwr"] = True
        print("RDWR: OK", f"maj={ms['maj']} min={ms['min']}")
    except OSError as e:
        ms["open_rdwr"] = False
        ms["open_error"] = f"{e.errno} {errno.errorcode.get(e.errno, e.errno)} {e.strerror}"
        print("RDWR: FAIL", ms["open_error"])
else:
    print("MISSING node")

mk = {"tried": False, "ok": False, "error": None, "open_after": None}
if not ms.get("open_rdwr"):
    if not os.path.exists(p):
        mk["tried"] = True
        print("==== mknod c 195 254 ====")
        try:
            os.mknod(p, mode=stat.S_IFCHR | 0o666, device=os.makedev(195, 254))
            mk["ok"] = True
            print("mknod: OK")
        except OSError as e:
            mk["error"] = f"{e.errno} {errno.errorcode.get(e.errno, e.errno)} {e.strerror}"
            print("mknod: FAIL", mk["error"])
    if os.path.exists(p):
        try:
            fd = os.open(p, os.O_RDWR)
            os.close(fd)
            mk["open_after"] = True
            print("RDWR after mknod/existing: OK")
        except OSError as e:
            mk["open_after"] = False
            mk["open_after_error"] = (
                f"{e.errno} {errno.errorcode.get(e.errno, e.errno)} {e.strerror}"
            )
            print("RDWR after mknod/existing: FAIL", mk["open_after_error"])

report["modeset"] = ms
report["mknod_attempt"] = mk

# ICD candidates on shared disk
icd_candidates = [
    data / "cwr_dataset_wulann/nvidia-driver-libs/nvidia-535.183.01/nvidia_icd_abs.json",
    data / "cwr_dataset_wulann/nvidia-driver-libs/nvidia-535.183.01/nvidia_icd.json",
]
for icd in icd_candidates:
    report["nvidia_icd_files"].append({"path": str(icd), "exists": icd.is_file()})
print("==== nvidia ICD on shared disk ====")
for x in report["nvidia_icd_files"]:
    print(x)

# force graphics env
os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility,graphics,display,video"
report["nvidia_driver_capabilities_forced"] = os.environ["NVIDIA_DRIVER_CAPABILITIES"]
report["modeset_exists_after_force_env"] = os.path.exists(p)
print("==== after FORCE graphics env ====")
print("CAP=", report["nvidia_driver_capabilities_forced"])
print("modeset exists still?", report["modeset_exists_after_force_env"])

usable = bool(ms.get("open_rdwr")) or bool(mk.get("open_after"))
if usable:
    reason = "RDWR ok on /dev/nvidia-modeset — NVIDIA Vulkan headless path is possible"
    recommend = "评测入口可走 AHAWAM_VULKAN_MODE=nvidia（与 MAIN-2 同款）"
    plan = "nvidia"
else:
    reason = (
        "no usable /dev/nvidia-modeset "
        f"(exists_before={ms.get('exists_before')}, "
        f"cgroup_195_254={report['cgroup_has_195_254_modeset']}, "
        f"mknod={mk.get('error') or mk.get('open_after_error') or 'n/a'}). "
        "export NVIDIA_DRIVER_CAPABILITIES cannot inject the device at runtime."
    )
    recommend = (
        "当前集群任务没有可用 NVIDIA Vulkan 渲染；评测只能 lavapipe（慢）。"
        "请平台在【创建容器时】注入 graphics，使 cgroup 含 c 195:254 并挂上 /dev/nvidia-modeset。"
    )
    plan = "lavapipe"

report["conclusion"] = {
    "modeset_usable": usable,
    "nvidia_vulkan_available": usable,
    "plan": plan,
    "reason": reason,
    "recommend": recommend,
}
print("==== CONCLUSION ====")
print(json.dumps(report["conclusion"], ensure_ascii=False, indent=2))
(out / "forensics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print("WROTE", out / "forensics.json")
print(
    f"PROBE_RESULT modeset={1 if usable else 0} "
    f"nvidia_vulkan={1 if usable else 0} "
    f"cgroup_195_254={1 if report['cgroup_has_195_254_modeset'] else 0} "
    f"dev_exists={1 if ms['exists_before'] else 0} "
    f"env={report['env_hint']} "
    f"plan={plan}"
)

# human CONCLUSION.md
md = []
md.append(f"# Vulkan / modeset probe — {report['hostname']}")
md.append("")
md.append(f"- time: `{report['ts']}`")
md.append(f"- env: **{report['env_hint']}** (`{report['data_root']}`)")
md.append(f"- CAP raw: `{report['nvidia_driver_capabilities_raw']}`")
md.append(f"- CAP forced: `{report['nvidia_driver_capabilities_forced']}`")
md.append(f"- `/dev/nvidia*`: `{', '.join(devs) if devs else 'NONE'}`")
md.append(f"- cgroup 195:*: `{', '.join(report['cgroup_195']) or 'none'}`")
md.append(f"- cgroup has **195:254** (modeset): **{report['cgroup_has_195_254_modeset']}**")
md.append(f"- modeset node exists: **{ms['exists_before']}**")
md.append(f"- modeset RDWR: **{ms.get('open_rdwr')}** {ms.get('open_error') or ''}")
md.append(f"- mknod: **{mk}**")
md.append("")
md.append(f"## Verdict")
md.append("")
md.append(f"- **nvidia_vulkan_available = {usable}**")
md.append(f"- plan: `{plan}`")
md.append(f"- reason: {reason}")
md.append(f"- recommend: {recommend}")
md.append("")
(out / "CONCLUSION.md").write_text("\n".join(md) + "\n")
print("WROTE", out / "CONCLUSION.md")
PY

_MODESET_OK=0
if grep -q 'PROBE_RESULT modeset=1 ' "${PROBE_OUT_DIR}/forensics.txt" 2>/dev/null; then
    _MODESET_OK=1
fi
echo "_MODESET_OK=${_MODESET_OK}"

export NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics,display,video"
NVIDIA_DRIVER_DIR="${DATA}/cwr_dataset_wulann/nvidia-driver-libs/nvidia-535.183.01"
SAPIEN_LIBS="${DATA}/cwr_dataset_wulann/sapien-runtime-libs"

# Optional: if modeset OK and sapien present, smoke NVIDIA renderer
echo "==== optional SapienRenderer smoke ===="
set +e
if [[ "${_MODESET_OK}" -eq 1 ]]; then
    export AHAWAM_VULKAN_MODE=nvidia
    if [[ -f "${NVIDIA_DRIVER_DIR}/nvidia_icd_abs.json" ]]; then
        export VK_ICD_FILENAMES="${NVIDIA_DRIVER_DIR}/nvidia_icd_abs.json"
    elif [[ -f "${NVIDIA_DRIVER_DIR}/nvidia_icd.json" ]]; then
        export VK_ICD_FILENAMES="${NVIDIA_DRIVER_DIR}/nvidia_icd.json"
    fi
    export LD_LIBRARY_PATH="${NVIDIA_DRIVER_DIR}:${SAPIEN_LIBS}:${LD_LIBRARY_PATH:-}"
    echo "try NVIDIA ICD VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-<empty>}"
else
    export AHAWAM_VULKAN_MODE=lavapipe
    echo "skip NVIDIA ICD smoke (no modeset); would only work with lavapipe"
fi

if "${PYTHON}" -c "import sapien" >/dev/null 2>&1; then
    "${PYTHON}" - <<'PY' | tee -a "${PROBE_OUT_DIR}/forensics.txt"
import os, traceback
print("AHAWAM_VULKAN_MODE=", os.environ.get("AHAWAM_VULKAN_MODE"))
print("VK_ICD_FILENAMES=", os.environ.get("VK_ICD_FILENAMES"))
try:
    import sapien, sapien.render
    print("sapien", getattr(sapien, "__version__", "?"))
    r = sapien.render.SapienRenderer()
    print("SapienRenderer: ok")
    engine = sapien.Engine()
    engine.set_renderer(r)
    scene = engine.create_scene(sapien.SceneConfig())
    print("create_scene: ok")
    print("PROBE_SAPIEN=OK")
except Exception as e:
    print("PROBE_SAPIEN=FAIL", e)
    traceback.print_exc()
PY
else
    echo "sapien not in env — skip renderer smoke (modeset forensics still valid)"
    echo "PROBE_SAPIEN=SKIP" | tee -a "${PROBE_OUT_DIR}/forensics.txt"
fi
set -e

echo
if [[ "${_MODESET_OK}" -eq 1 ]]; then
    echo "FINAL: cluster HAS usable modeset → NVIDIA Vulkan possible"
else
    echo "FINAL: cluster has NO usable modeset → NO NVIDIA Vulkan for Sapien (lavapipe only)"
fi
echo "Read on explore machine:"
echo "  ${PROBE_OUT}/CONCLUSION.md"
echo "  ${LATEST}/CONCLUSION.md  (symlink/latest)"
# always exit 0 so platform marks job success and you can read artifacts
exit 0
