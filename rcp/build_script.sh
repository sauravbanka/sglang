#!/bin/bash
# Build the BubbleTea SGLang RCP image, push to the EPFL hub, and submit a job.
# Modeled on Ramya's build_script.sh. Run from sglang/rcp/ ON A MACHINE WITH
# DOCKER (not the jump host — it has no docker daemon).
#
#   bash build_script.sh <job-name> [--gpu N] [--no-rcp]
#   e.g.  bash build_script.sh banka-sglang-bt --gpu 2
set -e
cd "$(dirname "$0")"

JOB_NAME="${1:-banka-sglang-bt}"
GPU_COUNT=1
for ((i = 1; i <= $#; i++)); do
    a="${!i}"; n=$((i + 1))
    [[ $a == "--gpu" ]] && GPU_COUNT="${!n}"
done

IMAGE="registry.rcp.epfl.ch/sacs-sbanka/sglang-bt"   # per-user Harbor project

# ── build: FROM published lmsysorg/sglang -> DeepEP + cu13 kernels already
#    inside; NOTHING is recompiled. Identity build-args = YOUR EPFL LDAP. ──
docker build -t "$IMAGE" . \
    --build-arg LDAP_USERNAME=banka \
    --build-arg LDAP_UID=341698 \
    --build-arg LDAP_GROUPNAME=SACS-StaffU \
    --build-arg LDAP_GID=11259

docker push "$IMAGE"

load_env_flags() {
    local f="${1:-.env}"; local flags=()
    [[ -f $f ]] || { return 0; }
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ $line =~ ^[[:space:]]*# ]] && continue
        line=$(echo "$line" | xargs); [[ -z $line ]] && continue
        flags+=(-e "$line")
    done < "$f"
    printf '%s ' "${flags[@]}"
}

if [[ "$@" =~ --no-rcp ]]; then
    echo "--no-rcp: image built + pushed, no job submitted."
else
    # 2 GPUs (TP=2) exceeds the *interactive* per-user cap, so this is a plain
    # train-type submit (no --interactive), exactly like Ramya's script.
    runai submit --name "$JOB_NAME" \
        -i "$IMAGE" \
        --gpu "$GPU_COUNT" --cpu 32 --memory 128G --large-shm \
        --pvc sacs-scratch:/mnt/nfs \
        --backoff-limit 0 \
        $(load_env_flags .env)
    echo
    echo "submitted $JOB_NAME (${GPU_COUNT} GPU)."
    echo "  status : runai list jobs"
    echo "  shell  : runai bash $JOB_NAME"
    echo "  logs   : runai logs $JOB_NAME -f"
fi
