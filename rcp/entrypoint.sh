#!/bin/bash
# RCP entrypoint: start SSH, overlay the BubbleTea fork's Python from NFS onto
# the image's installed sglang (NO recompile — kernels/DeepEP/rust-ext stay as
# built in the image; only .py files are refreshed), then idle.
set -e

# 1. SSH (key from the runtime env var set by submit-job.sh).
if [ -n "${SSH_PUBLIC_KEY:-}" ] && [ -n "${USER_NAME:-}" ]; then
    mkdir -p "/home/${USER_NAME}/.ssh"
    echo "${SSH_PUBLIC_KEY}" > "/home/${USER_NAME}/.ssh/authorized_keys"
    chown -R "${USER_NAME}:${USER_NAME}" "/home/${USER_NAME}/.ssh"
    chmod 700 "/home/${USER_NAME}/.ssh"; chmod 600 "/home/${USER_NAME}/.ssh/authorized_keys"
fi
/usr/sbin/sshd || service ssh start || true

# 2. Overlay fork Python from NFS. FORK_DIR points at the fork on the NFS PVC,
#    e.g. /mnt/nfs/home/<user>/EPFL/vllm/sglang. rsync WITHOUT --delete merges,
#    so our .py edits + bubble_profile.py land while the installed .so/DeepEP
#    stay in place. Re-runs each start, so live NFS edits take effect on restart.
if [ -n "${FORK_DIR:-}" ] && [ -d "${FORK_DIR}/python/sglang" ]; then
    SITE="$(python -c 'import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))')"
    # Overlay the WHOLE sglang package (not just srt/): the fork can be a
    # different version than the base image, and a partial srt-only overlay
    # mixes new srt against old jit_kernel/etc and fails to import. rsync
    # WITHOUT --delete keeps the image's compiled .so files in place.
    echo "Overlaying ${FORK_DIR}/python/sglang -> ${SITE}/sglang (whole package)"
    rsync -a --exclude '__pycache__' "${FORK_DIR}/python/sglang/" "${SITE}/sglang/"
else
    echo "FORK_DIR not set or not found — using the image's stock sglang."
fi

echo "Ready. SSH: port 2500. Run the profiler per sglang/bubbletea/README.md."
exec sleep infinity
