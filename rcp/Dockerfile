# syntax=docker/dockerfile:1.7
# Thin RCP wrapper over the PUBLISHED SGLang image.
#
# The published image already contains sglang + sgl-kernel + flashinfer + DeepEP
# (sglang/docker/Dockerfile builds DeepEP from source). We add NOTHING that
# recompiles CUDA/kernels — only SSH, tools, and an RCP user matching your EPFL
# LDAP UID/GID. Your fork's pure-Python edits are overlaid at runtime from NFS by
# entrypoint.sh, so code changes never need an image rebuild.
#
# NB: this differs from Ramya's vLLM Dockerfile (FROM nvidia/cuda:12.8) on
# purpose — basing on the published SGLang image gives DeepEP + cu13 kernels for
# free instead of rebuilding them from a bare CUDA base.
ARG SGLANG_TAG=latest
FROM lmsysorg/sglang:${SGLANG_TAG}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server sudo vim git tmux htop curl rsync ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── RCP CaaS requirement: a user matching your EPFL LDAP UID/GID, so files
#    written to the mounted NFS (sacs-scratch) are owned by you. Passed as
#    build-args by build_script.sh (same pattern as Ramya's Dockerfile). ──
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID
RUN (getent group "${LDAP_GID}" || groupadd -g "${LDAP_GID}" "${LDAP_GROUPNAME}") && \
    useradd -rm -d "/home/${LDAP_USERNAME}" -s /bin/bash -g "${LDAP_GID}" -u "${LDAP_UID}" "${LDAP_USERNAME}" && \
    usermod -aG sudo "${LDAP_USERNAME}" && \
    echo "${LDAP_USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    mkdir -p /mnt/nfs && chown "${LDAP_UID}:${LDAP_GID}" /mnt/nfs
ENV USER_NAME=${LDAP_USERNAME}

# SSH on port 2500 (RCP convention), key-based only — needed for VS Code Remote.
RUN mkdir -p /var/run/sshd && ssh-keygen -A && \
    sed -i 's/#\?Port .*/Port 2500/'                                 /etc/ssh/sshd_config && \
    sed -i 's/#\?PubkeyAuthentication .*/PubkeyAuthentication yes/'   /etc/ssh/sshd_config && \
    sed -i 's/#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
EXPOSE 2500

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Hand the editable sglang tree to banka so the entrypoint (running as banka)
# can overlay the fork's Python onto it, and run the pod as banka so NFS writes
# aren't root-squashed. sshd (root-only, VS Code Remote) is best-effort and
# non-fatal in entrypoint.sh; use `runai bash` for a shell.
RUN chown -R ${LDAP_UID}:${LDAP_GID} /sgl-workspace/sglang 2>/dev/null || true
USER ${LDAP_USERNAME}
ENTRYPOINT ["/entrypoint.sh"]
