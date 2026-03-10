#!/usr/bin/env bash
set -euo pipefail

cat <<'INTRO'
================================================================================
脚本 0：下载审稿测试所需依赖（源码与插件）
================================================================================
用途：
1) 下载 LAMMPS 两个版本源码：
   - 10Dec2025
   - 23Jun2022
2) 下载 Intel MPI 安装包（oneAPI HPC Toolkit 或 MPI 独立安装包）
3) 下载两个 LAMMPS 外部插件源码：
   - tabGAP 插件
   - NEP_CPU 插件

输出目录（位于当前仓库）：
- opt/downloads      : 各类压缩包/安装包
- opt/src            : 解压后的源码与插件仓库

可选环境变量（覆盖默认行为）：
- LAMMPS_2025_URL    : 指定 10Dec2025 tar.gz 下载地址
- LAMMPS_2022_URL    : 指定 23Jun2022 tar.gz 下载地址
- INTEL_MPI_URL      : 指定 Intel MPI/oneAPI 安装包地址
- TABGAP_PLUGIN_REPO : 指定 tabGAP 插件 Git 地址（若不设置，默认使用 LAMMPS 插件集合）
- NEP_PLUGIN_REPO    : 指定 NEP_CPU 插件 Git 地址

说明：
- 若默认地址不可用，脚本会尝试候选地址。
- 若仍失败，请通过上述环境变量显式传入可用地址后重试。
================================================================================
INTRO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOWNLOAD_DIR="${REPO_ROOT}/opt/downloads"
SRC_DIR="${REPO_ROOT}/opt/src"

mkdir -p "${DOWNLOAD_DIR}" "${SRC_DIR}"

download_first_available() {
  local output_file="$1"
  shift
  local urls=("$@")
  local ok=0
  for u in "${urls[@]}"; do
    if [[ -z "${u}" ]]; then
      continue
    fi
    echo "[下载] ${u}"
    if curl -fL --retry 3 --connect-timeout 20 -o "${output_file}" "${u}"; then
      ok=1
      break
    fi
  done
  if [[ "${ok}" -ne 1 ]]; then
    echo "[错误] 所有候选地址都不可用：${output_file}" >&2
    return 1
  fi
}

extract_lammps_tarball() {
  local tarball="$1"
  local expected_dir="$2"
  if [[ -d "${expected_dir}" ]]; then
    echo "[跳过] 已存在：${expected_dir}"
    return 0
  fi
  tar -xf "${tarball}" -C "${SRC_DIR}"
}

clone_or_update_repo() {
  local dest="$1"
  shift
  local candidates=("$@")
  if [[ -d "${dest}/.git" ]]; then
    echo "[更新] ${dest}"
    git -C "${dest}" fetch --all --tags --prune
    git -C "${dest}" pull --ff-only || true
    return 0
  fi
  local ok=0
  for r in "${candidates[@]}"; do
    if [[ -z "${r}" ]]; then
      continue
    fi
    echo "[克隆] ${r} -> ${dest}"
    if git clone --depth 1 "${r}" "${dest}"; then
      ok=1
      break
    fi
  done
  if [[ "${ok}" -ne 1 ]]; then
    echo "[错误] 无法克隆仓库：${dest}" >&2
    return 1
  fi
}

LMP_2025_TAR="${DOWNLOAD_DIR}/lammps-10Dec2025.tar.gz"
LMP_2022_TAR="${DOWNLOAD_DIR}/lammps-23Jun2022.tar.gz"
INTEL_MPI_INSTALLER="${DOWNLOAD_DIR}/intel_mpi_or_hpc_toolkit.sh"

LMP_2025_URLS=(
  "${LAMMPS_2025_URL:-}"
  "https://github.com/lammps/lammps/releases/download/stable_10Dec2025/lammps-src-10Dec2025.tar.gz"
  "https://download.lammps.org/tars/lammps-10Dec2025.tar.gz"
  "https://download.lammps.org/tars/stable/lammps-10Dec2025.tar.gz"
)

LMP_2022_URLS=(
  "${LAMMPS_2022_URL:-}"
  "https://github.com/lammps/lammps/archive/refs/tags/stable_23Jun2022.tar.gz"
  "https://download.lammps.org/tars/lammps-23Jun2022.tar.gz"
  "https://download.lammps.org/tars/stable/lammps-23Jun2022.tar.gz"
)

INTEL_MPI_URLS=(
  "${INTEL_MPI_URL:-}"
  "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/19144/l_HPCKit_p_2024.1.0.596_offline.sh"
  "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/1/l_mpi_oneapi_p_2021.12.1.8_offline.sh"
)

echo "[阶段] 下载 LAMMPS 10Dec2025"
download_first_available "${LMP_2025_TAR}" "${LMP_2025_URLS[@]}"

echo "[阶段] 下载 LAMMPS 23Jun2022"
download_first_available "${LMP_2022_TAR}" "${LMP_2022_URLS[@]}"

echo "[阶段] 下载 Intel MPI / oneAPI 安装包"
download_first_available "${INTEL_MPI_INSTALLER}" "${INTEL_MPI_URLS[@]}"
chmod +x "${INTEL_MPI_INSTALLER}"

echo "[阶段] 解压 LAMMPS 源码"
extract_lammps_tarball "${LMP_2025_TAR}" "${SRC_DIR}/lammps-10Dec2025"
extract_lammps_tarball "${LMP_2022_TAR}" "${SRC_DIR}/lammps-23Jun2022"

TABGAP_PLUGIN_CANDIDATES=(
  "${TABGAP_PLUGIN_REPO:-}"
  "https://gitlab.com/jezper/tabgap.git"
)

NEP_PLUGIN_CANDIDATES=(
  "${NEP_PLUGIN_REPO:-}"
  "https://github.com/brucefan1983/NEP_CPU.git"
)

echo "[阶段] 下载 tabGAP 插件源码"
clone_or_update_repo "${SRC_DIR}/tabgap-plugin" "${TABGAP_PLUGIN_CANDIDATES[@]}"

echo "[阶段] 下载 NEP_CPU 插件源码"
clone_or_update_repo "${SRC_DIR}/nep-cpu-plugin" "${NEP_PLUGIN_CANDIDATES[@]}"

echo "================================================================================"
echo "下载完成"
echo "LAMMPS 2025 源码: ${SRC_DIR}/lammps-10Dec2025"
echo "LAMMPS 2022 源码: ${SRC_DIR}/lammps-23Jun2022"
echo "Intel MPI 安装包: ${INTEL_MPI_INSTALLER}"
echo "tabGAP 插件目录 : ${SRC_DIR}/tabgap-plugin"
echo "NEP_CPU 插件目录: ${SRC_DIR}/nep-cpu-plugin"
echo "================================================================================"
