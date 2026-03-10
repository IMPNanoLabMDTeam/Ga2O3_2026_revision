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
- TABGAP_PLUGIN_REPO : 指定 tabGAP 插件 Git 地址（默认 jezper/tabgap）
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

cd "${REPO_ROOT}"
mkdir -p "${DOWNLOAD_DIR}" "${SRC_DIR}"

is_valid_download_file() {
  local file_path="$1"
  if [[ ! -s "${file_path}" ]]; then
    return 1
  fi
  if [[ "${file_path}" == *.tar.gz ]]; then
    tar -tf "${file_path}" >/dev/null 2>&1 || return 1
  fi
  return 0
}

download_first_available() {
  local output_file="$1"
  shift
  local urls=("$@")
  if is_valid_download_file "${output_file}"; then
    echo "[跳过] 已下载：${output_file}"
    return 0
  fi
  rm -f "${output_file}"
  local ok=0
  for u in "${urls[@]}"; do
    if [[ -z "${u}" ]]; then
      continue
    fi
    echo "[下载] ${u}"
    if wget -O "${output_file}" --tries=3 --timeout=20 --continue --no-verbose "${u}"; then
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
  if ! tar -tf "${tarball}" >/dev/null 2>&1; then
    echo "[错误] 压缩包不可用或损坏：${tarball}" >&2
    return 1
  fi
  local topdir
  topdir="$(tar -tf "${tarball}" | awk -F/ 'NR==1{first=$1} END{print first}')"
  if [[ -n "${topdir}" && -d "${SRC_DIR}/${topdir}" ]]; then
    mv "${SRC_DIR}/${topdir}" "${expected_dir}"
    echo "[完成] 复用已解压目录：${expected_dir}"
    return 0
  fi
  echo "[解压] ${tarball} -> ${expected_dir}"
  tar -xf "${tarball}" -C "${SRC_DIR}"
  if [[ -n "${topdir}" && "${SRC_DIR}/${topdir}" != "${expected_dir}" && -d "${SRC_DIR}/${topdir}" && ! -d "${expected_dir}" ]]; then
    mv "${SRC_DIR}/${topdir}" "${expected_dir}"
  fi
  echo "[完成] 解压完成：${expected_dir}"
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
    for attempt in 1 2 3; do
      rm -rf "${dest}"
      echo "[克隆] ${r} -> ${dest} (尝试 ${attempt}/3)"
      if git clone --depth 1 "${r}" "${dest}"; then
        ok=1
        break 2
      fi
      sleep 2
    done
  done
  if [[ "${ok}" -ne 1 ]]; then
    echo "[错误] 无法克隆仓库：${dest}" >&2
    return 1
  fi
}

download_repo_archive() {
  local archive_url="$1"
  local dest="$2"
  local temp_tar="${DOWNLOAD_DIR}/repo_archive_$(date +%s%N).tar.gz"
  echo "[下载归档] ${archive_url}"
  wget -O "${temp_tar}" --tries=3 --timeout=20 --continue --no-verbose "${archive_url}"
  local topdir
  topdir="$(tar -tf "${temp_tar}" | awk -F/ 'NR==1{first=$1} END{print first}')"
  rm -rf "${dest}"
  tar -xf "${temp_tar}" -C "${SRC_DIR}"
  if [[ -n "${topdir}" && -d "${SRC_DIR}/${topdir}" ]]; then
    mv "${SRC_DIR}/${topdir}" "${dest}"
  fi
  rm -f "${temp_tar}"
}

LMP_2025_TAR="${DOWNLOAD_DIR}/lammps-10Dec2025.tar.gz"
LMP_2022_TAR="${DOWNLOAD_DIR}/lammps-23Jun2022.tar.gz"
INTEL_MPI_INSTALLER="${DOWNLOAD_DIR}/intel_mpi_or_hpc_toolkit.sh"

LMP_2025_URLS=(
  "${LAMMPS_2025_URL:-}"
  "https://github.com/lammps/lammps/archive/refs/tags/patch_10Dec2025.tar.gz"
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
  "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/4f5871da-0533-4f62-b563-905edfb2e9b7/l_mpi_oneapi_p_2021.10.0.49374_offline.sh"
  "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/1ff1b38a-8218-4c53-9956-f0b264de35a4/l_HPCKit_p_2023.1.0.46346_offline.sh"
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
NEP_PLUGIN_ARCHIVE_URL="${NEP_PLUGIN_ARCHIVE_URL:-https://codeload.github.com/brucefan1983/NEP_CPU/tar.gz/refs/heads/master}"

echo "[阶段] 下载 tabGAP 插件源码"
clone_or_update_repo "${SRC_DIR}/tabgap-plugin" "${TABGAP_PLUGIN_CANDIDATES[@]}"

echo "[阶段] 下载 NEP_CPU 插件源码"
if ! clone_or_update_repo "${SRC_DIR}/nep-cpu-plugin" "${NEP_PLUGIN_CANDIDATES[@]}"; then
  download_repo_archive "${NEP_PLUGIN_ARCHIVE_URL}" "${SRC_DIR}/nep-cpu-plugin"
fi

echo "================================================================================"
echo "下载完成"
echo "LAMMPS 2025 源码: ${SRC_DIR}/lammps-10Dec2025"
echo "LAMMPS 2022 源码: ${SRC_DIR}/lammps-23Jun2022"
echo "Intel MPI 安装包: ${INTEL_MPI_INSTALLER}"
echo "tabGAP 插件目录 : ${SRC_DIR}/tabgap-plugin"
echo "NEP_CPU 插件目录: ${SRC_DIR}/nep-cpu-plugin"
echo "================================================================================"
