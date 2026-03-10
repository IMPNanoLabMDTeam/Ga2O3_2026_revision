#!/usr/bin/env bash
set -euo pipefail

cat <<'INTRO'
================================================================================
脚本 1：编译两个 LAMMPS 版本并执行基础测试
================================================================================
用途：
1) 编译 LAMMPS 10Dec2025（OpenMP 编译）
2) 编译 LAMMPS 23Jun2022（使用 Intel MPI 编译）
3) 两个版本都尝试接入 tabGAP 与 NEP_CPU 插件
4) 产物统一放入当前仓库 opt 目录
5) 编译后执行基础自检（-h 与 run 0）

输入依赖（由脚本 0 准备）：
- opt/src/lammps-10Dec2025
- opt/src/lammps-23Jun2022
- opt/src/tabgap-plugin
- opt/src/nep-cpu-plugin
- opt/downloads/intel_mpi_or_hpc_toolkit.sh

输出：
- opt/lammps-10Dec2025/bin/lmp
- opt/lammps-23Jun2022-intelmpi/bin/lmp
- opt/build/lammps-10Dec2025
- opt/build/lammps-23Jun2022-intelmpi

说明：
- 默认会尝试在本仓库安装并优先使用 Intel MPI；可设置 INSTALL_INTEL_MPI=0 关闭本地安装。
- 默认尝试自动探测 Intel MPI 环境；若失败可手动导出 I_MPI_ROOT 再运行。
================================================================================
INTRO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SRC_DIR="${REPO_ROOT}/opt/src"
BUILD_ROOT="${REPO_ROOT}/opt/build"
INSTALL_ROOT="${REPO_ROOT}/opt"
INTEL_LOCAL_ROOT="${REPO_ROOT}/opt/intel/oneapi"
INTEL_INSTALLER="${REPO_ROOT}/opt/downloads/intel_mpi_or_hpc_toolkit.sh"

LMP2025_SRC="${SRC_DIR}/lammps-10Dec2025"
LMP2022_SRC="${SRC_DIR}/lammps-23Jun2022"
TABGAP_SRC="${SRC_DIR}/tabgap-plugin"
NEP_SRC="${SRC_DIR}/nep-cpu-plugin"

BUILD_2025="${BUILD_ROOT}/lammps-10Dec2025"
BUILD_2022="${BUILD_ROOT}/lammps-23Jun2022-intelmpi"

PREFIX_2025="${INSTALL_ROOT}/lammps-10Dec2025"
PREFIX_2022="${INSTALL_ROOT}/lammps-23Jun2022-intelmpi"

INSTALL_INTEL_MPI="${INSTALL_INTEL_MPI:-1}"
PREFER_LOCAL_INTEL_MPI="${PREFER_LOCAL_INTEL_MPI:-1}"

mkdir -p "${BUILD_ROOT}" "${PREFIX_2025}/bin" "${PREFIX_2022}/bin"

require_dir() {
  local d="$1"
  if [[ ! -d "${d}" ]]; then
    echo "[错误] 缺少目录：${d}" >&2
    exit 1
  fi
}

require_dir "${LMP2025_SRC}"
require_dir "${LMP2022_SRC}"
require_dir "${TABGAP_SRC}"
require_dir "${NEP_SRC}"

load_intel_env_file() {
  local env_file="$1"
  if [[ -f "${env_file}" ]]; then
    set +u
    source "${env_file}" >/dev/null 2>&1
    set -u
  fi
}

detect_impi_root_from_commands() {
  if [[ -n "${I_MPI_ROOT:-}" && -d "${I_MPI_ROOT}" ]]; then
    return 0
  fi
  local mpiicc_path
  mpiicc_path="$(command -v mpiicc || true)"
  if [[ -n "${mpiicc_path}" ]]; then
    local guessed_root
    guessed_root="$(cd "$(dirname "${mpiicc_path}")/.." && pwd)"
    if [[ -x "${guessed_root}/bin/mpicc" || -x "${guessed_root}/bin/mpiicc" ]]; then
      export I_MPI_ROOT="${guessed_root}"
      return 0
    fi
  fi
  return 1
}

find_local_impi_root() {
  local candidates=(
    "${INTEL_LOCAL_ROOT}/mpi/latest"
    "${INTEL_LOCAL_ROOT}/mpi"
    "${INTEL_LOCAL_ROOT}"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -x "${c}/bin/mpiicc" || -x "${c}/bin/mpicc" ]]; then
      echo "${c}"
      return 0
    fi
  done
  return 1
}

install_tabgap_to_lammps() {
  local lammps_src="$1"
  require_dir "${TABGAP_SRC}/lammps"
  if [[ ! -f "${TABGAP_SRC}/lammps/pair_tabgap.cpp" || ! -f "${TABGAP_SRC}/lammps/pair_tabgap.h" ]]; then
    echo "[错误] tabGAP 插件缺少 pair_tabgap.cpp / pair_tabgap.h" >&2
    exit 1
  fi
  echo "[插件] 安装 tabGAP -> ${lammps_src}"
  cp "${TABGAP_SRC}/lammps/pair_tabgap.cpp" "${lammps_src}/src/"
  cp "${TABGAP_SRC}/lammps/pair_tabgap.h" "${lammps_src}/src/"
}

enable_user_nep_package() {
  local lammps_src="$1"
  local cmake_lists="${lammps_src}/cmake/CMakeLists.txt"
  local nep_pkg="${lammps_src}/cmake/Modules/Packages/USER-NEP.cmake"
  cp "${NEP_SRC}/interface/lammps/USER-NEP.cmake" "${nep_pkg}"
  python3 - "${cmake_lists}" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
lines = path.read_text().splitlines()
for i, line in enumerate(lines):
    if "foreach(PKG_WITH_INCL " in line and "USER-NEP" not in line and line.rstrip().endswith(")"):
        lines[i] = line.rstrip()[:-1] + " USER-NEP)"
        break
start = -1
end = -1
for i, line in enumerate(lines):
    if "set(STANDARD_PACKAGES" in line:
        start = i
        continue
    if start != -1 and line.strip() == ")":
        end = i
        break
if start != -1 and end != -1:
    block_text = "\n".join(lines[start:end + 1])
    if "USER-NEP" not in block_text:
        lines.insert(end, "  USER-NEP")
path.write_text("\n".join(lines) + "\n")
PY
}

install_nep_to_lammps() {
  local lammps_src="$1"
  require_dir "${NEP_SRC}/interface/lammps"
  require_dir "${NEP_SRC}/src"
  echo "[插件] 安装 NEP_CPU -> ${lammps_src}"
  rm -rf "${lammps_src}/src/USER-NEP"
  cp -r "${NEP_SRC}/interface/lammps/USER-NEP" "${lammps_src}/src/"
  cp "${NEP_SRC}/src/"* "${lammps_src}/src/USER-NEP/"
  enable_user_nep_package "${lammps_src}"
}

prepare_intel_mpi() {
  local local_impi_root=""
  local_impi_root="$(find_local_impi_root || true)"
  if [[ "${PREFER_LOCAL_INTEL_MPI}" == "1" && -n "${local_impi_root}" ]]; then
    export I_MPI_ROOT="${local_impi_root}"
    load_intel_env_file "${I_MPI_ROOT}/env/vars.sh"
    return 0
  fi

  if [[ -n "${I_MPI_ROOT:-}" && -d "${I_MPI_ROOT}" ]]; then
    return 0
  fi

  load_intel_env_file "${INTEL_LOCAL_ROOT}/setvars.sh"
  load_intel_env_file "${INTEL_LOCAL_ROOT}/mpi/latest/env/vars.sh"
  local_impi_root="$(find_local_impi_root || true)"
  if [[ "${PREFER_LOCAL_INTEL_MPI}" == "1" && -n "${local_impi_root}" ]]; then
    export I_MPI_ROOT="${local_impi_root}"
    load_intel_env_file "${I_MPI_ROOT}/env/vars.sh"
    return 0
  fi
  load_intel_env_file "/opt/intel/oneapi/setvars.sh"
  load_intel_env_file "/opt/intel/oneapi/mpi/latest/env/vars.sh"
  detect_impi_root_from_commands || true

  if [[ -n "${I_MPI_ROOT:-}" && -d "${I_MPI_ROOT}" ]]; then
    return 0
  fi

  if [[ "${INSTALL_INTEL_MPI}" == "1" ]]; then
    if [[ ! -x "${INTEL_INSTALLER}" ]]; then
      echo "[错误] 未找到可执行 Intel 安装包：${INTEL_INSTALLER}" >&2
      exit 1
    fi
    mkdir -p "${INTEL_LOCAL_ROOT}"
    bash "${INTEL_INSTALLER}" -a --silent --eula accept --install-dir "${INTEL_LOCAL_ROOT}" || \
      bash "${INTEL_INSTALLER}" --silent --eula accept --install-dir "${INTEL_LOCAL_ROOT}" || true
    load_intel_env_file "${INTEL_LOCAL_ROOT}/setvars.sh"
    load_intel_env_file "${INTEL_LOCAL_ROOT}/mpi/latest/env/vars.sh"
    local_impi_root="$(find_local_impi_root || true)"
    if [[ -n "${local_impi_root}" ]]; then
      export I_MPI_ROOT="${local_impi_root}"
      load_intel_env_file "${I_MPI_ROOT}/env/vars.sh"
    fi
    detect_impi_root_from_commands || true
  fi

  if [[ -z "${I_MPI_ROOT:-}" || ! -d "${I_MPI_ROOT}" ]]; then
    echo "[错误] Intel MPI 环境未就绪，请先安装并导出 I_MPI_ROOT。" >&2
    echo "      可选1：INSTALL_INTEL_MPI=1 bash reviewer_tests/1_build_lammps.sh" >&2
    echo "      可选2：source /opt/intel/oneapi/setvars.sh && bash reviewer_tests/1_build_lammps.sh" >&2
    exit 1
  fi
}

build_lammps_2025() {
  echo "[构建] LAMMPS 10Dec2025（OpenMP）"
  rm -rf "${BUILD_2025}"
  cmake -S "${LMP2025_SRC}/cmake" -B "${BUILD_2025}" \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_MPI=off \
    -D BUILD_OMP=on \
    -D PKG_MANYBODY=on \
    -D PKG_USER-NEP=on
  cmake --build "${BUILD_2025}" -j"$(nproc)"
  cp "${BUILD_2025}/lmp" "${PREFIX_2025}/bin/lmp"
  chmod +x "${PREFIX_2025}/bin/lmp"
}

build_lammps_2022_with_intel_mpi() {
  echo "[构建] LAMMPS 23Jun2022（Intel MPI）"
  prepare_intel_mpi

  local mpi_cxx="${I_MPI_ROOT}/bin/mpicxx"
  local mpi_cc="${I_MPI_ROOT}/bin/mpicc"
  local mpi_fc="${I_MPI_ROOT}/bin/mpifort"
  if [[ ! -x "${mpi_cxx}" ]]; then
    mpi_cxx="${I_MPI_ROOT}/bin/mpiicpc"
  fi
  if [[ ! -x "${mpi_cc}" ]]; then
    mpi_cc="${I_MPI_ROOT}/bin/mpiicc"
  fi
  if [[ ! -x "${mpi_fc}" ]]; then
    mpi_fc="${I_MPI_ROOT}/bin/mpiifort"
  fi
  if [[ ! -x "${mpi_cxx}" ]]; then
    mpi_cxx="$(command -v mpiicpc || true)"
  fi
  if [[ ! -x "${mpi_cc}" ]]; then
    mpi_cc="$(command -v mpiicc || true)"
  fi
  if [[ ! -x "${mpi_fc}" ]]; then
    mpi_fc="$(command -v mpiifort || true)"
  fi
  if [[ -z "${mpi_cc}" || -z "${mpi_cxx}" ]]; then
    echo "[错误] 未找到 Intel MPI 编译器包装器 mpiicc/mpiicpc" >&2
    exit 1
  fi

  rm -rf "${BUILD_2022}"
  cmake -S "${LMP2022_SRC}/cmake" -B "${BUILD_2022}" \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_MPI=on \
    -D BUILD_OMP=on \
    -D PKG_MANYBODY=on \
    -D PKG_USER-NEP=on \
    -D CMAKE_C_COMPILER="${mpi_cc}" \
    -D CMAKE_CXX_COMPILER="${mpi_cxx}" \
    -D CMAKE_Fortran_COMPILER="${mpi_fc}"
  cmake --build "${BUILD_2022}" -j"$(nproc)"
  cp "${BUILD_2022}/lmp" "${PREFIX_2022}/bin/lmp"
  chmod +x "${PREFIX_2022}/bin/lmp"
}

run_smoke_test() {
  local lmp_bin="$1"
  local label="$2"
  local tmp_in
  local tmp_help
  tmp_in="$(mktemp)"
  tmp_help="$(mktemp)"
  cat > "${tmp_in}" <<'EOF'
units metal
dimension 3
boundary p p p
atom_style atomic
lattice sc 3.0
region box block 0 1 0 1 0 1
create_box 1 box
create_atoms 1 box
mass 1 69.723
pair_style zero 10.0
pair_coeff * *
run 0
EOF

  echo "[测试] ${label} -h"
  "${lmp_bin}" -h > "${tmp_help}"
  head -n 40 "${tmp_help}" >/dev/null

  echo "[测试] ${label} run 0"
  "${lmp_bin}" -in "${tmp_in}" -log none >/dev/null

  echo "[测试] ${label} 插件关键字检查（tabgap / nep）"
  if grep -Eiq 'tabgap|nep' "${tmp_help}"; then
    echo "[通过] ${label} 检测到 tabgap/nep 关键字"
  else
    echo "[警告] ${label} 未在 -h 中检测到 tabgap/nep 关键字，请人工确认插件是否生效"
  fi

  rm -f "${tmp_in}" "${tmp_help}"
}

echo "[阶段] 向两个版本源码接入插件"
install_tabgap_to_lammps "${LMP2025_SRC}"
install_nep_to_lammps "${LMP2025_SRC}"
install_tabgap_to_lammps "${LMP2022_SRC}"
install_nep_to_lammps "${LMP2022_SRC}"

echo "[阶段] 编译 LAMMPS 2025"
build_lammps_2025

echo "[阶段] 编译 LAMMPS 2022 (Intel MPI)"
build_lammps_2022_with_intel_mpi

echo "[阶段] 执行基础测试"
run_smoke_test "${PREFIX_2025}/bin/lmp" "LAMMPS 10Dec2025"
run_smoke_test "${PREFIX_2022}/bin/lmp" "LAMMPS 23Jun2022 + Intel MPI"

echo "================================================================================"
echo "编译与测试完成"
echo "2025 可执行文件: ${PREFIX_2025}/bin/lmp"
echo "2022 可执行文件: ${PREFIX_2022}/bin/lmp"
echo "================================================================================"
