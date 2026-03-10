#!/usr/bin/env python3
"""
LAMMPS计算完整工作流
整合所有步骤：分割xyz -> 转换为LAMMPS格式 -> 链接力场和脚本 -> 批量运行LAMMPS

用法示例：
  # 默认使用2025版
  python scripts/run_lammps_workflow.py train_dataset/nep_baseline/nep2025.xyz -f forcefield/nep/3.3.0.txt

  # 选择2022版
  python scripts/run_lammps_workflow.py train_dataset/nep_baseline/nep2025.xyz -f forcefield/nep/3.3.0.txt --lammps-version 2022

  # 手动指定可执行文件（覆盖版本默认）
  python scripts/run_lammps_workflow.py train_dataset/nep_baseline/nep2025.xyz -f forcefield/nep/3.3.0.txt --lammps /custom/path/lmp
"""

import argparse
import os
import sys
import subprocess
import numpy as np
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed


class XyzToLammpsConverter:
    """GPUMD xyz格式到LAMMPS data文件的转换器"""
    
    def __init__(self):
        self.element_to_mass = {
            'O': 15.9994,
            'Ga': 69.723,
            'Al': 26.9815,
            'In': 114.818,
        }
        self.standard_element_order = ['O', 'Ga']
    
    def parse_xyz_file(self, filename: str) -> Dict:
        """解析GPUMD xyz文件"""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            raise ValueError(f"文件 {filename} 格式不正确")
        
        n_atoms = int(lines[0].strip())
        header_line = lines[1].strip()
        
        lattice_match = re.search(r'Lattice="([^"]*)"', header_line)
        if not lattice_match:
            raise ValueError("未找到晶格信息")
        
        lattice_str = lattice_match.group(1)
        lattice_values = [float(x) for x in lattice_str.split()]
        
        if len(lattice_values) != 9:
            raise ValueError(f"晶格参数应为9个数值")
        
        lattice_vectors = np.array(lattice_values).reshape(3, 3)
        has_velocities = 'vel:R:3' in header_line
        
        atoms = []
        for i in range(2, 2 + n_atoms):
            if i >= len(lines):
                raise ValueError(f"文件中原子数据不足")
                
            parts = lines[i].strip().split()
            if has_velocities and len(parts) < 7:
                raise ValueError(f"第{i+1}行数据不完整")
            elif not has_velocities and len(parts) < 4:
                raise ValueError(f"第{i+1}行数据不完整")
            
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            
            atom_data = {'element': element, 'x': x, 'y': y, 'z': z}
            
            if has_velocities and len(parts) >= 7:
                atom_data['vx'] = float(parts[4])
                atom_data['vy'] = float(parts[5])
                atom_data['vz'] = float(parts[6])
            
            atoms.append(atom_data)
        
        return {
            'n_atoms': n_atoms,
            'lattice_vectors': lattice_vectors,
            'atoms': atoms,
            'has_velocities': has_velocities
        }
    
    def get_lammps_box_and_transform(self, lattice_vectors: np.ndarray) -> tuple:
        """
        从晶格向量计算LAMMPS盒子参数和坐标变换矩阵
        
        LAMMPS要求晶格向量满足特定约束：
        - a向量沿x轴: a' = [lx, 0, 0]
        - b向量在xy平面: b' = [xy, ly, 0]
        - c向量任意: c' = [xz, yz, lz]
        """
        a = lattice_vectors[0]
        b = lattice_vectors[1]
        c = lattice_vectors[2]
        
        la = np.linalg.norm(a)
        lb = np.linalg.norm(b)
        lc = np.linalg.norm(c)
        
        cos_alpha = np.dot(b, c) / (lb * lc)
        cos_beta = np.dot(a, c) / (la * lc)
        cos_gamma = np.dot(a, b) / (la * lb)
        
        lx = la
        xy = lb * cos_gamma
        ly = np.sqrt(lb**2 - xy**2)
        xz = lc * cos_beta
        yz = (lb * lc * cos_alpha - xy * xz) / ly
        lz = np.sqrt(lc**2 - xz**2 - yz**2)

        eps = 1e-8
        if abs(xy) >= 0.5 * lx:
            xy = np.sign(xy) * (0.5 * lx - eps)
        if abs(xz) >= 0.5 * lx:
            xz = np.sign(xz) * (0.5 * lx - eps)
        if abs(yz) >= 0.5 * ly:
            yz = np.sign(yz) * (0.5 * ly - eps)
        
        is_triclinic = (abs(xy) > 1e-6 or abs(xz) > 1e-6 or abs(yz) > 1e-6)
        
        box_params = {
            'xlo': 0.0, 'xhi': lx,
            'ylo': 0.0, 'yhi': ly,
            'zlo': 0.0, 'zhi': lz,
            'xy': xy, 'xz': xz, 'yz': yz,
            'is_triclinic': is_triclinic
        }
        
        new_lattice = np.array([
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz]
        ])
        
        transform_matrix = np.linalg.inv(lattice_vectors) @ new_lattice
        
        return box_params, transform_matrix
    
    def convert_to_lammps(self, xyz_file: str, output_file: str) -> str:
        """将xyz文件转换为LAMMPS data格式"""
        data = self.parse_xyz_file(xyz_file)
        box_params, transform_matrix = self.get_lammps_box_and_transform(data['lattice_vectors'])
        
        # 固定原子类型映射：O=1, Ga=2
        atom_type_map = {'O': 1, 'Ga': 2}
        
        comment = f"LAMMPS data file converted from {Path(xyz_file).name}"
        
        with open(output_file, 'w') as f:
            f.write(f"# {comment}\n")
            f.write(f"{data['n_atoms']} atoms\n")
            f.write(f"2 atom types\n")
            
            f.write(f"{box_params['xlo']:.12f} {box_params['xhi']:.12f} xlo xhi\n")
            f.write(f"{box_params['ylo']:.12f} {box_params['yhi']:.12f} ylo yhi\n")
            f.write(f"{box_params['zlo']:.12f} {box_params['zhi']:.12f} zlo zhi\n")
            
            if box_params['is_triclinic']:
                f.write(f"{box_params['xy']:.12f} {box_params['xz']:.12f} {box_params['yz']:.12f} xy xz yz\n")
            
            f.write("\n")
            f.write("Masses\n\n")
            f.write(f"1 {self.element_to_mass['O']:.4f}  # O\n")
            f.write(f"2 {self.element_to_mass['Ga']:.4f}  # Ga\n")
            f.write("\n")
            f.write("Atoms  # atomic\n\n")
            
            for i, atom in enumerate(data['atoms']):
                atom_id = i + 1
                atom_type = atom_type_map[atom['element']]
                r_old = np.array([atom['x'], atom['y'], atom['z']])
                r_new = r_old @ transform_matrix
                x, y, z = r_new[0], r_new[1], r_new[2]
                f.write(f"{atom_id} {atom_type} {x:.12f} {y:.12f} {z:.12f}\n")
        
        return output_file


def read_xyz_frames(xyz_file):
    """读取xyz文件并返回所有帧"""
    frames = []
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        try:
            n_atoms = int(lines[i].strip())
        except (ValueError, IndexError):
            break
        
        frame_lines = lines[i:i + n_atoms + 2]
        
        if len(frame_lines) < n_atoms + 2:
            break
        
        frames.append(frame_lines)
        i += n_atoms + 2
    
    return frames


def save_frames_to_folders(frames, output_dir, start_index=0):
    """将帧保存到编号的子文件夹中"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx, frame in enumerate(frames):
        frame_num = start_index + idx
        frame_dir = output_path / f"{frame_num:06d}"
        frame_dir.mkdir(exist_ok=True)
        
        output_file = frame_dir / "structure.xyz"
        with open(output_file, 'w') as f:
            f.writelines(frame)


def convert_structures_to_lammps(root_dir: str):
    """转换所有子文件夹中的structure.xyz为model.data"""
    root_path = Path(root_dir)
    converter = XyzToLammpsConverter()
    
    subdirs = [d for d in root_path.iterdir() if d.is_dir() and (d / "structure.xyz").exists()]
    
    success_count = 0
    fail_count = 0
    
    for subdir in sorted(subdirs):
        input_file = subdir / "structure.xyz"
        output_file = subdir / "model.data"
        
        try:
            converter.convert_to_lammps(str(input_file), str(output_file))
            success_count += 1
        except Exception as e:
            print(f"  ✗ {subdir.name}: 转换失败 - {str(e)}")
            fail_count += 1
    
    return success_count, fail_count


def create_symlinks_forcefield(forcefield_dir: str, target_root_dir: str, potential_type: str = "tabgap"):
    """在所有子文件夹中创建力场文件的软连接"""
    forcefield_path = Path(forcefield_dir).resolve()
    target_root_path = Path(target_root_dir).resolve()
    
    # 判断是文件还是目录
    if forcefield_path.is_file():
        # 如果是单个文件，直接使用该文件
        forcefield_files = [forcefield_path]
    elif forcefield_path.is_dir():
        # 如果是目录，获取目录中的所有文件
        forcefield_files = [f for f in forcefield_path.iterdir() if f.is_file()]
    else:
        print(f"  警告：力场路径 {forcefield_dir} 不存在")
        return 0
    
    subdirs = [d for d in target_root_path.iterdir() if d.is_dir()]
    
    success_count = 0
    
    for subdir in sorted(subdirs):
        for ff_file in forcefield_files:
            # NEP势函数特殊处理：统一命名为nep.txt
            if potential_type == "nep":
                link_name = "nep.txt"
            else:
                link_name = ff_file.name
            
            link_path = subdir / link_name
            
            if link_path.exists() or link_path.is_symlink():
                continue
            
            try:
                rel_path = os.path.relpath(ff_file, subdir)
                os.symlink(rel_path, link_path)
                success_count += 1
            except Exception:
                pass
    
    return success_count


def create_symlinks_run_script(source_file: str, target_root_dir: str):
    """在所有子文件夹中创建run.in的软连接"""
    source_path = Path(source_file).resolve()
    target_root_path = Path(target_root_dir).resolve()
    
    subdirs = [d for d in target_root_path.iterdir() if d.is_dir()]
    
    success_count = 0
    
    for subdir in sorted(subdirs):
        link_path = subdir / "run.in"
        
        if link_path.exists() or link_path.is_symlink():
            continue
        
        try:
            rel_path = os.path.relpath(source_path, subdir)
            os.symlink(rel_path, link_path)
            success_count += 1
        except Exception:
            pass
    
    return success_count


def run_lammps_in_directory(lammps_exe: Path, work_dir: Path) -> tuple:
    """在指定目录中运行LAMMPS"""
    input_path = work_dir / "run.in"
    
    if not input_path.exists():
        return False, "run.in不存在"
    
    cmd = [str(lammps_exe), "-in", "run.in", "-log", "lammps.log"]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            return True, "成功"
        else:
            return False, f"失败 (返回码 {result.returncode})"
            
    except subprocess.TimeoutExpired:
        return False, "超时"
    except Exception as e:
        return False, f"异常: {str(e)}"


def run_lammps_wrapper(args_tuple):
    """包装函数用于并行执行"""
    lammps_exe, work_dir, index, total = args_tuple
    dir_name = work_dir.name
    success, message = run_lammps_in_directory(lammps_exe, work_dir)
    return index, dir_name, success, message


def generate_test_name(train_xyz: str, forcefield: str) -> str:
    """
    根据训练集和力场自动生成测试名称
    
    规则：
    - 训练集名称：从文件名提取（去掉.xyz后缀），如 nep2025.xyz -> nep2025
    - 力场名称：
      - NEP: 使用文件名（去掉.txt后缀），如 3.3.0.txt -> 3.3.0
      - tabGAP: 使用 "tabgap"
    - 最终格式：{力场名称}_{训练集名称}
    
    示例：
    - train_dataset/nep_baseline/nep2025.xyz + forcefield/nep/3.3.0.txt -> 3.3.0_nep2025
    - train_dataset/nep_baseline/npj2023.xyz + forcefield/tabgap -> tabgap_npj2023
    """
    # 提取训练集名称：使用文件名（去掉.xyz）
    train_path = Path(train_xyz)
    dataset_name = train_path.stem
    
    # 提取力场名称
    ff_path = Path(forcefield)
    if ff_path.is_file() and ff_path.suffix == '.txt':
        # NEP势函数：使用文件名（去掉.txt）
        ff_name = ff_path.stem
    elif ff_path.is_dir() or ff_path.name == 'tabgap':
        # tabGAP势函数
        ff_name = 'tabgap'
    else:
        # 其他情况，使用文件/目录名
        ff_name = ff_path.stem if ff_path.is_file() else ff_path.name
    
    return f"{ff_name}_{dataset_name}"


def main():
    parser = argparse.ArgumentParser(
        description="LAMMPS计算完整工作流（包含误差分析）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
完整工作流包括：
  1. 分割xyz文件为单独的帧
  2. 转换为LAMMPS data格式
  3. 链接力场文件
  4. 链接run.in脚本
  5. 批量运行LAMMPS计算
  6. 误差分析（可选）

简化用法（自动生成名称）:
  # 使用 NEP 势函数（自动生成名称：3.3.0_nep2025，默认完整分析）
  python scripts/run_lammps_workflow.py train_dataset/nep_baseline/nep2025.xyz -f forcefield/nep/3.3.0.txt
  
  # 使用 tabGAP 势函数（自动生成名称：tabgap_nep2025）
  python scripts/run_lammps_workflow.py train_dataset/nep_baseline/nep2025.xyz -f forcefield/tabgap
  
  # 使用简化分析模式（只做全部训练集分析）
  python scripts/run_lammps_workflow.py train_dataset/nep_baseline/nep2025.xyz -f forcefield/nep/2.3.1.txt --simple
  
  # 跳过误差分析
  python scripts/run_lammps_workflow.py train_dataset/nep_baseline/nep2025.xyz -f forcefield/nep/2.3.1.txt --skip-analysis
  
  # 只运行前10个任务（测试）
  python scripts/run_lammps_workflow.py train_dataset/nep_baseline/nep2025.xyz -f forcefield/nep/2.3.1.txt --max-jobs 10
  
手动指定名称:
  # 如果需要自定义名称，使用 -n 参数
  python scripts/run_lammps_workflow.py train_dataset/nep_baseline/nep2025.xyz -n my_custom_name -f forcefield/nep/3.0.0.txt
        """
    )
    
    parser.add_argument(
        "train_xyz",
        type=str,
        help="训练集xyz文件路径"
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        default=None,
        help="测试名称（可选，不指定则自动生成）"
    )
    parser.add_argument(
        "-f", "--forcefield",
        type=str,
        required=True,
        help="力场文件或文件夹路径（.txt文件=NEP，文件夹=tabGAP）"
    )
    parser.add_argument(
        "--run-script",
        type=str,
        default=None,
        help="run.in脚本路径（默认：根据势函数类型自动选择）"
    )
    parser.add_argument(
        "--lammps",
        type=str,
        default=None,
        help="LAMMPS可执行文件或命令（默认根据版本标签选择）"
    )
    parser.add_argument(
        "--lammps-version",
        type=str,
        choices=["2025", "2022"],
        default="2025",
        help="LAMMPS版本标签（默认：2025）"
    )
    parser.add_argument(
        "--lammps-2025",
        type=str,
        default="/home/abel/opt/lammps-nep-gap/build/lmp",
        help="2025版LAMMPS可执行文件路径"
    )
    parser.add_argument(
        "--lammps-2022",
        type=str,
        default="/home/abel/opt/lammps-23Jun2022-intelmpi-2021.17.2/bin/lmp",
        help="2022版LAMMPS可执行文件路径"
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="最大运行任务数（默认：无限制）"
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="跳过LAMMPS运行步骤（只准备文件）"
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=None,
        help="并行运行的进程数（默认：CPU核心数）"
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="跳过误差分析步骤（默认会自动进行误差分析）"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="简化分析模式：只做全部训练集分析，不做低能和指定组分析，但绘图时按config_type分类显示不同颜色"
    )
    parser.add_argument(
        "-t", "--low-energy-threshold",
        type=float,
        default=0.5,
        help="误差分析的低能量区间阈值（eV）（默认：0.5，仅在非简化模式下使用）"
    )
    
    args = parser.parse_args()
    lammps_version_label = args.lammps_version
    if args.lammps is None:
        args.lammps = args.lammps_2025 if lammps_version_label == "2025" else args.lammps_2022
    
    # 自动检测势函数类型
    ff_path = Path(args.forcefield)
    if ff_path.is_file() and ff_path.suffix == '.txt':
        potential_type = "nep"
    elif ff_path.is_dir():
        potential_type = "tabgap"
    else:
        print(f"警告：无法识别力场类型，默认使用 tabgap")
        potential_type = "tabgap"
    
    # 路径管理逻辑
    workspace_root = Path(__file__).parent.parent
    
    # 如果没有指定名称，自动生成
    if args.name is None:
        test_name = generate_test_name(args.train_xyz, args.forcefield)
        print(f"自动生成测试名称: {test_name}")
    else:
        test_name = args.name
    
    raw_data_dir = workspace_root / "run" / "raw_data" / test_name
    analysis_dir = workspace_root / "run" / "analysis" / test_name
    output_dir = raw_data_dir
    
    # 根据势函数类型设置默认值
    if args.run_script is None:
        if potential_type == "nep":
            args.run_script = "scripts/run_nep.in"
        else:  # tabgap
            args.run_script = "scripts/run_gap.in"
    
    # 检查输入文件
    train_xyz_path = Path(args.train_xyz)
    if not train_xyz_path.exists():
        print(f"错误：训练集文件 {args.train_xyz} 不存在")
        return 1
    
    print("=" * 80)
    print("LAMMPS计算完整工作流")
    print("=" * 80)
    print(f"测试名称: {test_name}")
    if analysis_dir:
        print(f"原始数据目录: {output_dir}")
        print(f"分析结果目录: {analysis_dir}")
    else:
        print(f"输出目录: {output_dir}")
    print(f"势函数类型: {potential_type.upper()}")
    print(f"训练集: {train_xyz_path}")
    print(f"Run脚本: {args.run_script}")
    print(f"力场: {args.forcefield}")
    print(f"LAMMPS: {args.lammps} (版本: {lammps_version_label})")
    if not args.skip_analysis:
        if args.simple:
            print(f"误差分析: 启用（简化模式）")
        else:
            print(f"误差分析: 启用（完整模式，低能量阈值: {args.low_energy_threshold} eV）")
    else:
        print(f"误差分析: 跳过")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # 步骤1: 分割xyz文件
    print("\n[步骤 1/5] 分割xyz文件为单独的帧...")
    frames = read_xyz_frames(str(train_xyz_path))
    print(f"  读取到 {len(frames)} 帧")
    
    save_frames_to_folders(frames, str(output_dir))
    print(f"  ✓ 已保存到 {output_dir}")
    
    # 步骤2: 转换为LAMMPS格式
    print("\n[步骤 2/5] 转换structure.xyz为LAMMPS格式...")
    success, fail = convert_structures_to_lammps(str(output_dir))
    print(f"  ✓ 转换完成: 成功 {success} 个，失败 {fail} 个")
    
    # 步骤3: 链接力场文件
    print("\n[步骤 3/5] 链接力场文件...")
    ff_count = create_symlinks_forcefield(args.forcefield, str(output_dir), potential_type)
    print(f"  ✓ 创建 {ff_count} 个力场文件软连接")
    
    # 步骤4: 链接run.in脚本
    print("\n[步骤 4/5] 链接run.in脚本...")
    run_count = create_symlinks_run_script(args.run_script, str(output_dir))
    print(f"  ✓ 创建 {run_count} 个run.in软连接")
    
    # 步骤5: 批量运行LAMMPS
    if not args.skip_run:
        print("\n[步骤 5/5] 批量运行LAMMPS计算...")
        
        lammps_candidate = args.lammps
        found = shutil.which(lammps_candidate)
        lammps_path = Path(found) if found else Path(lammps_candidate).resolve()
        if not lammps_path.exists() or not os.access(str(lammps_path), os.X_OK):
            print(f"  错误：无法找到可执行的 LAMMPS ({args.lammps})，请检查PATH或提供绝对路径")
            return 1
        
        subdirs = [d for d in output_dir.iterdir() if d.is_dir() and (d / "run.in").exists()]
        subdirs = sorted(subdirs)
        
        if args.max_jobs is not None and args.max_jobs > 0:
            subdirs = subdirs[:args.max_jobs]
            print(f"  限制运行前 {args.max_jobs} 个任务")
        
        # 确定并行进程数
        n_cores = args.n_cores if args.n_cores else os.cpu_count()
        print(f"  开始运行 {len(subdirs)} 个任务（使用 {n_cores} 个进程）...")
        
        success_count = 0
        fail_count = 0
        completed = 0
        
        # 准备任务参数
        tasks = [(lammps_path, subdir, i+1, len(subdirs)) for i, subdir in enumerate(subdirs)]
        
        # 并行执行
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(run_lammps_wrapper, task): task for task in tasks}
            
            # 处理完成的任务
            for future in as_completed(future_to_task):
                completed += 1
                try:
                    index, dir_name, success, message = future.result()
                    
                    if success:
                        print(f"  [{completed}/{len(subdirs)}] {dir_name}: ✓ {message}")
                        success_count += 1
                    else:
                        print(f"  [{completed}/{len(subdirs)}] {dir_name}: ✗ {message}")
                        fail_count += 1
                        
                except Exception as e:
                    print(f"  [{completed}/{len(subdirs)}] 任务异常: {str(e)}")
                    fail_count += 1
        
        print(f"  ✓ 运行完成: 成功 {success_count} 个，失败 {fail_count} 个")
    else:
        print("\n[步骤 5/5] 跳过LAMMPS运行步骤")
    
    # 步骤6: 误差分析（默认开启）
    if not args.skip_analysis and not args.skip_run and analysis_dir:
        print("\n[步骤 6/6] 运行误差分析...")
        
        try:
            # 导入误差分析模块
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            
            # 构建分析命令
            analysis_cmd = [
                sys.executable,
                str(Path(__file__).parent / "analyze_errors.py"),
                str(output_dir),
                "-o", str(analysis_dir),
            ]
            
            # 简化模式：只做全部训练集分析
            if args.simple:
                analysis_cmd.append("--simple")
                print("  使用简化分析模式（仅全部训练集分析）")
            else:
                # 完整模式：包含低能和指定组分析
                analysis_cmd.extend(["-t", str(args.low_energy_threshold)])
                print(f"  使用完整分析模式（低能量阈值: {args.low_energy_threshold} eV）")
            
            print(f"  运行命令: {' '.join(analysis_cmd)}")
            
            result = subprocess.run(
                analysis_cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("  ✓ 误差分析完成")
                print(f"  分析结果保存在: {analysis_dir}")
            else:
                print(f"  ✗ 误差分析失败")
                if result.stderr:
                    print(f"  错误信息: {result.stderr}")
        
        except Exception as e:
            print(f"  ✗ 误差分析出错: {str(e)}")
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print(f"工作流完成！总耗时: {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print("=" * 80)
    
    print(f"\n结果位置:")
    print(f"  原始数据: {output_dir}")
    if not args.skip_analysis and analysis_dir:
        print(f"  分析结果: {analysis_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
