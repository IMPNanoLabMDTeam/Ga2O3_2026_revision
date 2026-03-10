#!/usr/bin/env python3
"""
误差分析脚本
分析LAMMPS预测值与DFT参考值的误差
包括：能量、力、virial
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_xyz_properties(xyz_file: str) -> Dict:
    """
    从xyz文件中提取DFT参考值
    包括：能量、力、virial、config_type
    """
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        raise ValueError(f"文件 {xyz_file} 格式不正确")
    
    n_atoms = int(lines[0].strip())
    header_line = lines[1].strip()
    
    # 提取 config_type
    config_type_match = re.search(r'config_type=(\S+)', header_line, re.IGNORECASE)
    config_type = config_type_match.group(1) if config_type_match else None
    
    # 提取能量 (eV) - 注意：已经是总能量，需要除以原子数
    energy_match = re.search(r'energy=([-\d.]+)', header_line, re.IGNORECASE)
    if energy_match:
        total_energy = float(energy_match.group(1))
        energy_per_atom = total_energy / n_atoms
    else:
        energy_per_atom = None
    
    # 提取virial (eV) - 格式为3x3矩阵的9个值: xx xy xz xy yy yz xz yz zz
    virial_match = re.search(r'virial="([^"]*)"', header_line, re.IGNORECASE)
    if virial_match:
        virial_str = virial_match.group(1)
        virial_values = [float(x) for x in virial_str.split()]
        
        if len(virial_values) == 9:
            # 提取对称矩阵的6个独立分量: xx, yy, zz, xy, xz, yz
            virial = [
                virial_values[0],  # xx
                virial_values[4],  # yy
                virial_values[8],  # zz
                virial_values[1],  # xy (或 virial_values[3])
                virial_values[2],  # xz (或 virial_values[6])
                virial_values[5],  # yz (或 virial_values[7])
            ]
            # 转换为每原子virial (eV/atom)
            virial_per_atom = [v / n_atoms for v in virial]
        else:
            virial_per_atom = None
    else:
        virial_per_atom = None
    
    # 提取力 (eV/Å)
    forces = []
    for i in range(2, 2 + n_atoms):
        if i >= len(lines):
            break
        parts = lines[i].strip().split()
        # 假设格式: element x y z fx fy fz ...
        if len(parts) >= 7:
            fx, fy, fz = float(parts[4]), float(parts[5]), float(parts[6])
            forces.append([fx, fy, fz])
    
    forces = np.array(forces) if forces else None
    
    return {
        'n_atoms': n_atoms,
        'energy_per_atom': energy_per_atom,
        'virial_per_atom': virial_per_atom,
        'forces': forces,
        'config_type': config_type
    }


def parse_lammps_forces(dump_file: str) -> np.ndarray:
    """从dump.forces文件中提取力"""
    data = []
    with open(dump_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('ITEM:'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                # 格式: id type x y z fx fy fz
                fx, fy, fz = float(parts[5]), float(parts[6]), float(parts[7])
                data.append([fx, fy, fz])
    return np.array(data)


def parse_lammps_summary(summary_file: str) -> Dict:
    """从summary.txt文件中提取LAMMPS计算结果"""
    try:
        with open(summary_file, 'r') as f:
            content = f.read()
        
        result = {}
        
        # 提取平均能量 (eV/atom)
        energy_match = re.search(r'Average potential energy \(eV/atom\):\s*([-\d.]+)', content)
        if energy_match:
            result['energy'] = float(energy_match.group(1))
        else:
            result['energy'] = None
        
        # 提取virial分量 (eV/atom)
        virial_pattern = r'Average virial per atom \(eV/atom\):\s*vxx\s*=\s*([-\d.]+)\s*vyy\s*=\s*([-\d.]+)\s*vzz\s*=\s*([-\d.]+)\s*vxy\s*=\s*([-\d.]+)\s*vxz\s*=\s*([-\d.]+)\s*vyz\s*=\s*([-\d.]+)'
        virial_match = re.search(virial_pattern, content)
        if virial_match:
            vxx = float(virial_match.group(1))
            vyy = float(virial_match.group(2))
            vzz = float(virial_match.group(3))
            vxy = float(virial_match.group(4))
            vxz = float(virial_match.group(5))
            vyz = float(virial_match.group(6))
            result['virial'] = [vxx, vyy, vzz, vxy, vxz, vyz]
        else:
            result['virial'] = None
        
        return result
    except Exception as e:
        return {'energy': None, 'virial': None}


def parse_lammps_energy(dump_file: str) -> float:
    """从dump.pe文件中提取平均能量（备用方法）"""
    energies = []
    with open(dump_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('ITEM:'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                # 格式: id type pe
                pe = float(parts[2])
                energies.append(pe)
    return np.mean(energies) if energies else None


def parse_lammps_virial(dump_file: str) -> List[float]:
    """从dump.virial文件中提取平均virial (eV/atom)（备用方法）"""
    virials = []
    with open(dump_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('ITEM:'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                # 格式: id type vxx vyy vzz vxy vxz vyz
                # 注意：这些值已经在run.in中转换为eV单位
                vxx, vyy, vzz = float(parts[2]), float(parts[3]), float(parts[4])
                vxy, vxz, vyz = float(parts[5]), float(parts[6]), float(parts[7])
                virials.append([vxx, vyy, vzz, vxy, vxz, vyz])
    
    if virials:
        # 返回平均virial (eV/atom) - 对所有原子的virial取平均
        return np.mean(virials, axis=0).tolist()
    return None


def collect_data(root_dir: str) -> Tuple[Dict, Dict, List]:
    """
    收集所有子文件夹的DFT参考值和LAMMPS预测值
    返回: (dft_data, lammps_data, energy_details)
    energy_details: [(dir_name, dft_energy, lammps_energy, abs_error, rel_error), ...]
    """
    root_path = Path(root_dir)
    
    dft_data = {
        'energy': [],
        'forces': [],
        'virial': [],
        'config_types': [],  # 存储每个结构的 config_type
        'force_structure_indices': [],  # 存储每个力分量对应的结构索引
        'virial_structure_indices': []  # 存储每个virial分量对应的结构索引
    }
    
    lammps_data = {
        'energy': [],
        'forces': [],
        'virial': [],
        'config_types': [],
        'force_structure_indices': [],
        'virial_structure_indices': []
    }
    
    energy_details = []  # 存储详细的能量误差信息
    
    subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    
    print(f"正在收集数据...")
    success_count = 0
    fail_count = 0
    structure_index = 0  # 结构索引计数器
    
    for subdir in subdirs:
        xyz_file = subdir / "structure.xyz"
        force_file = subdir / "dump.forces"
        summary_file = subdir / "summary.txt"
        
        # 检查必要文件是否存在
        if not xyz_file.exists():
            fail_count += 1
            continue
        
        if not (force_file.exists() and summary_file.exists()):
            fail_count += 1
            continue
        
        try:
            # 解析DFT参考值
            dft_props = parse_xyz_properties(str(xyz_file))
            
            # 解析LAMMPS预测值
            lammps_forces = parse_lammps_forces(str(force_file))
            
            # 优先从summary.txt读取能量和virial
            lammps_summary = parse_lammps_summary(str(summary_file))
            lammps_energy = lammps_summary['energy']
            lammps_virial = lammps_summary['virial']
            
            # 检查数据完整性
            if dft_props['energy_per_atom'] is not None and lammps_energy is not None:
                dft_data['energy'].append(dft_props['energy_per_atom'])
                lammps_data['energy'].append(lammps_energy)
                dft_data['config_types'].append(dft_props['config_type'])
                lammps_data['config_types'].append(dft_props['config_type'])
                
                # 计算能量误差
                abs_error = abs(lammps_energy - dft_props['energy_per_atom'])
                if abs(dft_props['energy_per_atom']) > 1e-6:
                    rel_error = abs_error / abs(dft_props['energy_per_atom']) * 100  # 百分比
                else:
                    rel_error = float('inf')
                
                energy_details.append((
                    subdir.name,
                    dft_props['energy_per_atom'],
                    lammps_energy,
                    abs_error,
                    rel_error,
                    dft_props['config_type']  # 添加 config_type
                ))
            
            if dft_props['forces'] is not None and len(lammps_forces) > 0:
                dft_forces_flat = dft_props['forces'].flatten()
                lammps_forces_flat = lammps_forces.flatten()
                dft_data['forces'].append(dft_forces_flat)
                lammps_data['forces'].append(lammps_forces_flat)
                # 为每个力分量记录对应的结构索引
                dft_data['force_structure_indices'].extend([structure_index] * len(dft_forces_flat))
                lammps_data['force_structure_indices'].extend([structure_index] * len(lammps_forces_flat))
            
            if dft_props['virial_per_atom'] is not None and lammps_virial is not None:
                dft_data['virial'].append(dft_props['virial_per_atom'])
                lammps_data['virial'].append(lammps_virial)
                # 为每个virial分量记录对应的结构索引
                dft_data['virial_structure_indices'].extend([structure_index] * len(dft_props['virial_per_atom']))
                lammps_data['virial_structure_indices'].extend([structure_index] * len(lammps_virial))
            
            success_count += 1
            structure_index += 1  # 递增结构索引
            
        except Exception as e:
            print(f"  警告: {subdir.name} 处理失败 - {str(e)}")
            fail_count += 1
    
    print(f"数据收集完成: 成功 {success_count} 个，失败 {fail_count} 个")
    
    # 转换为numpy数组
    dft_data['energy'] = np.array(dft_data['energy'])
    lammps_data['energy'] = np.array(lammps_data['energy'])
    dft_data['config_types'] = np.array(dft_data['config_types'])
    lammps_data['config_types'] = np.array(lammps_data['config_types'])
    
    if dft_data['forces']:
        dft_data['forces'] = np.concatenate(dft_data['forces'])
        lammps_data['forces'] = np.concatenate(lammps_data['forces'])
        dft_data['force_structure_indices'] = np.array(dft_data['force_structure_indices'])
        lammps_data['force_structure_indices'] = np.array(lammps_data['force_structure_indices'])
    else:
        dft_data['forces'] = np.array([])
        lammps_data['forces'] = np.array([])
        dft_data['force_structure_indices'] = np.array([])
        lammps_data['force_structure_indices'] = np.array([])
    
    if dft_data['virial']:
        dft_data['virial'] = np.array(dft_data['virial']).flatten()
        lammps_data['virial'] = np.array(lammps_data['virial']).flatten()
        dft_data['virial_structure_indices'] = np.array(dft_data['virial_structure_indices'])
        lammps_data['virial_structure_indices'] = np.array(lammps_data['virial_structure_indices'])
    else:
        dft_data['virial'] = np.array([])
        lammps_data['virial'] = np.array([])
        dft_data['virial_structure_indices'] = np.array([])
        lammps_data['virial_structure_indices'] = np.array([])
    
    return dft_data, lammps_data, energy_details


def calculate_r2(true, pred):
    """计算决定系数R²"""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1 - (ss_res / ss_tot)


def plot_phase_comparison(dft_data: Dict, lammps_data: Dict, output_file: str, dataset_name: str = ''):
    """绘制按相分类的对比图（包括能量、力、位力）"""
    # 定义相的分类和颜色
    phase_config = {
        'beta': {
            'name': 'Beta Phase',
            'config_types': ['bulk_beta_phase'],
            'color': '#1f77b4',  # 蓝色
            'marker': 'o'
        },
        'gamma': {
            'name': 'Gamma Phase',
            'config_types': ['bulk_gamma_phase'],
            'color': '#ff7f0e',  # 橙色
            'marker': 's'
        },
        'melted': {
            'name': 'Melted Phase',
            'config_types': ['RSS', 'melted_phase'],
            'color': '#2ca02c',  # 绿色
            'marker': '^'
        }
    }
    
    # 创建2x2子图布局
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    if dataset_name:
        main_title = f'DFT vs LAMMPS by Phase - {dataset_name} (Low Energy Region)'
    else:
        main_title = 'DFT vs LAMMPS by Phase (Low Energy Region)'
    
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    # 1. 能量子图（左上）
    print("\n能量分析（按相分类）:")
    all_energy_dft = []
    all_energy_lammps = []
    
    for phase_key, phase_info in phase_config.items():
        mask = np.isin(dft_data['config_types'], phase_info['config_types'])
        
        if np.sum(mask) > 0 and len(dft_data['energy']) > 0:
            phase_dft = dft_data['energy'][mask]
            phase_lammps = lammps_data['energy'][mask]
            
            all_energy_dft.extend(phase_dft)
            all_energy_lammps.extend(phase_lammps)
            
            axs[0, 0].scatter(phase_dft, phase_lammps, 
                            alpha=0.6, s=30, 
                            color=phase_info['color'],
                            marker=phase_info['marker'],
                            edgecolor='none',
                            label=f"{phase_info['name']} (n={len(phase_dft)})")
            
            rmse = np.sqrt(np.mean((phase_lammps - phase_dft)**2))
            mae = np.mean(np.abs(phase_lammps - phase_dft))
            print(f"  {phase_info['name']}: n={len(phase_dft)}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    if all_energy_dft:
        all_energy_dft = np.array(all_energy_dft)
        all_energy_lammps = np.array(all_energy_lammps)
        min_val = min(all_energy_dft.min(), all_energy_lammps.min())
        max_val = max(all_energy_dft.max(), all_energy_lammps.max())
        axs[0, 0].plot([min_val, max_val], [min_val, max_val], 
                      color='red', linewidth=1.5, linestyle='--', label='Perfect')
        
        rmse_total = np.sqrt(np.mean((all_energy_lammps - all_energy_dft)**2))
        mae_total = np.mean(np.abs(all_energy_lammps - all_energy_dft))
        r2_total = calculate_r2(all_energy_dft, all_energy_lammps)
        
        stats_text = f'Overall:\nRMSE = {rmse_total:.4f} eV/atom\nMAE = {mae_total:.4f} eV/atom\nR² = {r2_total:.4f}'
        axs[0, 0].text(0.05, 0.95, stats_text, transform=axs[0, 0].transAxes,
                      verticalalignment='top', fontsize=12,
                      bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
    
    axs[0, 0].set_xlabel('DFT Energy (eV/atom)', fontsize=12)
    axs[0, 0].set_ylabel('LAMMPS Energy (eV/atom)', fontsize=12)
    axs[0, 0].set_title('Energy Comparison', fontsize=14)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(loc='lower right', fontsize=9)
    
    # 2. 力子图（右上）
    print("\n力分析（按相分类）:")
    if len(dft_data['forces']) > 0 and len(dft_data.get('force_structure_indices', [])) > 0:
        all_force_dft = []
        all_force_lammps = []
        
        # 为每个力分量获取对应的 config_type
        force_config_types = dft_data['config_types'][dft_data['force_structure_indices']]
        
        for phase_key, phase_info in phase_config.items():
            mask = np.isin(force_config_types, phase_info['config_types'])
            
            if np.sum(mask) > 0:
                phase_dft = dft_data['forces'][mask]
                phase_lammps = lammps_data['forces'][mask]
                
                all_force_dft.extend(phase_dft)
                all_force_lammps.extend(phase_lammps)
                
                axs[0, 1].scatter(phase_dft, phase_lammps, 
                                alpha=0.3, s=5, 
                                color=phase_info['color'],
                                marker=phase_info['marker'],
                                edgecolor='none',
                                label=f"{phase_info['name']} (n={len(phase_dft)})")
                
                rmse = np.sqrt(np.mean((phase_lammps - phase_dft)**2))
                mae = np.mean(np.abs(phase_lammps - phase_dft))
                print(f"  {phase_info['name']}: n={len(phase_dft)}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        if all_force_dft:
            all_force_dft = np.array(all_force_dft)
            all_force_lammps = np.array(all_force_lammps)
            min_val = min(all_force_dft.min(), all_force_lammps.min())
            max_val = max(all_force_dft.max(), all_force_lammps.max())
            axs[0, 1].plot([min_val, max_val], [min_val, max_val], 
                          color='red', linewidth=1.5, linestyle='--', label='Perfect')
            
            rmse_total = np.sqrt(np.mean((all_force_lammps - all_force_dft)**2))
            mae_total = np.mean(np.abs(all_force_lammps - all_force_dft))
            r2_total = calculate_r2(all_force_dft, all_force_lammps)
            
            stats_text = f'Overall:\nRMSE = {rmse_total:.4f} eV/Å\nMAE = {mae_total:.4f} eV/Å\nR² = {r2_total:.4f}'
            axs[0, 1].text(0.05, 0.95, stats_text, transform=axs[0, 1].transAxes,
                          verticalalignment='top', fontsize=12,
                          bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            print(f"  Overall: n={len(all_force_dft)}, RMSE={rmse_total:.4f}, MAE={mae_total:.4f}")
        
        axs[0, 1].legend(loc='lower right', fontsize=9)
    else:
        axs[0, 1].axis('off')
        axs[0, 1].set_title('Force Data Not Available', fontsize=14)
    
    axs[0, 1].set_xlabel('DFT Force (eV/Å)', fontsize=12)
    axs[0, 1].set_ylabel('LAMMPS Force (eV/Å)', fontsize=12)
    axs[0, 1].set_title('Force Comparison', fontsize=14)
    axs[0, 1].grid(True, alpha=0.3)
    
    # 3. Virial子图（左下）
    print("\nVirial分析（按相分类）:")
    if len(dft_data['virial']) > 0 and len(dft_data.get('virial_structure_indices', [])) > 0:
        all_virial_dft = []
        all_virial_lammps = []
        
        # 为每个virial分量获取对应的 config_type
        virial_config_types = dft_data['config_types'][dft_data['virial_structure_indices']]
        
        # 按相分类绘制virial数据
        for phase_key, phase_info in phase_config.items():
            mask = np.isin(virial_config_types, phase_info['config_types'])
            
            if np.sum(mask) > 0:
                phase_dft = dft_data['virial'][mask]
                phase_lammps = lammps_data['virial'][mask]
                
                all_virial_dft.extend(phase_dft)
                all_virial_lammps.extend(phase_lammps)
                
                axs[1, 0].scatter(phase_dft, phase_lammps, 
                                alpha=0.3, s=5, 
                                color=phase_info['color'],
                                marker=phase_info['marker'],
                                edgecolor='none',
                                label=f"{phase_info['name']} (n={len(phase_dft)})")
                
                rmse = np.sqrt(np.mean((phase_lammps - phase_dft)**2))
                mae = np.mean(np.abs(phase_lammps - phase_dft))
                print(f"  {phase_info['name']}: n={len(phase_dft)}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        if all_virial_dft:
            all_virial_dft = np.array(all_virial_dft)
            all_virial_lammps = np.array(all_virial_lammps)
            min_val = min(all_virial_dft.min(), all_virial_lammps.min())
            max_val = max(all_virial_dft.max(), all_virial_lammps.max())
            axs[1, 0].plot([min_val, max_val], [min_val, max_val], 
                          color='red', linewidth=1.5, linestyle='--', label='Perfect')
            
            rmse_virial = np.sqrt(np.mean((all_virial_lammps - all_virial_dft)**2))
            mae_virial = np.mean(np.abs(all_virial_lammps - all_virial_dft))
            r2_virial = calculate_r2(all_virial_dft, all_virial_lammps)
            
            stats_text = f'Overall:\nRMSE = {rmse_virial:.4f} eV/atom\nMAE = {mae_virial:.4f} eV/atom\nR² = {r2_virial:.4f}'
            axs[1, 0].text(0.05, 0.95, stats_text, transform=axs[1, 0].transAxes,
                          verticalalignment='top', fontsize=12,
                          bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            print(f"  Overall: n={len(all_virial_dft)}, RMSE={rmse_virial:.4f}, MAE={mae_virial:.4f}")
        
        axs[1, 0].legend(loc='lower right', fontsize=9)
    else:
        axs[1, 0].axis('off')
        axs[1, 0].set_title('Virial Data Not Available', fontsize=14)
    
    axs[1, 0].set_xlabel('DFT Virial (eV/atom)', fontsize=12)
    axs[1, 0].set_ylabel('LAMMPS Virial (eV/atom)', fontsize=12)
    axs[1, 0].set_title('Virial Comparison', fontsize=14)
    axs[1, 0].grid(True, alpha=0.3)
    
    # 4. 隐藏右下角子图
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n按相分类图表已保存到: {output_file}")


def plot_comparison(dft_data: Dict, lammps_data: Dict, output_file: str, title_suffix: str = '', dataset_name: str = '', color_by_config: bool = False):
    """绘制对比图"""
    # 使用 matplotlib 的默认颜色循环来自动分配颜色
    default_colors = plt.cm.tab10.colors  # 10种不同的颜色
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # 构建标题
    if dataset_name:
        main_title = f'DFT vs LAMMPS Predictions - {dataset_name}{title_suffix}'
    else:
        main_title = f'DFT vs LAMMPS Predictions{title_suffix}'
    
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    line_kwargs = {'color': 'red', 'linewidth': 1.5, 'linestyle': '--'}
    stats_kwargs = {'fontsize': 12, 'bbox': {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8}}
    
    # 1. Energy子图
    if len(dft_data['energy']) > 0:
        dft_energy = dft_data['energy']
        lammps_energy = lammps_data['energy']
        config_types = dft_data['config_types']
        
        rmse_energy = np.sqrt(np.mean((lammps_energy - dft_energy)**2))
        mae_energy = np.mean(np.abs(lammps_energy - dft_energy))
        r2_energy = calculate_r2(dft_energy, lammps_energy)
        
        if color_by_config and len(config_types) > 0:
            # 按 config_type 分类绘制，动态分配颜色
            unique_configs = np.unique(config_types[config_types != None])  # 排除 None 值
            for i, config_type in enumerate(unique_configs):
                if config_type is None:
                    continue
                mask = config_types == config_type
                color = default_colors[i % len(default_colors)]  # 循环使用颜色
                axs[0, 0].scatter(dft_energy[mask], lammps_energy[mask], 
                                alpha=0.6, s=20, color=color, edgecolor='none',
                                label=f'{config_type} (n={np.sum(mask)})')
            
            # 处理 None 值（如果有的话）
            none_mask = config_types == None
            if np.sum(none_mask) > 0:
                axs[0, 0].scatter(dft_energy[none_mask], lammps_energy[none_mask], 
                                alpha=0.6, s=20, color='#cccccc', edgecolor='none',
                                label=f'unknown (n={np.sum(none_mask)})')
            
            axs[0, 0].legend(loc='lower right', fontsize=9)
        else:
            # 统一颜色绘制
            scatter_kwargs = {'alpha': 0.4, 's': 15, 'edgecolor': 'none'}
            axs[0, 0].scatter(dft_energy, lammps_energy, **scatter_kwargs)
        
        min_val = min(dft_energy.min(), lammps_energy.min())
        max_val = max(dft_energy.max(), lammps_energy.max())
        axs[0, 0].plot([min_val, max_val], [min_val, max_val], **line_kwargs)
        axs[0, 0].set_xlabel('DFT Energy (eV/atom)', fontsize=12)
        axs[0, 0].set_ylabel('LAMMPS Energy (eV/atom)', fontsize=12)
        axs[0, 0].set_title('Energy Comparison', fontsize=14)
        axs[0, 0].grid(True, alpha=0.3)
        
        stats_text = f'RMSE = {rmse_energy:.4f} eV/atom\nMAE = {mae_energy:.4f} eV/atom\nR² = {r2_energy:.4f}'
        axs[0, 0].text(0.05, 0.95, stats_text, transform=axs[0, 0].transAxes,
                      verticalalignment='top', **stats_kwargs)
    else:
        axs[0, 0].axis('off')
        axs[0, 0].set_title('Energy Data Not Available', fontsize=14)
    
    # 2. Force子图
    if len(dft_data['forces']) > 0:
        dft_force = dft_data['forces']
        lammps_force = lammps_data['forces']
        
        rmse_force = np.sqrt(np.mean((lammps_force - dft_force)**2))
        mae_force = np.mean(np.abs(lammps_force - dft_force))
        r2_force = calculate_r2(dft_force, lammps_force)
        
        if color_by_config and len(dft_data.get('force_structure_indices', [])) > 0:
            # 为每个力分量获取对应的 config_type
            force_config_types = dft_data['config_types'][dft_data['force_structure_indices']]
            unique_configs = np.unique(force_config_types[force_config_types != None])
            
            for i, config_type in enumerate(unique_configs):
                if config_type is None:
                    continue
                mask = force_config_types == config_type
                color = default_colors[i % len(default_colors)]
                axs[0, 1].scatter(dft_force[mask], lammps_force[mask], 
                                alpha=0.3, s=5, color=color, edgecolor='none',
                                label=f'{config_type} (n={np.sum(mask)})')
            
            # 处理 None 值
            none_mask = force_config_types == None
            if np.sum(none_mask) > 0:
                axs[0, 1].scatter(dft_force[none_mask], lammps_force[none_mask], 
                                alpha=0.3, s=5, color='#cccccc', edgecolor='none',
                                label=f'unknown (n={np.sum(none_mask)})')
            
            axs[0, 1].legend(loc='lower right', fontsize=9)
        else:
            scatter_kwargs = {'alpha': 0.4, 's': 15, 'edgecolor': 'none'}
            axs[0, 1].scatter(dft_force, lammps_force, **scatter_kwargs)
        
        min_val = min(dft_force.min(), lammps_force.min())
        max_val = max(dft_force.max(), lammps_force.max())
        axs[0, 1].plot([min_val, max_val], [min_val, max_val], **line_kwargs)
        axs[0, 1].set_xlabel('DFT Force (eV/Å)', fontsize=12)
        axs[0, 1].set_ylabel('LAMMPS Force (eV/Å)', fontsize=12)
        axs[0, 1].set_title('Force Comparison', fontsize=14)
        axs[0, 1].grid(True, alpha=0.3)
        
        stats_text = f'RMSE = {rmse_force:.4f} eV/Å\nMAE = {mae_force:.4f} eV/Å\nR² = {r2_force:.4f}'
        axs[0, 1].text(0.05, 0.95, stats_text, transform=axs[0, 1].transAxes,
                      verticalalignment='top', **stats_kwargs)
    else:
        axs[0, 1].axis('off')
        axs[0, 1].set_title('Force Data Not Available', fontsize=14)
    
    # 3. Virial子图
    if len(dft_data['virial']) > 0:
        dft_virial = dft_data['virial']
        lammps_virial = lammps_data['virial']
        
        rmse_virial = np.sqrt(np.mean((lammps_virial - dft_virial)**2))
        mae_virial = np.mean(np.abs(lammps_virial - dft_virial))
        r2_virial = calculate_r2(dft_virial, lammps_virial)
        
        if color_by_config and len(dft_data.get('virial_structure_indices', [])) > 0:
            # 为每个virial分量获取对应的 config_type
            virial_config_types = dft_data['config_types'][dft_data['virial_structure_indices']]
            unique_configs = np.unique(virial_config_types[virial_config_types != None])
            
            for i, config_type in enumerate(unique_configs):
                if config_type is None:
                    continue
                mask = virial_config_types == config_type
                color = default_colors[i % len(default_colors)]
                axs[1, 0].scatter(dft_virial[mask], lammps_virial[mask], 
                                alpha=0.3, s=5, color=color, edgecolor='none',
                                label=f'{config_type} (n={np.sum(mask)})')
            
            # 处理 None 值
            none_mask = virial_config_types == None
            if np.sum(none_mask) > 0:
                axs[1, 0].scatter(dft_virial[none_mask], lammps_virial[none_mask], 
                                alpha=0.3, s=5, color='#cccccc', edgecolor='none',
                                label=f'unknown (n={np.sum(none_mask)})')
            
            axs[1, 0].legend(loc='lower right', fontsize=9)
        else:
            scatter_kwargs = {'alpha': 0.4, 's': 15, 'edgecolor': 'none'}
            axs[1, 0].scatter(dft_virial, lammps_virial, **scatter_kwargs)
        
        min_val = min(dft_virial.min(), lammps_virial.min())
        max_val = max(dft_virial.max(), lammps_virial.max())
        axs[1, 0].plot([min_val, max_val], [min_val, max_val], **line_kwargs)
        axs[1, 0].set_xlabel('DFT Virial (eV/atom)', fontsize=12)
        axs[1, 0].set_ylabel('LAMMPS Virial (eV/atom)', fontsize=12)
        axs[1, 0].set_title('Virial Comparison', fontsize=14)
        axs[1, 0].grid(True, alpha=0.3)
        
        stats_text = f'RMSE = {rmse_virial:.4f} eV/atom\nMAE = {mae_virial:.4f} eV/atom\nR² = {r2_virial:.4f}'
        axs[1, 0].text(0.05, 0.95, stats_text, transform=axs[1, 0].transAxes,
                      verticalalignment='top', **stats_kwargs)
    else:
        axs[1, 0].axis('off')
        axs[1, 0].set_title('Virial Data Not Available', fontsize=14)
    
    # 4. 预留子图（可用于其他分析）
    axs[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_file}")


def save_energy_details_csv(energy_details: List, output_file: str):
    """保存详细的能量误差到CSV文件"""
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow([
            'Structure_ID',
            'DFT_Energy_eV_per_atom',
            'LAMMPS_Energy_eV_per_atom',
            'Absolute_Error_eV_per_atom',
            'Relative_Error_percent'
        ])
        
        # 写入数据
        for detail in energy_details:
            if len(detail) == 6:
                dir_name, dft_e, lammps_e, abs_err, rel_err, config_type = detail
            else:
                # 兼容旧格式
                dir_name, dft_e, lammps_e, abs_err, rel_err = detail
                config_type = 'unknown'
            
            # 处理无穷大的相对误差
            rel_err_str = f"{rel_err:.6f}" if rel_err != float('inf') else "inf"
            writer.writerow([
                dir_name,
                f"{dft_e:.12f}",
                f"{lammps_e:.12f}",
                f"{abs_err:.12f}",
                rel_err_str
            ])
    
    print(f"详细能量误差已保存到: {output_file}")


def save_summary(dft_data: Dict, lammps_data: Dict, output_file: str, energy_details: List = None):
    """保存统计摘要"""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("误差分析摘要\n")
        f.write("=" * 80 + "\n\n")
        
        # Energy
        if len(dft_data['energy']) > 0:
            dft_energy = dft_data['energy']
            lammps_energy = lammps_data['energy']
            rmse = np.sqrt(np.mean((lammps_energy - dft_energy)**2))
            mae = np.mean(np.abs(lammps_energy - dft_energy))
            r2 = calculate_r2(dft_energy, lammps_energy)
            
            f.write("能量 (Energy)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  样本数: {len(dft_energy)}\n")
            f.write(f"  RMSE: {rmse:.6f} eV/atom\n")
            f.write(f"  MAE:  {mae:.6f} eV/atom\n")
            f.write(f"  R²:   {r2:.6f}\n\n")
        
        # Force
        if len(dft_data['forces']) > 0:
            dft_force = dft_data['forces']
            lammps_force = lammps_data['forces']
            rmse = np.sqrt(np.mean((lammps_force - dft_force)**2))
            mae = np.mean(np.abs(lammps_force - dft_force))
            r2 = calculate_r2(dft_force, lammps_force)
            
            f.write("力 (Force)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  样本数: {len(dft_force)}\n")
            f.write(f"  RMSE: {rmse:.6f} eV/Å\n")
            f.write(f"  MAE:  {mae:.6f} eV/Å\n")
            f.write(f"  R²:   {r2:.6f}\n\n")
        
        # Virial
        if len(dft_data['virial']) > 0:
            dft_virial = dft_data['virial']
            lammps_virial = lammps_data['virial']
            rmse = np.sqrt(np.mean((lammps_virial - dft_virial)**2))
            mae = np.mean(np.abs(lammps_virial - dft_virial))
            r2 = calculate_r2(dft_virial, lammps_virial)
            
            f.write("Virial\n")
            f.write("-" * 80 + "\n")
            f.write(f"  样本数: {len(dft_virial)}\n")
            f.write(f"  RMSE: {rmse:.6f} eV/atom\n")
            f.write(f"  MAE:  {mae:.6f} eV/atom\n")
            f.write(f"  R²:   {r2:.6f}\n\n")
        
        # 如果提供了能量详细信息，输出误差最大的10个结构
        if energy_details is not None and len(energy_details) > 0:
            f.write("能量误差最大的10个结构\n")
            f.write("-" * 80 + "\n")
            
            # 按绝对误差排序
            sorted_details = sorted(energy_details, key=lambda x: x[3], reverse=True)[:10]
            
            f.write(f"{'目录':<12} {'DFT能量':<18} {'LAMMPS能量':<18} {'绝对误差':<18} {'相对误差':<12} {'相':<15}\n")
            f.write(f"{'':12} {'(eV/atom)':<18} {'(eV/atom)':<18} {'(eV/atom)':<18} {'(%)':<12} {'':<15}\n")
            f.write("-" * 95 + "\n")
            
            for detail in sorted_details:
                if len(detail) == 6:
                    dir_name, dft_e, lammps_e, abs_err, rel_err, config_type = detail
                else:
                    dir_name, dft_e, lammps_e, abs_err, rel_err = detail
                    config_type = 'unknown'
                
                rel_err_str = f"{rel_err:.2f}" if rel_err != float('inf') else "inf"
                config_type_str = config_type if config_type else 'unknown'
                f.write(f"{dir_name:<12} {dft_e:<18.6f} {lammps_e:<18.6f} {abs_err:<18.6f} {rel_err_str:<12} {config_type_str:<15}\n")
            
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"统计摘要已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="误差分析脚本：对比DFT参考值与LAMMPS预测值",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法（完整分析）
  python scripts/analyze_errors.py run/raw_data/3.3.0_nep2025 -o run/analysis/3.3.0_nep2025
  
  # 简化分析模式（只做全部训练集分析，按config_type分类显示颜色）
  python scripts/analyze_errors.py run/raw_data/3.3.0_nep2025 -o run/analysis/3.3.0_nep2025 --simple
  
  # 自定义低能量区间阈值
  python scripts/analyze_errors.py run/raw_data/3.3.0_nep2025 -o run/analysis/3.3.0_nep2025 -t 5.0
        """
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="输入目录（包含所有子文件夹的根目录）"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认：与输入目录相同）"
    )
    parser.add_argument(
        "-t", "--low-energy-threshold",
        type=float,
        default=3.0,
        help="低能量区间阈值（eV），相对于最低能量的偏移量（默认：3.0）"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="简化分析模式：只做全部训练集分析，不做低能和指定组分析，但绘图时按config_type分类显示不同颜色"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"错误：输入目录 {args.input_dir} 不存在")
        return 1
    
    # 设置输出目录
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path.parent
    
    # 提取数据集名称（输出目录的最后一级文件夹名）
    dataset_name = output_path.name
    
    print("=" * 80)
    print("误差分析")
    print("=" * 80)
    print(f"输入目录: {input_path}")
    print(f"输出目录: {output_path}")
    print(f"数据集名称: {dataset_name}")
    print("=" * 80)
    
    # 收集数据
    dft_data, lammps_data, energy_details = collect_data(str(input_path))
    
    # 检查数据
    if len(dft_data['energy']) == 0 and len(dft_data['forces']) == 0 and len(dft_data['virial']) == 0:
        print("错误：未找到有效数据")
        return 1
    
    # 保存详细能量误差CSV
    if energy_details:
        csv_file = output_path / "energy_errors_detailed.csv"
        save_energy_details_csv(energy_details, str(csv_file))
    
    # 保存统计摘要
    summary_file = output_path / "error_analysis_summary.txt"
    save_summary(dft_data, lammps_data, str(summary_file))
    
    # 绘制对比图
    plot_file = output_path / "error_analysis_comparison.png"
    # 在 simple 模式下启用按 config_type 分类的颜色显示
    plot_comparison(dft_data, lammps_data, str(plot_file), dataset_name=dataset_name, color_by_config=args.simple)
    
    # 如果是简化模式，跳过低能量和相分类分析
    if args.simple:
        print("\n" + "=" * 80)
        print("简化分析模式：跳过低能量和相分类分析")
        print("=" * 80)
        print("\n" + "=" * 80)
        print("分析完成！")
        print("=" * 80)
        return 0
    
    # 低能量区间分析
    print("\n" + "=" * 80)
    print(f"低能量区间分析（DFT能量 < 最低值 + {args.low_energy_threshold} eV）")
    print("=" * 80)
    
    if len(dft_data['energy']) > 0:
        # 找到最低能量
        min_energy = np.min(dft_data['energy'])
        energy_threshold = min_energy + args.low_energy_threshold
        
        print(f"最低DFT能量: {min_energy:.6f} eV/atom")
        print(f"能量阈值: {energy_threshold:.6f} eV/atom")
        
        # 筛选低能量区间的数据
        low_energy_mask = dft_data['energy'] < energy_threshold
        
        # 获取低能量区间的结构索引和 config_types
        low_energy_indices = np.where(low_energy_mask)[0]  # 低能量结构的索引
        low_energy_config_types = dft_data['config_types'][low_energy_mask]
        
        low_energy_dft = {
            'energy': dft_data['energy'][low_energy_mask],
            'forces': np.array([]),
            'virial': np.array([]),
            'config_types': low_energy_config_types,
            'force_structure_indices': np.array([]),
            'virial_structure_indices': np.array([])
        }
        
        low_energy_lammps = {
            'energy': lammps_data['energy'][low_energy_mask],
            'forces': np.array([]),
            'virial': np.array([]),
            'config_types': low_energy_config_types,
            'force_structure_indices': np.array([]),
            'virial_structure_indices': np.array([])
        }
        
        # 创建索引映射：原始索引 -> 新索引
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(low_energy_indices)}
        
        # 对于力和virial，根据结构索引来筛选对应的数据
        if len(dft_data['forces']) > 0 and len(dft_data['force_structure_indices']) > 0:
            # 筛选属于低能量结构的力数据
            force_mask = np.isin(dft_data['force_structure_indices'], low_energy_indices)
            low_energy_dft['forces'] = dft_data['forces'][force_mask]
            low_energy_lammps['forces'] = lammps_data['forces'][force_mask]
            # 重新映射索引到新的范围 [0, len(low_energy_indices))
            old_indices = dft_data['force_structure_indices'][force_mask]
            new_indices = np.array([index_mapping[idx] for idx in old_indices])
            low_energy_dft['force_structure_indices'] = new_indices
            low_energy_lammps['force_structure_indices'] = new_indices
        
        if len(dft_data['virial']) > 0 and len(dft_data['virial_structure_indices']) > 0:
            # 筛选属于低能量结构的virial数据
            virial_mask = np.isin(dft_data['virial_structure_indices'], low_energy_indices)
            low_energy_dft['virial'] = dft_data['virial'][virial_mask]
            low_energy_lammps['virial'] = lammps_data['virial'][virial_mask]
            # 重新映射索引到新的范围 [0, len(low_energy_indices))
            old_indices = dft_data['virial_structure_indices'][virial_mask]
            new_indices = np.array([index_mapping[idx] for idx in old_indices])
            low_energy_dft['virial_structure_indices'] = new_indices
            low_energy_lammps['virial_structure_indices'] = new_indices
        
        print(f"筛选后样本数: {len(low_energy_dft['energy'])} / {len(dft_data['energy'])}")
        
        if len(low_energy_dft['energy']) > 0:
            # 筛选低能量区间的能量详细信息
            low_energy_details = []
            for detail in energy_details:
                if len(detail) == 6:
                    dir_name, dft_e, lammps_e, abs_err, rel_err, config_type = detail
                else:
                    dir_name, dft_e, lammps_e, abs_err, rel_err = detail
                
                if dft_e < energy_threshold:
                    low_energy_details.append(detail)
            
            # 保存低能量区间统计摘要（包含误差最大的结构）
            low_energy_summary_file = output_path / "error_analysis_summary_low_energy.txt"
            save_summary(low_energy_dft, low_energy_lammps, str(low_energy_summary_file), 
                        energy_details=low_energy_details)
            
            # 绘制低能量区间对比图
            low_energy_plot_file = output_path / "error_analysis_comparison_low_energy.png"
            plot_comparison(low_energy_dft, low_energy_lammps, str(low_energy_plot_file), 
                          title_suffix=' (Low Energy Region)', dataset_name=dataset_name)
            
            # 按相分类的专项分析
            print("\n" + "=" * 80)
            print("按相分类的专项分析（低能量区间）")
            print("=" * 80)
            
            # 绘制按相分类的对比图
            phase_plot_file = output_path / "error_analysis_comparison_by_phase.png"
            plot_phase_comparison(low_energy_dft, low_energy_lammps, str(phase_plot_file), 
                                dataset_name=dataset_name)
        else:
            print("警告：筛选后没有数据点")
    else:
        print("跳过：没有能量数据")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
