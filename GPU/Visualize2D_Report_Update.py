## Copyright@2024 Shuoyu Yue
## Email: sy481@cam.ac.uk
## Created on 21/02/2025
## Description: Used to visualize the simulation results of written assignment

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
import math


## Global Parameters ##
gamma = 1.4

# case_id = 1
# case_name = 'QudrantTest'
# tStop = 0.3
# figure_id = 11

case_id = 2
case_name = 'Shock-Bubble'
tStop = 7.8
figure_id = 15

if case_id in [1]:
    x0, x1 = 0.0, 1.0
    y0, y1 = 0.0, 1.0
    nCellsX, nCellsY = 400, 400
    var_id_list = [0, 1, 2, 3]  # var_id: 0-Density, 1-Velocity, 2-Pressure, 3-Specific Energy

if case_id in [2]:
    x0, x1 = 0.0, 0.225
    y0, y1 = 0.0, 0.089
    nCellsX, nCellsY = 500, 197
    var_id_list = [0, 1, 2, 3]  # var_id: 0-Density, 1-Velocity, 2-Pressure, 3-Specific Energy

dx = (x1 - x0) / nCellsX
dy = (y1 - y0) / nCellsY


## visualization preparation
if case_id in [1]:
    fig, axes = plt.subplots(2, 4, figsize=(33, 13), dpi=300)
if case_id in [2]:
    fig, axes = plt.subplots(2, 4, figsize=(32, 7), dpi=300, gridspec_kw={"hspace": 0.4, "wspace": 0.1})

label_list = ['Density', 'Velocity', 'Pressure', 'Specific Energy']


## extract files
folder_path_cpu = "D:/Study_Master/WrittenAssignment/WorkSpace_CUDA/CPU/res/Case_%d/" % case_id
folder_path_gpu = "D:/Study_Master/WrittenAssignment/WorkSpace_CUDA/GPU/res/Case_%d/" % case_id
folder_path_list = [folder_path_cpu, folder_path_gpu]
device_list = ['CPU', 'GPU']


## visualization function
def visualize_single(folder_path, device, cur_ax, var_id, t):
    
    file_name_1 = "T=%d.txt" % t
    file_path_1 = os.path.join(folder_path, file_name_1)
    file_name_2 = "T=%.1f.txt" % t
    file_path_2 = os.path.join(folder_path, file_name_2)
    file_name_3 = "T=%.2f.txt" % t
    file_path_3 = os.path.join(folder_path, file_name_3)

    ## data storage
    rho = np.zeros((nCellsY, nCellsX))
    v = np.zeros((nCellsY, nCellsX))
    p = np.zeros((nCellsY, nCellsX))
    e = np.zeros((nCellsY, nCellsX))
    
    try:
        with open(file_path_1) as file:
            contents = file.readlines()
    except:
        try:
            with open(file_path_2) as file:
                contents = file.readlines()
        except:
            with open(file_path_3) as file:
                contents = file.readlines()

    for line in contents:
        data = [float(i) for i in line.strip().split(', ')]
        x, y, cur_rho, cur_vx, cur_vy, cur_p = data
        cur_e = cur_p / (gamma - 1) / cur_rho
        
        col = int(round((x - x0) / dx - 0.5))
        row = int(round((nCellsY - 1) - ((y - y0) / dy - 0.5)))

        rho[row, col] = cur_rho
        v[row, col] = math.sqrt(cur_vx**2 + cur_vy**2)
        p[row, col] = cur_p
        e[row, col] = cur_e
        
    # 数据映射
    data_list = [rho, v, p, e]

    # 定义实际坐标值
    x_ticks = np.linspace(x0, x1, 6).round(3)
    y_ticks = np.linspace(y0, y1, 6).round(3)
    y_ticks = np.flipud(y_ticks)

    data = data_list[var_id]

    # 定义真实的网格坐标
    X = np.linspace(x0, x1, nCellsX)
    Y = np.linspace(y0, y1, nCellsY)
    X, Y = np.meshgrid(X, Y)

    # ========== 1. 用 pcolormesh 画背景 ==========
    data = np.flipud(data)
    img = cur_ax.pcolormesh(X, Y, data, cmap='rocket', shading='auto')

    # ========== 2. 添加 contour 线 ==========
    cur_ax.contour(X, Y, data, levels=20, colors='black', linewidths=0.7)

    # ========== 3. 设置 colorbar ==========
    cbar = plt.colorbar(img, ax=cur_ax)
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()

    # ========== 4. 设置标签 ==========
    cur_ax.set_title(f"{label_list[var_id]}, {device}, T = {tStop:.1f}", size=15)
    cur_ax.set_xlabel('X', size=12)
    cur_ax.set_ylabel('Y', size=12)

    # 设置坐标轴标签
    xtick_positions = np.linspace(x0, x1, 6)
    ytick_positions = np.linspace(y0, y1, 6)

    cur_ax.set_xticks(xtick_positions)
    cur_ax.set_yticks(ytick_positions)

    cur_ax.set_xticklabels(x_ticks, rotation=0)
    cur_ax.set_yticklabels(y_ticks)


if __name__ == "__main__":

    for folder_path_index in range(len(folder_path_list)):
        folder_path = folder_path_list[folder_path_index]
        device = device_list[folder_path_index]
        cur_axes = axes[folder_path_index]  # upper cpu, lower gpu
        
        for i, ax in enumerate(cur_axes.flat):
            var_id = var_id_list[i]
            visualize_single(folder_path, device, ax, var_id, tStop)
        
    plt.savefig("D:/Study_Master/WrittenAssignment/Writing_CUDA/Figure%d_%s_T=%.1f.png" % (figure_id, case_name, tStop), 
        bbox_inches='tight', pad_inches=0.1)
    plt.subplots_adjust(left=0.05, right=0.95)
    # plt.tight_layout()
    # plt.show()

