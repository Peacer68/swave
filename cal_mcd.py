import sys
from tqdm import tqdm
import subprocess
import os, json

for id in [2,10]:

    # root_dir = f"./test_result_{id}_100"
    root_dir = f"./distill_res/{id}-yes"



    tol_mcd = 0
    num = 100
    mcd_arr = []
    for i in tqdm(range(num)):
        command = f'mcd-cli {root_dir}/audio_{i}_step{id}.wav ./final_res_step2/audio_{i}_gt.wav'
        result = subprocess.check_output(command, shell=True, text=True)

        # 解析结果，提取 MCD 值
        mcd = result.split(":")[1].split("\n")[0]  # 查找 MCD 值所在位置
        mcd_value = float(mcd)  # 提取 MCD 值
        tol_mcd += mcd_value
        mcd_arr.append(mcd_value)

    avg_mcd = tol_mcd/num

    with open(f"{root_dir}/statistics.json", "r") as f:
        stats = json.load(f)
    stats["average_mcd"] = round(avg_mcd, 2)
    stats["mcd_arr"] = mcd_arr
    print(root_dir, avg_mcd)

    with open(f"{root_dir}/statistics.json", "w") as f:
        json.dump(stats, f)