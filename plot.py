import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# 解析指令列參數
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", required=True, help="Input File Path")
parser.add_argument("-o", "--output_dir", required=True, help="Output File Path")
args = parser.parse_args()

input_file = args.input_file

# 建立輸出資料夾
os.makedirs("png", exist_ok=True)

# 讀取 CSV
df = pd.read_csv(input_file)

# 指標名稱對應
metrics = {
    "system_total_stopped": "System Total Stopped",
    "system_total_waiting_time": "System Total Waiting Time",
    "system_mean_waiting_time": "System Mean Waiting Time",
    "system_mean_speed": "System Mean Speed"
}

# 單一指標繪圖
for col, title in metrics.items():
    plt.figure()
    plt.plot(df["step"], df[col], marker="o")
    plt.xlabel("Step")
    plt.ylabel(title)
    plt.title(f"{title} Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"png/{col}.png")
    plt.close()

# 多指標總覽圖
plt.figure(figsize=(12, 8))
for col, title in metrics.items():
    plt.plot(df["step"], df[col], label=title)

plt.xlabel("Step")
plt.ylabel("Value")
plt.title("System Metrics Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "all_metrics.png"))
plt.close()
