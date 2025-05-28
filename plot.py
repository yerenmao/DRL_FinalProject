import os
import pandas as pd
import matplotlib.pyplot as plt

# Create output folder
os.makedirs("png", exist_ok=True)

# Load CSV
df = pd.read_csv("outputs/double/a2c_conn0_ep1.csv")

# Metrics to plot
metrics = {
    "system_total_stopped": "System Total Stopped",
    "system_total_waiting_time": "System Total Waiting Time",
    "system_mean_waiting_time": "System Mean Waiting Time",
    "system_mean_speed": "System Mean Speed"
}

# Plot each metric separately
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

# Plot all metrics together
plt.figure(figsize=(12, 8))
for col, title in metrics.items():
    plt.plot(df["step"], df[col], label=title)

plt.xlabel("Step")
plt.ylabel("Value")
plt.title("System Metrics Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("png/all_metrics.png")
plt.close()
