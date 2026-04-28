import json
import matplotlib.pyplot as plt
import numpy as np

def load_json(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def normalize(arr):
    arr = np.array(arr)
    return (arr-arr.min())/(arr.max()-arr.min())

ipo = load_json("ipo_rank8/metrics.jsonl")

steps = [i["step"] for i in ipo]
ipo_loss = [i["ipo_loss"] for i in ipo]
accuracy = [i["accuracy"] for i in ipo]
margin   = [i["margin"] for i in ipo]

ipo_loss_norm = normalize(ipo_loss)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("IPO Training Metrics")

axes[0, 0].plot(steps, ipo_loss_norm)
axes[0, 0].set_title("IPO Loss (Normalized)")
axes[0, 0].set_xlabel("Step")

axes[0, 1].axis("off")

axes[1, 0].plot(steps, accuracy)
axes[1, 0].set_title("Accuracy")
axes[1, 0].set_xlabel("Step")

axes[1, 1].plot(steps, margin)
axes[1, 1].set_title("Margin")
axes[1, 1].set_xlabel("Step")

plt.tight_layout()
plt.savefig("ipo.png")
plt.show()

dpo = load_json("dpo_metrics.jsonl")

steps = [i["step"] for i in dpo]
dpo_loss = [i["dpo_loss"] for i in dpo]
accuracy = [i["accuracy"] for i in dpo]
margin   = [i["margin"] for i in dpo]

dpo_loss_norm = normalize(dpo_loss)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DPO Training Metrics")

axes[0, 0].plot(steps, dpo_loss_norm)
axes[0, 0].set_title("DPO Loss (Normalized)")
axes[0, 0].set_xlabel("Step")

axes[0, 1].axis("off")

axes[1, 0].plot(steps, accuracy)
axes[1, 0].set_title("Accuracy")
axes[1, 0].set_xlabel("Step")

axes[1, 1].plot(steps, margin)
axes[1, 1].set_title("Margin")
axes[1, 1].set_xlabel("Step")

plt.tight_layout()
plt.savefig("dpo.png")
plt.show()