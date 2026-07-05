import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_result(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = data[0]
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_json", type=str, required=True,
                        help="Path to result JSON file")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory to save roc_curve.png and pr_curve.png")
    args = parser.parse_args()

    d = load_result(args.result_json)

    out_dir = args.out_dir or os.path.dirname(args.result_json)
    os.makedirs(out_dir, exist_ok=True)

    name = d.get("name", "MIA_Result")

    # ROC
    fpr = np.array(d["metrics"]["fpr"])
    tpr = np.array(d["metrics"]["tpr"])
    roc_auc = d["metrics"]["roc_auc"]

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=200)
    plt.close()

    # PR
    recall = np.array(d["pr_metrics"]["recall"])
    precision = np.array(d["pr_metrics"]["precision"])
    pr_auc = d["pr_metrics"]["pr_auc"]

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=200)
    plt.close()

    print("Saved:", os.path.join(out_dir, "roc_curve.png"))
    print("Saved:", os.path.join(out_dir, "pr_curve.png"))


if __name__ == "__main__":
    main()