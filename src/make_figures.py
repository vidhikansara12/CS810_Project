import os, argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_csv", type=str, default="/content/drive/MyDrive/CS810/results/metrics/metrics.csv")
    p.add_argument("--out_dir", type=str, default="/content/drive/MyDrive/CS810/results/figures")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.metrics_csv)

    # Create a nice label for plotting
    df["label"] = df["model"] + " (" + df["setting"] + ")"

    # --- Plot 1: LeNet intended-label accuracy ---
    plt.figure(figsize=(8,4))
    plt.bar(df["label"], df["lenet_acc_intended"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("LeNet accuracy on intended label (↑)")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    out1 = os.path.join(args.out_dir, "fig_lenet_acc.png")
    plt.savefig(out1, dpi=200)
    plt.close()

    # --- Plot 2: Feature Fréchet (lower is better) ---
    plt.figure(figsize=(8,4))
    plt.bar(df["label"], df["feature_frechet"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Feature Fréchet distance (↓)")
    plt.tight_layout()
    out2 = os.path.join(args.out_dir, "fig_feature_frechet.png")
    plt.savefig(out2, dpi=200)
    plt.close()

    # --- Table for report (CSV + simple LaTeX) ---
    table_cols = ["model", "setting", "lenet_acc_intended", "feature_frechet"]
    df_out = df[table_cols].copy()
    out_csv = os.path.join(os.path.dirname(args.metrics_csv), "table_metrics.csv")
    df_out.to_csv(out_csv, index=False)

    out_tex = os.path.join(os.path.dirname(args.metrics_csv), "table_metrics.tex")
    with open(out_tex, "w") as f:
        f.write(df_out.to_latex(index=False, float_format="%.4f"))

    print("Wrote figures:")
    print(" -", out1)
    print(" -", out2)
    print("Wrote tables:")
    print(" -", out_csv)
    print(" -", out_tex)

if __name__ == "__main__":
    main()