import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGO_ORDER = ["GA", "RL-GA"]
ALGO_LABELS = {
    "GA": "GA",
    "RL-GA": "RL-GA",
}
ALGO_COLORS = {
    "GA": "#66c2a5",
    "RL-GA": "#fc8d62",
}


def load_results_data(excel_path: str, sheet_name: str = "Results") -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Find header row automatically
    for i in range(10):
        temp = pd.read_excel(excel_path, sheet_name=sheet_name, header=i)
        if "Problem" in temp.columns and "Algorithm" in temp.columns:
            df = temp
            break

    # Clean
    df = df.dropna(subset=["Problem", "Algorithm"]).copy()
    df["Problem"] = df["Problem"].astype(int)

    return df


def grouped_bar_plot(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    error_col: str | None = None,
    figsize=(10, 5),
):
    problems = sorted(df["Problem"].unique())
    x = np.arange(len(problems))
    width = 0.36

    plt.figure(figsize=figsize)

    for i, algo in enumerate(ALGO_ORDER):
        algo_df = (
            df[df["Algorithm"] == algo]
            .sort_values("Problem")
            .set_index("Problem")
            .reindex(problems)
            .reset_index()
        )

        offset = (i - 0.5) * width
        y = algo_df[value_col].to_numpy(dtype=float)

        yerr = None
        if error_col is not None and error_col in algo_df.columns:
            yerr = algo_df[error_col].to_numpy(dtype=float)

        plt.bar(
            x + offset,
            y,
            width=width,
            label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
            alpha=0.9,
            yerr=yerr,
            capsize=3 if yerr is not None else 0,
            error_kw={"elinewidth": 1, "capthick": 1},
        )

    plt.xticks(x, [str(p) for p in problems])
    plt.xlabel("Problem")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def improvement_plot(
    df: pd.DataFrame,
    ga_col: str,
    rl_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    figsize=(10, 5),
):
    problems = sorted(df["Problem"].unique())

    ga_df = (
        df[df["Algorithm"] == "GA"]
        .sort_values("Problem")
        .set_index("Problem")
        .reindex(problems)
        .reset_index()
    )
    rl_df = (
        df[df["Algorithm"] == "RL-GA"]
        .sort_values("Problem")
        .set_index("Problem")
        .reindex(problems)
        .reset_index()
    )

    improvement_pct = ((ga_df[ga_col] - rl_df[rl_col]) / ga_df[ga_col]) * 100.0

    x = np.arange(len(problems))
    plt.figure(figsize=figsize)
    bars = plt.bar(x, improvement_pct.to_numpy(dtype=float), alpha=0.9)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, [str(p) for p in problems])
    plt.xlabel("Problem")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, improvement_pct):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot GA vs RL-GA comparison charts from results_500_300_8_summary.xlsx."
    )
    parser.add_argument(
        "--excel",
        default="results_500_300_8_summary.xlsx",
        help="Path to the Excel file.",
    )
    parser.add_argument(
        "--sheet",
        default="Results",
        help="Sheet name containing the summary results.",
    )
    parser.add_argument(
        "--outdir",
        default="results_500_300_8_plots",
        help="Output directory for saved plots.",
    )
    args = parser.parse_args()

    excel_path = Path(args.excel)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_results_data(excel_path, args.sheet)

    grouped_bar_plot(
        df=df,
        value_col="BestFitness_Avg",
        error_col="BestFitness_Std",
        title="Average Best Fitness by Problem (GA vs RL-GA)",
        ylabel="Average Best Fitness (Lower is Better)",
        output_path=outdir / "best_fitness_avg.png",
    )

    grouped_bar_plot(
        df=df,
        value_col="BestFitness_Min",
        title="Minimum Best Fitness by Problem (GA vs RL-GA)",
        ylabel="Minimum Best Fitness (Lower is Better)",
        output_path=outdir / "best_fitness_min.png",
    )

    grouped_bar_plot(
        df=df,
        value_col="ExecutionTime_Avg_s",
        error_col="ExecutionTime_Std_s",
        title="Average Execution Time by Problem (GA vs RL-GA)",
        ylabel="Execution Time (s)",
        output_path=outdir / "execution_time_avg.png",
    )

    grouped_bar_plot(
        df=df,
        value_col="TotalDistance_Avg",
        error_col="TotalDistance_Std",
        title="Average Total Distance by Problem (GA vs RL-GA)",
        ylabel="Average Total Distance (Lower is Better)",
        output_path=outdir / "total_distance_avg.png",
    )

    improvement_plot(
        df=df,
        ga_col="BestFitness_Avg",
        rl_col="BestFitness_Avg",
        title="RL-GA Improvement over GA by Problem (Average Best Fitness)",
        ylabel="Improvement (%)",
        output_path=outdir / "best_fitness_avg_improvement_pct.png",
    )

    improvement_plot(
        df=df,
        ga_col="BestFitness_Min",
        rl_col="BestFitness_Min",
        title="RL-GA Improvement over GA by Problem (Minimum Best Fitness)",
        ylabel="Improvement (%)",
        output_path=outdir / "best_fitness_min_improvement_pct.png",
    )

    print(f"Saved plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
