import pandas as pd
import matplotlib.pyplot as plt

# # Load data
# df = pd.read_csv('compare_methods.csv')
#
# # Filter only avg and rnd methods
# df_filtered = df[df['method'].isin(['average', 'rnd'])]
#
# # Compute average metrics
# metrics = ['recall@3', 'recall@5', 'ndcg@3', 'ndcg@5']
# avg_metrics = df_filtered.groupby('method')[metrics].mean().rename(index={'average': 'Our System', 'rnd': 'Random Selection'})
#
# # Transpose for plotting
# avg_metrics = avg_metrics.transpose()
#
# # Plot with increased figure height
# fig, ax = plt.subplots(figsize=(9, 7))  # Increased height from 6 to 7
# colors = ['#4C72B0', '#DD8452']
# avg_metrics.plot(kind='bar', color=colors, width=0.65, ax=ax)
#
# ax.set_title('Performance Comparison', fontsize=15, weight='bold')
# ax.set_ylabel('Score', fontsize=12)
# ax.set_xticklabels(avg_metrics.index, rotation=0, fontsize=11)
# ax.tick_params(axis='y', labelsize=11)
# ax.legend(title='Method', fontsize=11)
# ax.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout()
#
# # Save to file
# plt.savefig('method_comparison_taller.svg', format='svg')
# Load the two input files

file_path = "C:\\Users\\User\\Desktop\\University\\year4\\semA\\FinalProject\\evaluation\\"

# df1 = pd.read_csv(file_path+"compare_methods.csv", index_col="method")
# df2 = pd.read_csv(file_path+"compare_methods2.csv", index_col="method")
#
# # Define the number of responders for each file
# n1 = 18
# n2 = 11
#
# # Compute the weighted average
# combined_df = (df1 * n1 + df2 * n2) / (n1 + n2)
#
# # Save the result
# combined_path = "compare_methods_comb.csv"
# combined_df.to_csv(combined_path)

"""
Generate a grouped bar chart (SVG) comparing recommendation methods
on selected evaluation metrics.

Usage:
    python plot_metrics.py path/to/metrics.csv
"""



# --- configuration ----------------------------------------------------------
METRICS = ["recall@3", "recall@5", "recall@10",
           "ndcg@3",  "ndcg@5",  "ndcg@10"]

METHOD_ORDER = ["avg",  "max", "exc", "rnd"]  # desired legend / color order

LABELS = {
    "avg":    "Our system",
    "max":    "Max similarity",
    "exc":    "Excitement approach",
    "rnd": "Random selection",
}

COLORS = {
    "avg":   "#1f77b4",  # blue
    "rnd": "#ff7f0e", # orange
    "max":   "#8c564b",  # brown
    "exc":   "#9467bd",  # purple
}
OUTFILE = file_path+"metrics_comparison.svg"
BAR_WIDTH = 0.18
# ----------------------------------------------------------------------------


def load_and_reformat(csv_path: str) -> pd.DataFrame:
    """Load CSV and return DataFrame of shape (methods, metrics)."""
    df = pd.read_csv(csv_path, index_col=0)

    # Detect orientation and re-shape if needed
    if set(METRICS).issubset(df.columns):
        # rows = methods, cols = metrics
        df = df.loc[:, METRICS]
    else:
        # rows likely = metrics, so transpose
        df = df.set_index(df.columns[0]).loc[METRICS].T

    # Keep only configured methods in desired order
    df = df.reindex(METHOD_ORDER).dropna(how="all")
    return df


def plot_bars(df: pd.DataFrame) -> None:
    """Create and save grouped bar chart as SVG."""
    x = range(len(METRICS))
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(df.index):
        ax.bar(
            [p + i * BAR_WIDTH for p in x],
            df.loc[method, METRICS],
            width=BAR_WIDTH,
            label=LABELS.get(method, method),
            color=COLORS.get(method, "gray"),
        )

    # Aesthetics
    ax.set_xticks([p + BAR_WIDTH * (len(df.index) - 1) / 2 for p in x])
    ax.set_xticklabels(METRICS)
    ax.set_ylabel("Score")
    # ax.set_title("Recommendation Performance Comparison")
    ax.legend(title="Method")
    plt.tight_layout()
    plt.savefig(OUTFILE, format="svg")
    plt.close()


if __name__ == "__main__":

    csv_path = file_path+"compare_methods_comb.csv"
    data = load_and_reformat(csv_path)
    plot_bars(data)
    print(f"SVG saved to '{OUTFILE}'")
