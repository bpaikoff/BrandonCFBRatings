import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def plot_top_n(ratings: Dict[str, float], records: Dict[str, Tuple[int, int]], n: int = 25, title: str = ""):
    sorted_items = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)[:n]
    teams = [t for t, _ in sorted_items]
    values = [v for _, v in sorted_items]
    recs = [f"{records[t][0]}-{records[t][1]}" for t in teams]
    ypos = np.arange(n)
    plt.figure(figsize=(12, 10))
    plt.barh(ypos[::-1], values[::-1], color="#0A3161", height=0.8)
    plt.yticks(ypos[::-1], [f"{i+1}. {team} ({rec})" for i, (team, rec) in enumerate(zip(teams[::-1], recs[::-1]))])
    plt.xlabel("Rating")
    plt.title(title or "Top Teams")
    # Scale x-limits based on data spread
    value_range = max(values) - min(values)
    if value_range > 1e-8:
        xmin = min(values) - 0.05 * value_range
        xmax = max(values) + 0.05 * value_range
    else:
        # All values are identical, use a small default range
        avg_value = (max(values) + min(values)) / 2
        xmin = avg_value - 0.1
        xmax = avg_value + 0.1
    plt.xlim(xmin, xmax)
    plt.grid(axis="x", alpha=0.3)
    for i, rating in enumerate(values[::-1]):
        plt.text(rating + 0.01, i, f"{rating:.4f}", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.show()