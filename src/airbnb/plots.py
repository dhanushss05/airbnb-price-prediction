"""Figures for the report and slides (saved as PNG under results/figures)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

# Fixed categorical order: a city always keeps its colour.
CITY_COLORS = {"nyc": "#2a78d6", "paris": "#eb6834", "berlin": "#1baf7a"}
CITY_LABEL = {k: v["name"] for k, v in C.CITIES.items()}
BAR = "#2a78d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def _save(fig, name):
    fig.savefig(C.FIG_DIR / name)
    plt.close(fig)


def price_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    bins = np.linspace(0, 1000, 60)
    for city in C.CITIES:
        p = df.loc[df.city == city, "price"]
        axes[0].hist(p, bins=bins, histtype="step", lw=2, color=CITY_COLORS[city],
                     label=f"{CITY_LABEL[city]} (median ${p.median():.0f})", density=True)
        axes[1].hist(np.log(p), bins=50, histtype="step", lw=2, color=CITY_COLORS[city], density=True)
    axes[0].set(title="Nightly price (USD)", xlabel="price", ylabel="density")
    axes[1].set(title="log(price) is close to normal", xlabel="log(price)")
    axes[0].legend(fontsize=8)
    _save(fig, "price_distribution.png")


def model_comparison(e1: pd.DataFrame):
    cities = list(C.CITIES)
    models = list(dict.fromkeys(e1["model"]))
    fig, axes = plt.subplots(1, len(cities), figsize=(10, 3.2), sharey=True)
    for ax, city in zip(axes, cities):
        d = e1[e1.train_city == city].set_index("model").loc[models]
        y = np.arange(len(models))
        ax.barh(y, d["R2_log"], color=CITY_COLORS[city], height=0.6)
        for yi, v in zip(y, d["R2_log"]):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=8, color=INK2)
        ax.set_yticks(y, models)
        ax.invert_yaxis()
        ax.set(xlim=(0, 1), title=CITY_LABEL[city], xlabel="test R² (log price)")
        ax.grid(axis="y", visible=False)
    _save(fig, "model_comparison.png")


def pred_vs_actual(preds: dict):
    fig, axes = plt.subplots(1, len(preds), figsize=(10, 3.3), sharex=True, sharey=True)
    for ax, (city, (yt, yp)) in zip(axes, preds.items()):
        ax.scatter(np.exp(yt), np.exp(yp), s=3, alpha=0.25, color=CITY_COLORS[city], lw=0)
        ax.plot([10, 1000], [10, 1000], color=INK2, lw=1, ls="--")
        ax.set(xscale="log", yscale="log", title=f"{CITY_LABEL[city]} (XGBoost)",
               xlabel="actual price ($)")
    axes[0].set_ylabel("predicted price ($)")
    _save(fig, "pred_vs_actual.png")


def _pretty(name):
    name = name.split("__", 1)[-1]
    return name.replace("_", " ").replace("missingindicator ", "missing: ")


def feature_importance(pipe, city, top=15):
    names = pipe.named_steps["features"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    s = pd.Series(imp, index=[_pretty(n) for n in names]).sort_values(ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(s.index[::-1], s.values[::-1], color=BAR, height=0.6)
    ax.set(title=f"Top {top} features, XGBoost ({CITY_LABEL[city]})", xlabel="importance (gain share)")
    ax.grid(axis="y", visible=False)
    s.to_csv(C.RESULTS_DIR / f"feature_importance_{city}.csv", header=["importance"])
    _save(fig, "feature_importance.png")


def ablation(e2: pd.DataFrame):
    order = ["C", "O", "T", "C+O", "C+O+T"]
    labels = {"C": "Continuous", "O": "Categorical\n+amenities", "T": "Text", "C+O": "C + O",
              "C+O+T": "All (C+O+T)"}
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(order))
    w = 0.36
    for i, city in enumerate(["nyc", "paris"]):
        d = e2[e2.train_city == city].set_index("features").loc[order, "R2_log"]
        ax.bar(x + (i - 0.5) * w, d.values, w * 0.92, color=CITY_COLORS[city], label=CITY_LABEL[city])
        for xi, v in zip(x + (i - 0.5) * w, d.values):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [labels[o] for o in order])
    ax.set(ylim=(0, 1), ylabel="test R² (log price)")
    ax.set_title("Feature-group ablation (XGBoost)", pad=22)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    _save(fig, "feature_ablation.png")


def transfer(e34: pd.DataFrame):
    order = ["nyc", "paris", "nyc+paris", "berlin"]
    labels = {"nyc": "NYC only", "paris": "Paris only", "nyc+paris": "NYC + Paris",
              "berlin": "Berlin only\n(in-city reference)"}
    d = e34[e34.test_city == "berlin"].set_index("train_on").loc[order]
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#9ab8e0"] * 3 + [CITY_COLORS["berlin"]]
    ax.bar(range(4), d["R2_log"], color=colors, width=0.55)
    for i, v in enumerate(d["R2_log"]):
        ax.text(i, max(v, 0) + 0.015, f"{v:.3f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(4), [labels[o] for o in order])
    ax.set(ylim=(0, 1), ylabel="R² on Berlin test set (log price)",
           title="Predicting Berlin prices from other cities")
    ax.grid(axis="x", visible=False)
    _save(fig, "transfer_berlin.png")
