"""Professional financial visualizations."""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Consistent style
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "primary": "#1f4e79",
    "secondary": "#5b9bd5",
    "accent": "#ed7d31",
    "green": "#548235",
    "red": "#c00000",
    "gray": "#7f7f7f",
}


def _save(fig, path: str | None) -> None:
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_revenue_forecast(
    income_statement: pd.DataFrame,
    save_path: str | None = None,
    title: str = "Revenue Forecast",
) -> plt.Figure:
    rev = income_statement.loc["Revenue"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(rev.index.astype(str), rev.values, color=COLORS["primary"], edgecolor="white")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Revenue")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    for i, v in enumerate(rev.values):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
        return fig
    return fig


def plot_ebitda_margin(
    income_statement: pd.DataFrame,
    save_path: str | None = None,
    title: str = "EBITDA & Margin",
) -> plt.Figure:
    rev = income_statement.loc["Revenue"]
    ebitda = income_statement.loc["EBITDA"]
    margin = ebitda / rev
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(rev.index.astype(str), ebitda.values, color=COLORS["secondary"], label="EBITDA")
    ax1.set_ylabel("EBITDA")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax2 = ax1.twinx()
    ax2.plot(rev.index.astype(str), margin.values, color=COLORS["accent"], marker="o", linewidth=2, label="Margin")
    ax2.set_ylabel("EBITDA Margin")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
        return fig
    return fig


def plot_fcff_forecast(
    dcf_dataframe: pd.DataFrame,
    save_path: str | None = None,
    title: str = "FCFF Forecast",
) -> plt.Figure:
    fcff = dcf_dataframe.loc["FCFF"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(fcff.index.astype(str), fcff.values, color=COLORS["green"], edgecolor="white")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("FCFF")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
        return fig
    return fig


def plot_sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    save_path: str | None = None,
    title: str = "Sensitivity: Implied Share Price",
    cmap: str = "RdYlGn",
    base_wacc: str | None = None,
    base_growth: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(sensitivity_df.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(sensitivity_df.columns)))
    ax.set_xticklabels(sensitivity_df.columns, fontsize=8)
    ax.set_yticks(range(len(sensitivity_df.index)))
    ax.set_yticklabels(sensitivity_df.index, fontsize=8)
    ax.set_xlabel(sensitivity_df.columns.name or "Terminal Growth / Multiple")
    ax.set_ylabel(sensitivity_df.index.name or "WACC")
    ax.set_title(title, fontsize=12, fontweight="bold")
    # Annotate cells
    for i in range(len(sensitivity_df.index)):
        for j in range(len(sensitivity_df.columns)):
            val = sensitivity_df.values[i, j]
            if np.isnan(val):
                txt = "n/a"
            else:
                txt = f"{val:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="black", fontweight="bold" if (
                        (base_wacc is not None and sensitivity_df.index[i] == base_wacc) and
                        (base_growth is not None and sensitivity_df.columns[j] == base_growth)
                    ) else "normal")
    fig.colorbar(im, ax=ax, label="Implied Price")
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
        return fig
    return fig


def plot_cumulative_returns(
    prices: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    save_path: str | None = None,
    title: str = "Cumulative Returns",
) -> plt.Figure:
    if returns is not None:
        cum = (1 + returns).cumprod() - 1
    elif prices is not None:
        rets = prices.pct_change().dropna()
        cum = (1 + rets).cumprod() - 1
    else:
        raise ValueError("Provide prices or returns")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col in cum.columns:
        ax.plot(cum.index, cum[col], label=col, linewidth=1.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(frameon=True, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
        return fig
    return fig


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    min_var: dict | None = None,
    max_sharpe: dict | None = None,
    mc_df: pd.DataFrame | None = None,
    save_path: str | None = None,
    title: str = "Efficient Frontier",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    if mc_df is not None:
        ax.scatter(mc_df["volatility"], mc_df["return"], c=mc_df["sharpe"], cmap="viridis", alpha=0.4, s=10, label="Random portfolios")
    ax.plot(frontier_df["volatility"], frontier_df["expected_return"], color=COLORS["primary"], linewidth=2, label="Efficient frontier")
    if min_var is not None:
        ax.scatter([min_var["volatility"]], [min_var["expected_return"]], color=COLORS["red"], s=120, marker="*", label="Min Variance", edgecolors="black")
    if max_sharpe is not None:
        ax.scatter([max_sharpe["volatility"]], [max_sharpe["expected_return"]], color=COLORS["accent"], s=120, marker="*", label="Max Sharpe", edgecolors="black")
    ax.set_xlabel("Volatility (annualized)")
    ax.set_ylabel("Expected Return (annualized)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
        return fig
    return fig


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    save_path: str | None = None,
    title: str = "Correlation Matrix",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.values[i,j]) > 0.5 else "black")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
        return fig
    return fig


def plot_drawdown(
    prices: pd.Series | None = None,
    returns: pd.Series | None = None,
    label: str | None = None,
    save_path: str | None = None,
    title: str = "Drawdown",
) -> plt.Figure:
    if prices is not None:
        cum_max = prices.cummax()
        dd = (prices - cum_max) / cum_max
        idx = prices.index
    elif returns is not None:
        cum = (1 + returns).cumprod()
        cum_max = cum.cummax()
        dd = (cum - cum_max) / cum_max
        idx = returns.index
    else:
        raise ValueError("Provide prices or returns")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.fill_between(idx, dd.values, 0, color=COLORS["red"], alpha=0.4)
    ax.plot(idx, dd.values, color=COLORS["red"], linewidth=1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
        return fig
    return fig


def plot_risk_return_scatter(
    expected_returns: pd.Series,
    volatilities: pd.Series,
    save_path: str | None = None,
    title: str = "Risk-Return Profile",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(volatilities.values, expected_returns.values, color=COLORS["primary"], s=80, alpha=0.7, edgecolors="white")
    for i, label in enumerate(expected_returns.index):
        ax.annotate(label, (volatilities.values[i], expected_returns.values[i]), fontsize=8, xytext=(5,5), textcoords="offset points")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Expected Return")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
        return fig
    return fig
