"""Plotly chart builders.

Colour choices follow a validated, colour-blind-safe palette: categorical hues are
assigned in a fixed order (never cycled; a ninth category folds into "Other"), magnitude
uses a single blue ramp, and text never wears a series colour.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

from ..models.base import Ratings

PALETTE = {
    "surface": "rgba(0,0,0,0)",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "other": "#898781",
    "good": "#0ca30c",
    "critical": "#d03b3b",
}
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
SEQUENTIAL: tuple[str, ...] = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
MAX_CATEGORIES = len(CATEGORICAL)
OTHER = "Other"

_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def category_colors(categories: Sequence[str], *, counts: Mapping[str, int] | None = None) -> dict[str, str]:
    """Stable category → colour map. The eight most frequent keep a hue; the rest fold to Other."""
    uniq = list(dict.fromkeys(c for c in categories if c))
    if counts:
        uniq.sort(key=lambda c: (-counts.get(c, 0), c))
    else:
        uniq.sort()
    out = {c: CATEGORICAL[i] for i, c in enumerate(uniq[:MAX_CATEGORIES])}
    for c in uniq[MAX_CATEGORIES:]:
        out[c] = PALETTE["other"]
    return out


def _layout(fig: go.Figure, *, title: str | None = None, height: int | None = None, legend: bool = True) -> go.Figure:
    top = 24
    if title:
        top += 32
    if legend:
        top += 48  # room for a wrapped horizontal legend between title and plot
    fig.update_layout(
        title={"text": title, "y": 0.985, "yanchor": "top", "font": {"size": 16, "color": PALETTE["ink"]}}
        if title
        else None,
        template="plotly_white",
        font={"family": _FONT, "color": PALETTE["ink_secondary"], "size": 13},
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        margin={"l": 16, "r": 16, "t": top, "b": 32},
        height=height,
        hoverlabel={"font": {"family": _FONT}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0} if legend else None,
        showlegend=legend,
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["axis"], linecolor=PALETTE["axis"])
    fig.update_yaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["axis"], linecolor=PALETTE["axis"])
    return fig


def rating_bar(
    table: pd.DataFrame,
    *,
    top: int = 25,
    color_col: str = "conference",
    value_col: str = "rating",
    title: str | None = None,
) -> go.Figure:
    """Horizontal ranked bar chart of the top ``top`` teams, coloured by conference."""
    df = table.sort_values(value_col, ascending=False).head(top).copy()
    counts = table[color_col].value_counts().to_dict() if color_col in table else {}
    colors = category_colors(df[color_col].tolist(), counts=counts) if color_col in df else {}
    df["label"] = [f"{r}. {t}" for r, t in zip(df["rank"], df["team"], strict=True)]
    fig = go.Figure()
    seen: set[str] = set()
    for _, row in df.iloc[::-1].iterrows():
        cat = row.get(color_col, "") if color_col in df else ""
        color = colors.get(cat, CATEGORICAL[0])
        name = cat if colors.get(cat) != PALETTE["other"] else OTHER
        fig.add_bar(
            x=[row[value_col]],
            y=[row["label"]],
            orientation="h",
            name=name,
            marker={"color": color, "cornerradius": 4},
            showlegend=name not in seen,
            legendgroup=name,
            customdata=[[row.get("record", ""), row.get("sos", float("nan")), cat]],
            hovertemplate=(
                "<b>%{y}</b><br>Rating %{x:.3f}<br>Record %{customdata[0]}"
                "<br>SOS %{customdata[1]:.3f}<br>%{customdata[2]}<extra></extra>"
            ),
        )
        seen.add(name)
    fig.update_layout(barmode="overlay", bargap=0.25)
    fig.update_yaxes(type="category", tickfont={"size": 12})
    fig.update_xaxes(title_text=value_col.replace("_", " ").title())
    return _layout(fig, title=title, height=max(420, 24 * len(df) + 120))


def rating_sos_scatter(table: pd.DataFrame, *, label_top: int = 12, title: str | None = None) -> go.Figure:
    """Rating vs strength of schedule; the top teams are labelled directly."""
    df = table.copy()
    labels = [t if r <= label_top else "" for t, r in zip(df["team"], df["rank"], strict=True)]
    fig = go.Figure(
        go.Scatter(
            x=df["sos"],
            y=df["rating"],
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont={"color": PALETTE["ink_secondary"], "size": 11},
            marker={"size": 9, "color": CATEGORICAL[0], "line": {"width": 2, "color": "white"}},
            customdata=list(zip(df["team"], df["record"], df["rank"], strict=True)),
            hovertemplate="<b>%{customdata[0]}</b> (#%{customdata[2]}, %{customdata[1]})"
            "<br>Rating %{y:.3f}<br>SOS %{x:.3f}<extra></extra>",
            name="Teams",
        )
    )
    fig.update_xaxes(title_text="Strength of schedule (mean opponent rating)")
    fig.update_yaxes(title_text="Rating")
    return _layout(fig, title=title, height=520, legend=False)


def trajectory_chart(
    snapshots: Mapping[int, Ratings],
    teams: Sequence[str],
    *,
    metric: str = "rank",
    title: str | None = None,
) -> go.Figure:
    """Weekly rank (or rating) lines for up to eight teams."""
    teams = list(teams)[:MAX_CATEGORIES]
    colors = {t: CATEGORICAL[i] for i, t in enumerate(teams)}
    weeks = sorted(snapshots)
    fig = go.Figure()
    for t in teams:
        xs, ys = [], []
        for w in weeks:
            ratings = snapshots[w]
            if t not in ratings:
                continue
            if metric == "rank":
                ordered = sorted(ratings.items(), key=lambda kv: (-kv[1], kv[0]))
                ys.append(next(i + 1 for i, (name, _) in enumerate(ordered) if name == t))
            else:
                ys.append(ratings[t])
            xs.append(w)
        fig.add_scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name=t,
            line={"color": colors[t], "width": 2},
            marker={"size": 8, "line": {"width": 2, "color": "white"}},
            hovertemplate=f"<b>{t}</b><br>Week %{{x}}<br>{metric.title()} %{{y}}<extra></extra>",
        )
    fig.update_xaxes(title_text="Week", dtick=1)
    if metric == "rank":
        fig.update_yaxes(title_text="Rank", autorange="reversed", dtick=5)
    else:
        fig.update_yaxes(title_text="Rating")
    fig.update_layout(hovermode="x unified")
    return _layout(fig, title=title, height=460)


def weekly_metric_chart(
    by_week: Mapping[str, pd.DataFrame],
    *,
    metric: str = "accuracy",
    title: str | None = None,
) -> go.Figure:
    """One line per method for a backtest metric across evaluation weeks."""
    fig = go.Figure()
    for i, (method, df) in enumerate(list(by_week.items())[:MAX_CATEGORIES]):
        if metric not in df:
            continue
        fig.add_scatter(
            x=df["week"],
            y=df[metric],
            mode="lines+markers",
            name=method,
            line={"color": CATEGORICAL[i], "width": 2},
            marker={"size": 8, "line": {"width": 2, "color": "white"}},
            hovertemplate=f"<b>{method}</b><br>Week %{{x}}<br>{metric} %{{y:.3f}}<extra></extra>",
        )
    fig.update_xaxes(title_text="Week", dtick=1)
    fig.update_yaxes(title_text=metric.replace("_", " ").title())
    fig.update_layout(hovermode="x unified")
    return _layout(fig, title=title, height=420)


def calibration_chart(tables: Mapping[str, pd.DataFrame], *, title: str | None = None) -> go.Figure:
    """Reliability diagram: observed home-win rate against predicted probability."""
    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect calibration",
        line={"color": PALETTE["axis"], "dash": "dot", "width": 1.5},
        hoverinfo="skip",
    )
    for i, (method, df) in enumerate(list(tables.items())[:MAX_CATEGORIES]):
        d = df.dropna(subset=["observed"])
        fig.add_scatter(
            x=d["predicted"],
            y=d["observed"],
            mode="lines+markers",
            name=method,
            line={"color": CATEGORICAL[i], "width": 2},
            marker={"size": [max(6, min(18, 4 + n**0.5)) for n in d["n"]], "line": {"width": 2, "color": "white"}},
            customdata=d["n"],
            hovertemplate=f"<b>{method}</b><br>Predicted %{{x:.2f}}<br>Observed %{{y:.2f}}<br>n=%{{customdata}}<extra></extra>",
        )
    fig.update_xaxes(title_text="Predicted home win probability", range=[0, 1])
    fig.update_yaxes(title_text="Observed home win rate", range=[0, 1])
    return _layout(fig, title=title, height=460)


def conference_bar(conf_table: pd.DataFrame, *, title: str | None = None) -> go.Figure:
    df = conf_table.sort_values("strength_index", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=df["strength_index"],
            y=df["conference"],
            orientation="h",
            marker={"color": CATEGORICAL[0], "cornerradius": 4},
            customdata=list(zip(df["teams"], df["top_team"], df["mean_rating"], strict=True)),
            hovertemplate="<b>%{y}</b><br>Strength index %{x:.2f}<br>%{customdata[0]} teams"
            "<br>Best: %{customdata[1]}<br>Mean rating %{customdata[2]:.3f}<extra></extra>",
            name="Strength index",
        )
    )
    fig.update_xaxes(title_text="Strength index (0.6 to 1.6)")
    fig.update_yaxes(type="category")
    return _layout(fig, title=title, height=max(320, 32 * len(df) + 100), legend=False)
