"""Plotly chart builders.

Colour choices follow a validated, colour-blind-safe palette: categorical hues are
assigned in a fixed order (never cycled; a ninth category folds into "Other"), magnitude
uses a single blue ramp, and text never wears a series colour. Every builder takes
``theme="light" | "dark"`` so ink, grid and series steps are chosen for the surface the
chart is actually drawn on rather than flipped automatically.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

from ..models.base import Ratings

CATEGORICAL_LIGHT: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
CATEGORICAL_DARK: tuple[str, ...] = (
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
)
CATEGORICAL = CATEGORICAL_LIGHT
SEQUENTIAL: tuple[str, ...] = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
MAX_CATEGORIES = len(CATEGORICAL_LIGHT)
OTHER = "Other"

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "other": "#898781",
        "ring": "#fcfcfb",
    },
    "dark": {
        "ink": "#f0efec",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "other": "#898781",
        "ring": "#1a1a19",
    },
}
PALETTE = {**THEMES["light"], "surface": "rgba(0,0,0,0)", "good": "#0ca30c", "critical": "#d03b3b"}

_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _tokens(theme: str) -> dict[str, str]:
    return THEMES["dark" if theme == "dark" else "light"]


def series_colors(theme: str = "light") -> tuple[str, ...]:
    return CATEGORICAL_DARK if theme == "dark" else CATEGORICAL_LIGHT


def category_colors(
    categories: Sequence[str], *, counts: Mapping[str, int] | None = None, theme: str = "light"
) -> dict[str, str]:
    """Stable category → colour map. The eight most frequent keep a hue; the rest fold to Other."""
    uniq = list(dict.fromkeys(c for c in categories if c))
    if counts:
        uniq.sort(key=lambda c: (-counts.get(c, 0), c))
    else:
        uniq.sort()
    colors = series_colors(theme)
    out = {c: colors[i] for i, c in enumerate(uniq[:MAX_CATEGORIES])}
    for c in uniq[MAX_CATEGORIES:]:
        out[c] = _tokens(theme)["other"]
    return out


def _layout(
    fig: go.Figure, *, title: str | None = None, height: int | None = None, legend: bool = True, theme: str = "light"
) -> go.Figure:
    t = _tokens(theme)
    top = 24
    if title:
        top += 32
    if legend:
        top += 48  # room for a wrapped horizontal legend between title and plot
    fig.update_layout(
        title={"text": title, "y": 0.985, "yanchor": "top", "font": {"size": 16, "color": t["ink"]}} if title else None,
        template="plotly_dark" if theme == "dark" else "plotly_white",
        font={"family": _FONT, "color": t["ink_secondary"], "size": 13},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 16, "r": 16, "t": top, "b": 32},
        height=height,
        hoverlabel={"font": {"family": _FONT}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0} if legend else None,
        showlegend=legend,
    )
    fig.update_xaxes(gridcolor=t["grid"], zerolinecolor=t["axis"], linecolor=t["axis"], tickfont={"color": t["muted"]})
    fig.update_yaxes(gridcolor=t["grid"], zerolinecolor=t["axis"], linecolor=t["axis"], tickfont={"color": t["muted"]})
    return fig


def rating_bar(
    table: pd.DataFrame,
    *,
    top: int = 25,
    color_col: str = "conference",
    value_col: str = "rating",
    value_label: str | None = None,
    title: str | None = None,
    theme: str = "light",
) -> go.Figure:
    """Horizontal ranked bar chart of the top ``top`` teams, coloured by conference."""
    df = table.sort_values(value_col, ascending=False).head(top).copy()
    counts = table[color_col].value_counts().to_dict() if color_col in table else {}
    colors = category_colors(df[color_col].tolist(), counts=counts, theme=theme) if color_col in df else {}
    t = _tokens(theme)
    df["label"] = [f"{r}. {name}" for r, name in zip(df["rank"], df["team"], strict=True)]
    fig = go.Figure()
    seen: set[str] = set()
    # Legend entries in a stable (alphabetical) order regardless of bar order.
    for cat in sorted(colors, key=lambda c: (colors[c] == t["other"], c)):
        name = cat if colors[cat] != t["other"] else OTHER
        if name in seen:
            continue
        seen.add(name)
        fig.add_bar(x=[None], y=[None], orientation="h", name=name, marker={"color": colors[cat]}, legendgroup=name)
    for _, row in df.iloc[::-1].iterrows():
        cat = row.get(color_col, "") if color_col in df else ""
        color = colors.get(cat, series_colors(theme)[0])
        name = cat if colors.get(cat) != t["other"] else OTHER
        fig.add_bar(
            x=[row[value_col]],
            y=[row["label"]],
            orientation="h",
            name=name,
            marker={"color": color, "cornerradius": 4},
            showlegend=False,
            legendgroup=name,
            customdata=[[row.get("record", ""), row.get("sos", float("nan")), cat]],
            hovertemplate=(
                "<b>%{y}</b><br>Rating %{x:.3f}<br>Record %{customdata[0]}"
                "<br>SOS %{customdata[1]:.3f}<br>%{customdata[2]}<extra></extra>"
            ),
        )
    fig.update_layout(barmode="overlay", bargap=0.25)
    fig.update_yaxes(type="category", tickfont={"size": 12})
    fig.update_xaxes(title_text=value_label or value_col.replace("_", " ").title())
    return _layout(fig, title=title, height=max(420, 24 * len(df) + 140), theme=theme)


def rating_sos_scatter(
    table: pd.DataFrame, *, label_top: int = 6, title: str | None = None, theme: str = "light"
) -> go.Figure:
    """Rating vs strength of schedule; only the top teams are labelled, alternating above/below."""
    df = table.copy()
    t = _tokens(theme)
    labels, positions = [], []
    for team, r in zip(df["team"], df["rank"], strict=True):
        labels.append(team if r <= label_top else "")
        positions.append("top center" if r % 2 else "bottom center")
    fig = go.Figure(
        go.Scatter(
            x=df["sos"],
            y=df["rating"],
            mode="markers+text",
            text=labels,
            textposition=positions,
            textfont={"color": t["ink_secondary"], "size": 11},
            marker={"size": 9, "color": series_colors(theme)[0], "line": {"width": 2, "color": t["ring"]}},
            customdata=list(zip(df["team"], df["record"], df["rank"], strict=True)),
            hovertemplate="<b>%{customdata[0]}</b> (#%{customdata[2]}, %{customdata[1]})"
            "<br>Rating %{y:.3f}<br>SOS %{x:.3f}<extra></extra>",
            name="Teams",
        )
    )
    fig.update_xaxes(title_text="Strength of schedule (mean opponent rating)")
    fig.update_yaxes(title_text="Rating")
    return _layout(fig, title=title, height=480, legend=False, theme=theme)


def trajectory_chart(
    snapshots: Mapping[int, Ratings],
    teams: Sequence[str],
    *,
    metric: str = "rank",
    title: str | None = None,
    theme: str = "light",
) -> go.Figure:
    """Weekly rank (or rating) lines for up to eight teams."""
    teams = list(teams)[:MAX_CATEGORIES]
    colors = {team: series_colors(theme)[i] for i, team in enumerate(teams)}
    t = _tokens(theme)
    weeks = sorted(snapshots)
    fig = go.Figure()
    worst_rank = 1
    for team in teams:
        xs, ys = [], []
        for w in weeks:
            ratings = snapshots[w]
            if team not in ratings:
                continue
            if metric == "rank":
                ordered = sorted(((n, v) for n, v in ratings.items()), key=lambda kv: (-kv[1], kv[0]))
                rank = next(i + 1 for i, (name, _) in enumerate(ordered) if name == team)
                ys.append(rank)
                worst_rank = max(worst_rank, rank)
            else:
                ys.append(ratings[team])
            xs.append(w)
        fig.add_scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name=team,
            line={"color": colors[team], "width": 2},
            marker={"size": 8, "line": {"width": 2, "color": t["ring"]}},
            hovertemplate=f"<b>{team}</b><br>Week %{{x}}<br>{metric.title()} %{{y}}<extra></extra>",
        )
    fig.update_xaxes(title_text="Week", dtick=1)
    if metric == "rank":
        span = max(worst_rank + 1, 10)
        fig.update_yaxes(title_text="Rank", range=[span + 0.5, 0.5], dtick=1 if span <= 12 else 5)
    else:
        fig.update_yaxes(title_text="Rating")
    fig.update_layout(hovermode="x unified")
    return _layout(fig, title=title, height=440, legend=len(teams) > 1, theme=theme)


def weekly_metric_chart(
    by_week: Mapping[str, pd.DataFrame], *, metric: str = "accuracy", title: str | None = None, theme: str = "light"
) -> go.Figure:
    """One line per method for a backtest metric across evaluation weeks."""
    fig = go.Figure()
    t = _tokens(theme)
    for i, (method, df) in enumerate(list(by_week.items())[:MAX_CATEGORIES]):
        if metric not in df:
            continue
        fig.add_scatter(
            x=df["week"],
            y=df[metric],
            mode="lines+markers",
            name=method,
            line={"color": series_colors(theme)[i], "width": 2},
            marker={"size": 8, "line": {"width": 2, "color": t["ring"]}},
            hovertemplate=f"<b>{method}</b><br>Week %{{x}}<br>{metric} %{{y:.3f}}<extra></extra>",
        )
    fig.update_xaxes(title_text="Week", dtick=1)
    fig.update_yaxes(title_text=metric.replace("_", " ").title())
    fig.update_layout(hovermode="x unified")
    return _layout(fig, title=title, height=420, theme=theme)


def calibration_chart(
    tables: Mapping[str, pd.DataFrame], *, title: str | None = None, theme: str = "light"
) -> go.Figure:
    """Reliability diagram: observed home-win rate against predicted probability."""
    fig = go.Figure()
    t = _tokens(theme)
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect calibration",
        line={"color": t["axis"], "dash": "dot", "width": 1.5},
        hoverinfo="skip",
    )
    for i, (method, df) in enumerate(list(tables.items())[:MAX_CATEGORIES]):
        d = df.dropna(subset=["observed"])
        fig.add_scatter(
            x=d["predicted"],
            y=d["observed"],
            mode="lines+markers",
            name=method,
            line={"color": series_colors(theme)[i], "width": 2},
            marker={"size": [max(6, min(18, 4 + n**0.5)) for n in d["n"]], "line": {"width": 2, "color": t["ring"]}},
            customdata=d["n"],
            hovertemplate=f"<b>{method}</b><br>Predicted %{{x:.2f}}<br>Observed %{{y:.2f}}<br>n=%{{customdata}}<extra></extra>",
        )
    fig.update_xaxes(title_text="Predicted home win probability", range=[0, 1])
    fig.update_yaxes(title_text="Observed home win rate", range=[0, 1])
    return _layout(fig, title=title, height=460, theme=theme)


def conference_dots(conf_table: pd.DataFrame, *, title: str | None = None, theme: str = "light") -> go.Figure:
    """Dot plot of the conference strength index.

    The index is band-scaled to 0.6-1.6, so a bar from zero would flatten every real
    difference; a dot on a hairline with the band as the axis range is the honest form.
    """
    df = conf_table.sort_values("strength_index", ascending=True)
    t = _tokens(theme)
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_scatter(
            x=[0.6, row["strength_index"]],
            y=[row["conference"], row["conference"]],
            mode="lines",
            line={"color": t["grid"], "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    fig.add_scatter(
        x=df["strength_index"],
        y=df["conference"],
        mode="markers+text",
        text=[f"{v:.2f}" for v in df["strength_index"]],
        textposition="middle right",
        textfont={"color": t["ink_secondary"], "size": 12},
        marker={"size": 12, "color": series_colors(theme)[0], "line": {"width": 2, "color": t["ring"]}},
        customdata=list(zip(df["teams"], df["top_team"], df["mean_rating"], strict=True)),
        hovertemplate="<b>%{y}</b><br>Strength index %{x:.2f}<br>%{customdata[0]} teams"
        "<br>Best: %{customdata[1]}<br>Mean rating %{customdata[2]:.3f}<extra></extra>",
        name="Strength index",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Strength index", range=[0.5, 1.75])
    fig.update_yaxes(type="category")
    return _layout(fig, title=title, height=max(300, 34 * len(df) + 110), legend=False, theme=theme)


# Backwards-compatible alias.
conference_bar = conference_dots


def win_distribution_chart(
    distribution: Sequence[float], current_wins: int, *, team: str = "", theme: str = "light"
) -> go.Figure:
    """Probability of each final win total for one team."""
    t = _tokens(theme)
    totals = [current_wins + k for k in range(len(distribution))]
    fig = go.Figure(
        go.Bar(
            x=[str(w) for w in totals],
            y=list(distribution),
            marker={"color": series_colors(theme)[0], "cornerradius": 4},
            text=[f"{p:.0%}" if p >= 0.02 else "" for p in distribution],
            textposition="outside",
            textfont={"color": t["ink_secondary"]},
            hovertemplate="<b>%{x} wins</b><br>%{y:.1%}<extra></extra>",
            name=team or "Final wins",
        )
    )
    fig.update_xaxes(title_text="Final regular-season wins", type="category")
    fig.update_yaxes(title_text="Probability", tickformat=".0%", range=[0, max(distribution) * 1.25 + 0.02])
    return _layout(fig, title=f"{team}: projected win total" if team else None, height=340, legend=False, theme=theme)
