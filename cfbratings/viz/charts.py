"""Plotly chart builders.

Colour choices follow a validated, colour-blind-safe palette: categorical hues are
assigned in a fixed order (never cycled; a ninth category folds into "Other"), magnitude
uses a single blue ramp, and text never wears a series colour. Every builder takes
``theme="light" | "dark"`` so ink, grid and series steps are chosen for the surface the
chart is actually drawn on rather than flipped automatically.

Team colours are opt-in. Builders accept a plain ``{team: "#rrggbb"}`` mapping (or a single
``accent``) and pass every value through :func:`resolve_team_color`, which keeps a colour that
would vanish against the theme's surface legible: first by tinting or shading it toward the
surface's opposite while it keeps its hue, then by the team's alternate colour when that is a real
colour, and last by the default series hue, so a navy on a dark page comes back as a lighter navy
rather than as the team's white, and a maize on a light page never disappears into the background.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

from ..data.schema import is_pool_team
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

# Every token is validated against its theme's ``surface``: ink and ink_secondary clear 4.5:1,
# muted clears 3:1 on both surfaces (so tick labels survive a page/theme mismatch), grid and
# axis are deliberately recessive hairlines, and ring equals the surface so markers crossing a
# line keep a gap. ``hover_bg`` is the tooltip box, one step off the surface, read with ``ink``.
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "other": "#898781",
        "ring": "#fcfcfb",
        "hover_bg": "#ffffff",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "other": "#898781",
        "ring": "#1a1a19",
        "hover_bg": "#262624",
    },
}
PALETTE = {**THEMES["light"], "surface": "rgba(0,0,0,0)", "good": "#0ca30c", "critical": "#d03b3b"}

# Lowest WCAG contrast against the surface at which a team colour is still used for a mark. The
# light palette's own faintest slot sits at 2.1:1, so this keeps team colours to the same bar.
MIN_MARK_CONTRAST = 2.0
# A colour that misses the floor is first nudged toward the surface's opposite, mixed with white on a
# dark surface or with black on a light one, in ``MIX_STEP`` increments. Contrast is a ratio of
# luminances and luminance grows roughly with the cube of lightness, so a short tint lifts a navy or
# a crimson over the floor on a dark page while it still reads as that colour, but the light page's
# floor sits so far down the scale that a pale colour would have to be shaded into olive or mud to
# reach it. A tint may therefore go to 30 % white; a shade stops after one step, enough to rescue a
# colour that just misses, and anything paler goes to the alternate colour instead.
MIX_STEP = 0.05
MAX_TINT = 0.30
MAX_SHADE = 0.10
# Below this spread between the largest and smallest RGB channel a colour is a grey. A grey that also
# sits near either end of the scale (white, black and their neighbours) has no hue to keep, so it is
# never nudged, and it is skipped as an alternate colour: most teams' second colour is one of them.
ACHROMATIC_SPREAD = 32
NEAR_WHITE_LUMINANCE = 0.6
NEAR_BLACK_LUMINANCE = 0.05
# Below this OKLab distance (x100) two colours read as one hue; the later line is dashed instead.
NEAR_COLOR_DISTANCE = 15.0
_DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")

_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

RGB = tuple[int, int, int]
_HEX = re.compile(r"#?([0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})", re.IGNORECASE)
_RGB_FUNC = re.compile(r"rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*(?:,[^)]*)?\)", re.IGNORECASE)


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


# --- colour arithmetic -----------------------------------------------------------------------------


def parse_color(value: object) -> RGB | None:
    """``#rgb``, ``#rrggbb`` (an alpha suffix is ignored) or ``rgb(r, g, b)`` → ``(r, g, b)``.

    Returns ``None`` for ``None``, non-strings and anything malformed; it never raises.
    """
    if not isinstance(value, str):
        return None
    s = value.strip()
    m = _HEX.fullmatch(s)
    if m:
        digits = m.group(1)
        if len(digits) <= 4:
            digits = "".join(ch * 2 for ch in digits[:3])
        return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))
    m = _RGB_FUNC.fullmatch(s)
    if m:
        r, g, b = (int(c) for c in m.groups())
        return (r, g, b) if max(r, g, b) <= 255 else None
    return None


def _hex(rgb: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _linear(channel: int) -> float:
    v = channel / 255
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: RGB) -> float:
    """WCAG 2 relative luminance of an sRGB triple."""
    r, g, b = (_linear(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: RGB, b: RGB) -> float:
    la, lb = relative_luminance(a), relative_luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def contrast_ratio(a: str, b: str) -> float:
    """WCAG 2 contrast ratio between two colours; 1.0 (no contrast) when either is malformed."""
    pa, pb = parse_color(a), parse_color(b)
    if pa is None or pb is None:
        return 1.0
    return _contrast(pa, pb)


def _oklab(rgb: RGB) -> tuple[float, float, float]:
    r, g, b = (_linear(c) for c in rgb)
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l_ + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l_ - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l_ + 0.7827717662 * m - 0.8086757660 * s,
    )


def color_distance(a: str, b: str) -> float:
    """How far apart two colours look: Euclidean OKLab distance x100 (``inf`` when malformed).

    Raw RGB distance over-rates dark pairs and under-rates reds; OKLab tracks the eye, and about
    15 is the floor below which two series colours stop being easy to tell apart.
    """
    pa, pb = parse_color(a), parse_color(b)
    if pa is None or pb is None:
        return math.inf
    return 100 * math.dist(_oklab(pa), _oklab(pb))


def _mix(rgb: RGB, toward: RGB, t: float) -> RGB:
    """``rgb`` moved the fraction ``t`` of the way to ``toward``: 0 keeps it, 1 is ``toward``."""
    r, g, b = (round(c + t * (w - c)) for c, w in zip(rgb, toward, strict=True))
    return (r, g, b)


def _near_white_or_black(rgb: RGB) -> bool:
    """True for a grey near either end of the scale: white, black and their close neighbours."""
    if max(rgb) - min(rgb) >= ACHROMATIC_SPREAD:
        return False
    lum = relative_luminance(rgb)
    return lum >= NEAR_WHITE_LUMINANCE or lum <= NEAR_BLACK_LUMINANCE


def _nudged(rgb: RGB, surface: RGB, *, dark: bool) -> RGB | None:
    """``rgb`` tinted (dark surface) or shaded (light surface) just enough to clear the floor, or None.

    Mixing toward white or black scales every channel difference alike, so the hue is untouched; the
    mix stops at :data:`MAX_TINT` or :data:`MAX_SHADE` so the result still reads as the team's colour
    rather than as a pastel or a mud, and a grey is left alone because it has no hue to keep.
    """
    if _near_white_or_black(rgb):
        return None
    toward, limit = ((255, 255, 255), MAX_TINT) if dark else ((0, 0, 0), MAX_SHADE)
    for step in range(1, round(limit / MIX_STEP) + 1):
        candidate = _mix(rgb, toward, step * MIX_STEP)
        if _contrast(candidate, surface) >= MIN_MARK_CONTRAST:
            return candidate
    return None


def resolve_team_color(
    color: str | None, alt_color: str | None = None, theme: str = "light", fallback: str | None = None
) -> str:
    """A team colour that stays legible on the theme's chart surface, as lower-case ``#rrggbb``.

    ``color`` is used when its WCAG contrast against ``THEMES[theme]["surface"]`` reaches
    :data:`MIN_MARK_CONTRAST`. When it misses, the same colour is nudged toward the surface's
    opposite (tinted with white on a dark surface, shaded with black on a light one) in
    :data:`MIX_STEP` increments until it clears the floor, within :data:`MAX_TINT` or
    :data:`MAX_SHADE`, so a navy or a crimson on a dark page comes back as a lighter navy or crimson
    rather than as the team's white. ``alt_color`` is tried next, but only when it is a real colour:
    a near-white or near-black alternate is skipped. Then ``fallback`` is taken as-is and, failing
    that, the theme's first series colour. ``None``, short ``#rgb`` and malformed strings are all
    accepted: a bad value is skipped, never raised.
    """
    tokens = _tokens(theme)
    surface = parse_color(tokens["surface"])
    if surface is None:  # pragma: no cover - the theme tables are literals
        return series_colors(theme)[0]
    rgb = parse_color(color)
    if rgb is not None:
        if _contrast(rgb, surface) >= MIN_MARK_CONTRAST:
            return _hex(rgb)
        nudged = _nudged(rgb, surface, dark=tokens is THEMES["dark"])
        if nudged is not None:
            return _hex(nudged)
    alt = parse_color(alt_color)
    if alt is not None and not _near_white_or_black(alt) and _contrast(alt, surface) >= MIN_MARK_CONTRAST:
        return _hex(alt)
    rgb = parse_color(fallback)
    return _hex(rgb) if rgb is not None else series_colors(theme)[0]


def _accent(accent: str | None, theme: str) -> str:
    """The default series colour, or ``accent`` when it is given and legible on this theme."""
    default = series_colors(theme)[0]
    return default if accent is None else resolve_team_color(accent, None, theme, fallback=default)


def _dash_styles(colors: Sequence[str]) -> list[str]:
    """Line dashes that keep look-alike colours apart: the first stays solid, each later clash dashes."""
    levels: list[int] = []
    for i, color in enumerate(colors):
        clashes = [levels[j] for j in range(i) if color_distance(colors[j], color) < NEAR_COLOR_DISTANCE]
        levels.append(min(max(clashes, default=-1) + 1, len(_DASHES) - 1))
    return [_DASHES[k] for k in levels]


# --- layout ----------------------------------------------------------------------------------------


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
        # Plotly's default tooltip is a white box with auto text, which glows on a dark page;
        # pin it to the theme so the readout is ink on a surface-coloured card in both modes.
        hoverlabel={"bgcolor": t["hover_bg"], "bordercolor": t["axis"], "font": {"family": _FONT, "color": t["ink"]}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0} if legend else None,
        showlegend=legend,
    )
    fig.update_xaxes(gridcolor=t["grid"], zerolinecolor=t["axis"], linecolor=t["axis"], tickfont={"color": t["muted"]})
    fig.update_yaxes(gridcolor=t["grid"], zerolinecolor=t["axis"], linecolor=t["axis"], tickfont={"color": t["muted"]})
    return fig


# --- charts ----------------------------------------------------------------------------------------

# d3 number formats for the value a ranked bar carries: rating-scale columns keep three decimals,
# point-scale ones (PPoints, momentum) one, and momentum keeps its sign as the table shows it.
_VALUE_FORMATS = {"rating": ".3f", "sos": ".3f", "ppoints": ".1f", "momentum": "+.1f"}


def _value_format(value_col: str, values: pd.Series) -> str:
    """The hover format for ``value_col``; an unknown column gets one decimal past about 10, else three."""
    fmt = _VALUE_FORMATS.get(value_col.lower())
    if fmt is None:
        span = float(values.abs().max()) if len(values) else 0.0
        fmt = ".1f" if span >= 10 else ".3f"
    return fmt


def rating_bar(
    table: pd.DataFrame,
    *,
    top: int = 25,
    color_col: str = "conference",
    value_col: str = "rating",
    value_label: str | None = None,
    title: str | None = None,
    theme: str = "light",
    team_colors: Mapping[str, str] | None = None,
) -> go.Figure:
    """Horizontal ranked bar chart of the top ``top`` teams.

    Bars are coloured by ``color_col`` (conference) with a legend. With ``team_colors`` (team →
    colour) each bar wears its own team's colour instead, the legend goes away and the hover still
    names the conference; every colour is re-checked against the theme surface first. The hover
    names the value the bars show by ``value_label`` (or the column), at a precision that suits it,
    and adds the rating line when the bars show something else.
    """
    df = table.sort_values(value_col, ascending=False).head(top).copy()
    counts = table[color_col].value_counts().to_dict() if color_col in table else {}
    colors = category_colors(df[color_col].tolist(), counts=counts, theme=theme) if color_col in df else {}
    t = _tokens(theme)
    default = series_colors(theme)[0]
    df["label"] = [f"{r}. {name}" for r, name in zip(df["rank"], df["team"], strict=True)]
    label = value_label or value_col.replace("_", " ").title()
    lines = ["<b>%{y}</b>", f"{label} %{{x:{_value_format(value_col, df[value_col])}}}"]
    if value_col != "rating" and "rating" in df:
        lines.append("Rating %{customdata[3]:.3f}")
    lines.append("Record %{customdata[0]}")
    if value_col != "sos":
        lines.append("SOS %{customdata[1]:.3f}")
    lines.append("%{customdata[2]}")
    hover = "<br>".join(lines) + "<extra></extra>"
    fig = go.Figure()
    if team_colors is None:
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
        name = cat if colors.get(cat) != t["other"] else OTHER
        if team_colors is None:
            color = colors.get(cat, default)
        else:
            color = resolve_team_color(team_colors.get(row["team"]), None, theme, fallback=default)
        fig.add_bar(
            x=[row[value_col]],
            y=[row["label"]],
            orientation="h",
            name=name,
            marker={"color": color, "cornerradius": 4},
            showlegend=False,
            legendgroup=name,
            customdata=[[row.get("record", ""), row.get("sos", float("nan")), cat, row.get("rating", float("nan"))]],
            hovertemplate=hover,
        )
    fig.update_layout(barmode="overlay", bargap=0.25)
    fig.update_yaxes(type="category", tickfont={"size": 12})
    fig.update_xaxes(title_text=label)
    return _layout(fig, title=title, height=max(420, 24 * len(df) + 140), legend=team_colors is None, theme=theme)


def rating_sos_scatter(
    table: pd.DataFrame, *, label_top: int = 6, title: str | None = None, theme: str = "light"
) -> go.Figure:
    """Rating vs strength of schedule, with selective labels.

    The ``label_top`` highest-rated rows *of the frame as given* are labelled, alternating above and
    below their marks, plus the rows with the weakest and the strongest schedule, whose labels point
    inward from the edges. Labels never depend on a global rank, so a conference view keeps them.
    """
    df = table.reset_index(drop=True)
    t = _tokens(theme)
    labels = [""] * len(df)
    positions = ["top center"] * len(df)
    if label_top > 0 and len(df):
        best = df["rating"].sort_values(ascending=False).index[:label_top].tolist()
        for i, idx in enumerate(best):
            labels[idx] = df.at[idx, "team"]
            positions[idx] = "top center" if i % 2 == 0 else "bottom center"
        sos = df["sos"].dropna()
        if not sos.empty:
            for idx, position in ((sos.idxmin(), "middle right"), (sos.idxmax(), "middle left")):
                if not labels[idx]:
                    labels[idx] = df.at[idx, "team"]
                    positions[idx] = position
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
    team_colors: Mapping[str, str] | None = None,
) -> go.Figure:
    """Weekly rank (or rating) lines for up to eight teams.

    Lines take the categorical slots in selection order, or each team's own colour from
    ``team_colors`` (a team that is missing or illegible on this theme keeps its slot). When two
    team colours are too close to tell apart the first stays solid and the later one is dashed.
    """
    teams = list(teams)[:MAX_CATEGORIES]
    slots = series_colors(theme)
    if team_colors is None:
        colors = list(slots[: len(teams)])
        dashes = ["solid"] * len(teams)
    else:
        colors = [
            resolve_team_color(team_colors.get(team), None, theme, fallback=slots[i]) for i, team in enumerate(teams)
        ]
        dashes = _dash_styles(colors)
    t = _tokens(theme)
    weeks = sorted(snapshots)
    fig = go.Figure()
    worst_rank = 1
    for i, team in enumerate(teams):
        xs, ys = [], []
        for w in weeks:
            ratings = snapshots[w]
            if team not in ratings:
                continue
            if metric == "rank":
                ordered = sorted(
                    ((n, v) for n, v in ratings.items() if not is_pool_team(n)), key=lambda kv: (-kv[1], kv[0])
                )
                rank = next(i + 1 for i, (name, _) in enumerate(ordered) if name == team)
                ys.append(rank)
                worst_rank = max(worst_rank, rank)
            else:
                ys.append(ratings[team])
            xs.append(w)
        line: dict[str, object] = {"color": colors[i], "width": 2}
        if dashes[i] != "solid":
            line["dash"] = dashes[i]
        fig.add_scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name=team,
            line=line,
            marker={"size": 8, "line": {"width": 2, "color": t["ring"]}},
            hovertemplate=f"<b>{team}</b><br>Week %{{x}}<br>{metric.title()} %{{y}}<extra></extra>",
        )
    fig.update_xaxes(title_text="Week", dtick=1, tickformat="d")
    if weeks:  # half a week of padding each side; autorange would pad to -1 and 18 at dtick=1
        fig.update_xaxes(range=[weeks[0] - 0.5, weeks[-1] + 0.5])
    if metric == "rank":
        span = max(worst_rank + 1, 10)
        fig.update_yaxes(title_text="Rank", range=[span + 0.5, 0.5], dtick=1 if span <= 12 else 5)
    else:
        fig.update_yaxes(title_text="Rating")
    fig.update_layout(hovermode="x unified")
    return _layout(fig, title=title, height=440, legend=len(teams) > 1, theme=theme)


def weekly_metric_chart(
    by_week: Mapping[str, pd.DataFrame],
    *,
    metric: str = "accuracy",
    title: str | None = None,
    theme: str = "light",
    min_games: int = 5,
) -> go.Figure:
    """One line per method for a backtest metric across evaluation weeks.

    Weeks with fewer than ``min_games`` games (the frames' ``n`` column, as ``BacktestResult.by_week``
    emits it) are dropped from every series, so a two-game week cannot swing a line; the hover shows
    each week's game count. Frames without an ``n`` column are plotted whole.
    """
    fig = go.Figure()
    t = _tokens(theme)
    label = metric.replace("_", " ")
    plotted: list[int] = []
    for i, (method, df) in enumerate(list(by_week.items())[:MAX_CATEGORIES]):
        if metric not in df:
            continue
        has_n = "n" in df
        if has_n and min_games > 0:
            df = df[df["n"] >= min_games]
        plotted.extend(int(w) for w in df["week"].dropna())
        hover = f"<b>{method}</b><br>Week %{{x}}<br>{label} %{{y:.3f}}"
        if has_n:
            hover += "<br>%{customdata} games"
        fig.add_scatter(
            x=df["week"].tolist(),
            y=df[metric].tolist(),
            mode="lines+markers",
            name=method,
            line={"color": series_colors(theme)[i], "width": 2},
            marker={"size": 8, "line": {"width": 2, "color": t["ring"]}},
            customdata=df["n"].tolist() if has_n else None,
            hovertemplate=hover + "<extra></extra>",
        )
    fig.update_xaxes(title_text="Week", dtick=1, tickformat="d")
    if plotted:  # half a week of padding each side instead of autorange's 0...18
        fig.update_xaxes(range=[min(plotted) - 0.5, max(plotted) + 0.5])
    fig.update_yaxes(title_text=label.title())
    fig.update_layout(hovermode="x unified")
    return _layout(fig, title=title, height=460, theme=theme)


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


def conference_dots(
    conf_table: pd.DataFrame, *, title: str | None = None, theme: str = "light", accent: str | None = None
) -> go.Figure:
    """Dot plot of the conference strength index.

    The index is band-scaled to 0.6-1.6, so a bar from zero would flatten every real
    difference; a dot on a hairline with the band as the axis range is the honest form.
    ``accent`` recolours the dots (a team colour, checked against the surface first).
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
        marker={"size": 12, "color": _accent(accent, theme), "line": {"width": 2, "color": t["ring"]}},
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
    distribution: Sequence[float],
    current_wins: int,
    *,
    team: str = "",
    theme: str = "light",
    accent: str | None = None,
) -> go.Figure:
    """Probability of each final win total for one team; ``accent`` recolours the bars."""
    t = _tokens(theme)
    totals = [current_wins + k for k in range(len(distribution))]
    fig = go.Figure(
        go.Bar(
            x=[str(w) for w in totals],
            y=list(distribution),
            marker={"color": _accent(accent, theme), "cornerradius": 4},
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
