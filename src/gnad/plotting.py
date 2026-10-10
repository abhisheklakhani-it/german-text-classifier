"""Shared chart colours and axis style (reference palette: slot 1-2 hues, ink and chrome)."""

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def style_axis(ax, grid: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0, colors=INK_2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
