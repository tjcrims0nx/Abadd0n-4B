"""
CLI design system per https://yannglt.com/writing/designing-for-command-line-interface

Principles: systematic ANSI colors, UTF-8 iconography, consistent spacing (1ch horizontal,
1 line vertical), bold for labels / dim for secondary. Works in light & dark terminals.
"""

# ── ANSI colors (16 base + bright; use FG_DEFAULT for readable foreground) ───
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Semantic roles
SUCCESS = "\033[32m"   # green
WARNING = "\033[33m"   # yellow
ERROR = "\033[31m"     # red
INFO = "\033[94m"      # bright blue
ACCENT = "\033[96m"    # bright cyan
MUTED = "\033[90m"     # gray (dimmed)

# Base palette (for compatibility)
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
GRAY = "\033[90m"
WHITE = "\033[97m"
FG_DEFAULT = "\033[39m"
LABEL = "\033[94m"
CYAN = "\033[36m"
CODE_FG = "\033[90m"
FENCE_FG = "\033[33m"

# Box drawing
BOX_TL, BOX_TR = "\u250c", "\u2510"
BOX_BL, BOX_BR = "\u2514", "\u2518"
BOX_H, BOX_V = "\u2500", "\u2502"
BOX_LIGHT = "\u2500"  # light horizontal

# UTF-8 iconography
ICON_SUCCESS = "\u2713"   # ✓
ICON_WARNING = "\u26a0"   # ⚠
ICON_ERROR = "\u2717"    # ✗
ICON_ARROW = "\u2192"    # →
ICON_BULLET = "\u2022"   # •
ICON_INDENT = "\u2502"   # │
ICON_BRANCH = "\u251c"   # ├
ICON_END = "\u2514"      # └

# Spacing (1ch = one character width in monospace)
INDENT_1 = "  "
INDENT_2 = "    "
INDENT_3 = "      "


def success(msg: str, detail: str = "") -> str:
    """Success feedback: ✓ green message, optional dim detail."""
    s = f"{GREEN}{ICON_SUCCESS}{RESET} {SUCCESS}{msg}{RESET}"
    if detail:
        s += f" {DIM}{detail}{RESET}"
    return s


def warning(msg: str, detail: str = "") -> str:
    """Warning feedback: ⚠ yellow message."""
    s = f"{YELLOW}{ICON_WARNING}{RESET} {WARNING}{msg}{RESET}"
    if detail:
        s += f" {DIM}{detail}{RESET}"
    return s


def error(msg: str, detail: str = "") -> str:
    """Error feedback: ✗ red message."""
    s = f"{RED}{ICON_ERROR}{RESET} {ERROR}{msg}{RESET}"
    if detail:
        s += f" {DIM}{detail}{RESET}"
    return s


def info(msg: str) -> str:
    """Info label: blue, no icon."""
    return f"{INFO}{msg}{RESET}"


def muted(msg: str) -> str:
    """Secondary text: dim gray."""
    return f"{MUTED}{msg}{RESET}"


def rule(width: int = 48) -> str:
    """Horizontal rule for section separation."""
    return f"{DIM}{BOX_LIGHT * width}{RESET}"


def dict_for_coding_tools() -> dict[str, str]:
    """Theme dict for coding_tools.py compatibility."""
    return {
        "cyan": ACCENT,
        "label": LABEL,
        "gray": GRAY,
        "dim": DIM,
        "white": WHITE,
        "green": GREEN,
        "red": RED,
        "reset": RESET,
    }
