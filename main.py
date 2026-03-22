import sys
from pathlib import Path
# Must run in venv_win or venv_wsl (avoids loading from global Python without LoRA)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import venv_check
venv_check.require_abaddon_venv()

import pre_unsloth
pre_unsloth.before_import()

import inspect
import torch
import os
import threading
import warnings


import shutil
import contextlib
import signal
from pathlib import Path

# ── CLI design system (colors, icons, spacing) ───────────────────────────────
from cli_theme import (
    ACCENT, BOLD, CODE_FG, DIM, FG_DEFAULT, FENCE_FG, GRAY, GREEN, LABEL, RED, RESET,
    WHITE, YELLOW, BOX_BL, BOX_BR, BOX_H, BOX_TL, BOX_TR, BOX_V,
    ICON_ARROW, ICON_BULLET, ICON_ERROR, ICON_SUCCESS, ICON_WARNING,
    muted, success, warning, error as theme_error,
)

# Aliases for existing usages
CODE_BOX_TL, CODE_BOX_TR = BOX_TL, BOX_TR
CODE_BOX_BL, CODE_BOX_BR = BOX_BL, BOX_BR
CODE_BOX_H, CODE_BOX_V = BOX_H, BOX_V
DARK_RED = "\033[91m"
CYAN = "\033[36m"


def _thinking_spinner_frames_interval():
    """Frames + sleep seconds from [spinners](https://pypi.org/project/spinners/); fallback if missing."""
    try:
        from spinners import Spinners

        spec = Spinners.dots12.value
        return spec["frames"], spec["interval"] / 1000.0
    except Exception:
        return ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"], 0.08


@contextlib.contextmanager
def spinner_context(tty_enabled: bool = True):
    """Spinner while the model generates; cleans up on first token or cancel."""
    if not tty_enabled:
        sys.stdout.write(f"\n{GRAY}Abadd0n{RESET} {DIM}\u2192{RESET} {FG_DEFAULT} ")
        sys.stdout.flush()
        yield lambda: None
        return

    stop_event = threading.Event()
    frames, tick = _thinking_spinner_frames_interval()
    reply_started = [False]

    def _spin():
        i = 0
        while not stop_event.wait(tick):
            fr = frames[i % len(frames)]
            sys.stdout.write(f"\r{GRAY}Abadd0n{RESET} {DIM}thinking{RESET} {fr} ")
            sys.stdout.flush()
            i += 1

    sys.stdout.write("\n")
    sys.stdout.flush()
    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()

    def _on_first_put():
        reply_started[0] = True
        stop_event.set()
        thread.join(timeout=0.5)
        sys.stdout.write(f"\r\033[K{GRAY}Abadd0n{RESET} {DIM}\u2192{RESET} {FG_DEFAULT} ")
        sys.stdout.flush()

    try:
        yield _on_first_put
    finally:
        stop_event.set()
        if thread.is_alive():
            thread.join(timeout=0.2)
        if not reply_started[0]:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


def patch_unsloth_qwen3_rope_broadcast():
    """
    Unsloth's Qwen3Attention_fast_forward_inference does Qn *= cos where cos can span
    the full prompt (e.g. 163) while Qn is only the new token(s) — broadcast error.
    We splice cos/sin to match Qn.shape[2] using the *current* Unsloth source so we
    keep upstream behavior (including paged_attention_V write, SDPA path, etc.).
    """
    try:
        import unsloth.models.qwen3 as qwen3_models

        src = inspect.getsource(qwen3_models.Qwen3Attention_fast_forward_inference)
        needle = "    sin = sin[position_ids].unsqueeze(1)\n"
        graft = needle + (
            "    _qrope = Qn.shape[2]\n"
            "    if cos.shape[2] != _qrope:\n"
            "        cos = cos[:, :, -_qrope:, :].contiguous()\n"
            "        sin = sin[:, :, -_qrope:, :].contiguous()\n"
        )
        if needle not in src:
            raise RuntimeError("Unsloth qwen3.py changed; needle not found for RoPE patch")
        new_src = src.replace(needle, graft, 1)
        ns = dict(qwen3_models.__dict__)
        exec(compile(new_src, "<abadd0n_qwen3_rope_fix>", "exec"), ns)
        qwen3_models.Qwen3Attention_fast_forward_inference = ns[
            "Qwen3Attention_fast_forward_inference"
        ]
        print(muted("Abadd0n: Qwen3 RoPE broadcast fix applied."))
    except Exception as e:
        print(theme_error(f"Abadd0n: Qwen3 RoPE patch failed ({e})."))


# Disable with ABADDON_QWEN3_INFER_PATCH=0 if you need stock Unsloth for debugging.
if os.environ.get("ABADDON_QWEN3_INFER_PATCH", "1") != "0":
    patch_unsloth_qwen3_rope_broadcast()

# Set encoding for Windows CLI
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except: pass

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from unsloth import FastLanguageModel
from transformers import TextStreamer
import re

from coding_tools import handle_slash_command

try:
    from rich.console import Console
    from rich.syntax import Syntax
    from rich.panel import Panel
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


def _print_user_prompt_box(text: str) -> None:
    """Render user input in a Composer-style dark box with subtle border."""
    if not text.strip():
        return
    if _RICH_AVAILABLE:
        console = Console()
        console.print(
            Panel(
                text.strip(),
                title="[bold blue]You[/]",
                border_style="dim",
                padding=(0, 1),
                expand=False,
            )
        )
    else:
        w = shutil.get_terminal_size(fallback=(80, 24)).columns - 4
        w = max(40, min(76, w))
        print(f"\n{GRAY}{BOX_TL}{BOX_H * (w - 2)}{BOX_TR}{RESET}")
        print(f"{GRAY}{BOX_V}{RESET} {LABEL}You{RESET} {FG_DEFAULT}{text.strip()}{RESET}")
        print(f"{GRAY}{BOX_BL}{BOX_H * (w - 2)}{BOX_BR}{RESET}\n")


def _cli_panel_width() -> int:
    cols, _ = shutil.get_terminal_size(fallback=(80, 24))
    return max(52, min(78, cols - 4))


class _ReplyStreamer(TextStreamer):
    """Prose + markdown fences; code bodies render in a box fitted to content width."""

    _OPEN_FENCE = re.compile(r"^```([^`\n]*)(\n?)")

    def __init__(self, *args, on_first_put=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_first_put = on_first_put
        self._in_code = False
        self._pending = ""
        self._code_accum = ""
        self._code_lang = ""

    def put(self, value):
        if self._on_first_put is not None:
            fn = self._on_first_put
            self._on_first_put = None
            fn()
        super().put(value)

    @staticmethod
    def _trailing_backtick_run(s: str) -> int:
        n = 0
        for c in reversed(s):
            if c == "`":
                n += 1
            else:
                break
        return n

    def _flush_code_box(self, body: str, lang: str = "") -> None:
        if body == "":
            return
        lang = (lang or "").strip().lower()
        lexer = lang if lang in ("python", "py", "json", "yaml", "yml", "html", "javascript", "js", "bash", "sh", "sql", "markdown", "md", "xml", "css", "php", "go", "rust", "java", "c", "cpp") else "text"
        if lexer == "py":
            lexer = "python"
        elif lexer in ("yml", "yaml"):
            lexer = "yaml"
        elif lexer == "md":
            lexer = "markdown"
        elif lexer == "js":
            lexer = "javascript"
        elif lexer == "sh":
            lexer = "bash"

        if _RICH_AVAILABLE:
            try:
                console = Console()
                syntax = Syntax(body.rstrip(), lexer, theme="ansi_dark", line_numbers=False)
                console.print(Panel(syntax, border_style="dim", padding=(0, 1), expand=False))
                return
            except Exception:
                pass
        cols, _ = shutil.get_terminal_size(fallback=(80, 24))
        inner_max = max(8, cols - 4)
        raw = body.split("\n")
        rows: list[str] = []
        for line in raw:
            t = line.expandtabs(4)
            if len(t) > inner_max:
                t = t[: inner_max - 1] + "…"
            rows.append(t)
        w = max((len(r) for r in rows), default=1)
        b = GRAY
        top = f"{b}{CODE_BOX_TL}{CODE_BOX_H * w}{CODE_BOX_TR}{RESET}"
        bot = f"{b}{CODE_BOX_BL}{CODE_BOX_H * w}{CODE_BOX_BR}{RESET}"
        sys.stdout.write(f"{top}\n")
        for row in rows:
            sys.stdout.write(
                f"{b}{CODE_BOX_V}{RESET}{CODE_FG}{row.ljust(w)}{RESET}{b}{CODE_BOX_V}{RESET}\n"
            )
        sys.stdout.write(f"{bot}\n")

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self._emit_stream_chunk(text)
        if stream_end:
            if self._pending:
                if self._in_code:
                    self._code_accum += self._pending
                    self._pending = ""
                    self._flush_code_box(self._code_accum, self._code_lang)
                    self._code_accum = ""
                    self._code_lang = ""
                    self._in_code = False
                else:
                    sys.stdout.write(f"{FG_DEFAULT}{self._pending}{RESET}")
                    self._pending = ""
            elif self._in_code and self._code_accum:
                self._flush_code_box(self._code_accum, self._code_lang)
                self._code_accum = ""
                self._code_lang = ""
                self._in_code = False
            sys.stdout.write("\n")
        sys.stdout.flush()

    def _emit_stream_chunk(self, chunk: str) -> None:
        data = self._pending + chunk
        self._pending = ""
        while data:
            if not self._in_code:
                i = data.find("```")
                if i < 0:
                    tb = self._trailing_backtick_run(data)
                    if 0 < tb < 3:
                        self._pending = data[-tb:]
                        data = data[:-tb]
                    if data:
                        sys.stdout.write(f"{FG_DEFAULT}{data}{RESET}")
                    return
                if i > 0:
                    sys.stdout.write(f"{FG_DEFAULT}{data[:i]}{RESET}")
                data = data[i:]
                m = self._OPEN_FENCE.match(data)
                if not m:
                    self._pending = data
                    return
                fence = data[: m.end()]
                sys.stdout.write(f"{DIM}{FENCE_FG}{fence}{RESET}")
                data = data[m.end() :]
                self._in_code = True
                self._code_accum = ""
                self._code_lang = (m.group(1) or "").strip()
                continue

            i = data.find("```")
            if i < 0:
                tb = self._trailing_backtick_run(data)
                if 0 < tb < 3:
                    self._pending = data[-tb:]
                    data = data[:-tb]
                self._code_accum += data
                return
            self._code_accum += data[:i]
            self._flush_code_box(self._code_accum, self._code_lang)
            self._code_accum = ""
            self._code_lang = ""
            sys.stdout.write(f"{DIM}{FENCE_FG}```{RESET}")
            data = data[i + 3 :]
            self._in_code = False


PROJECT_ROOT = Path(__file__).resolve().parent

from persona import PERSONA


def _system_content_with_skills() -> str:
    """PERSONA + installed ClawHub skills (if any)."""
    try:
        from core.clawhub import load_installed_skills
        skills = load_installed_skills(PROJECT_ROOT)
        if skills:
            return PERSONA + skills[:6000]
    except ImportError:
        pass
    return PERSONA

_WRITE_FILE_RE = re.compile(
    r"<write_file\s+path\s*=\s*[\"']([^\"']+)[\"']\s*>(.*?)</write_file>",
    re.DOTALL | re.IGNORECASE,
)
_EDIT_FILE_RE = re.compile(
    r"<edit_file\s+path\s*=\s*[\"']([^\"']+)[\"']\s*>(.*?)</edit_file>",
    re.DOTALL | re.IGNORECASE,
)
_MATH_RE = re.compile(r"<math>(.*?)</math>", re.DOTALL | re.IGNORECASE)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)


def _write_roots() -> list[Path]:
    roots = [PROJECT_ROOT]
    extra = os.environ.get("ABADDON_WRITE_ROOT", "").strip()
    if extra:
        roots.append(Path(extra).resolve())
    return roots


def resolve_safe_write_path(raw: str) -> Path | None:
    """Paths must fall under PROJECT_ROOT or ABADDON_WRITE_ROOT unless ABADDON_ALLOW_WRITES_OUTSIDE_ROOT=1."""
    raw = (raw or "").strip()
    if not raw or "\x00" in raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        cand = p.resolve()
    else:
        cand = (PROJECT_ROOT / p).resolve()
    allow_out = os.environ.get("ABADDON_ALLOW_WRITES_OUTSIDE_ROOT", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if allow_out:
        return cand
    for root in _write_roots():
        try:
            cand.relative_to(root)
            return cand
        except ValueError:
            continue
    return None


def _cli_panel_width() -> int:
    cols, _ = shutil.get_terminal_size(fallback=(80, 24))
    return max(52, min(78, cols - 4))


def _file_payload_preview(text: str, *, max_lines: int = 5, max_chars: int = 72) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        if len(out) >= max_lines:
            break
        vis = line.expandtabs(4)
        if not vis.strip() and not out:
            continue
        if len(vis) > max_chars:
            vis = vis[: max_chars - 1] + "\u2026"
        out.append(vis)
    return out if out else ["(empty file)"]


def _lexer_from_path(path: Path) -> str:
    """Infer Rich lexer from file extension."""
    ext = path.suffix.lower().lstrip(".")
    _map = {".py": "python", ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".md": "markdown",
            ".html": "html", ".js": "javascript", ".ts": "typescript", ".sh": "bash",
            ".sql": "sql", ".xml": "xml", ".css": "css", ".php": "php", ".go": "go",
            ".rs": "rust", ".java": "java", ".c": "c", ".cpp": "cpp", ".h": "c"}
    return _map.get(f".{ext}", "text")


def _prompt_pending_file_write(
    *,
    index: int,
    total: int,
    rel: Path,
    absolute: Path,
    content: str,
) -> str:
    """
    Interactive confirmation for one proposed write. Returns:
      'skip' — do not write this file
      'write' — write this file only
      'write_all' — write this file and skip prompts for the rest
    """
    n_lines = len(content.splitlines())
    exists = absolute.exists()
    w = _cli_panel_width()

    if _RICH_AVAILABLE:
        console = Console()
        header = f"[bold]{rel.name}[/]"
        if not exists:
            header += " [green](new)[/]"
        header += f" [green]+{n_lines}[/]"
        lexer = _lexer_from_path(rel)
        preview_content = "\n".join(content.splitlines()[:30])
        if len(content.splitlines()) > 30:
            preview_content += "\n... (truncated)"
        try:
            syntax = Syntax(preview_content, lexer, theme="ansi_dark", line_numbers=False)
            inner = Panel(syntax, title=header, border_style="dim", padding=(0, 1))
            console.print(f"\n[dim]Pending file write — proposal {index} of {total}[/]")
            console.print(inner)
        except Exception:
            preview = _file_payload_preview(content)
            console.print(Panel("\n".join(preview), title=header, border_style="dim"))
    else:
        rule = f"{DIM}{'\u2500' * w}{RESET}"
        preview = _file_payload_preview(content)
        print(f"\n{rule}")
        print(f"{BOLD}{LABEL}Pending file write{RESET} {GRAY}\u2014{RESET} {DIM}proposal {index} of {total}{RESET}")
        print(rule)
        print(f"  {LABEL}{rel.as_posix()}{RESET}" + (f" {GREEN}(new){RESET}" if not exists else "") + f" {GREEN}+{n_lines}{RESET}")
        for pl in preview:
            print(f"    {DIM}\u2502{RESET} {CODE_FG}{pl}{RESET}")
        print(rule)

    print(f"  {GRAY}Options{RESET}")
    print(f"    {DIM}\u2022{RESET} {LABEL}y{RESET} {DIM}yes{RESET} \u2014 apply")
    print(f"    {DIM}\u2022{RESET} {LABEL}n{RESET} {DIM}no{RESET} \u2014 skip {GRAY}(default){RESET}")
    if index < total:
        print(f"    {DIM}\u2022{RESET} {LABEL}a{RESET} {DIM}all{RESET} \u2014 approve remaining")
    prompt = f"\n  {GRAY}[y/n/a]{RESET} "

    while True:
        raw = input(prompt).strip().lower()
        if raw in ("", "n", "no"):
            return "skip"
        if raw in ("y", "yes"):
            return "write"
        if raw in ("a", "all"):
            return "write_all"
        print(
            f"{DIM}  Unrecognized input. Enter {LABEL}y{RESET}{DIM} (yes), {LABEL}n{RESET}{DIM} (no),"
            f" or {LABEL}a{RESET}{DIM} (approve all remaining).{RESET}"
        )


def _handle_interactive_requests(user_input: str) -> tuple[str, str | None]:
    """
    Intercept math and search requests in chat. Returns (message_for_model, direct_response).
    If direct_response is not None, skip the model and show it. Else use message_for_model (may be enriched).
    """
    raw = user_input.strip()
    lower = raw.lower()

    # ── Search: "search for X", "google X", "look up X", "find X"
    search_prefixes = ("search for ", "search ", "google ", "look up ", "find ", "lookup ")
    for p in search_prefixes:
        if lower.startswith(p):
            query = raw[len(p):].strip()
            if query:
                try:
                    from core.web_search import web_search
                    r = web_search(query, max_results=6)
                    if r.get("ok") and r.get("results"):
                        ctx = "Web search results:\n" + "\n".join(
                            f"{i}. {x['title']} — {x['href']}\n   {x.get('body', '')[:200]}"
                            for i, x in enumerate(r["results"], 1)
                        )
                        enriched = f"[Context from Google search]\n{ctx}\n\nUser: {raw}"
                        return (enriched, None)
                except Exception:
                    pass
            break

    # ── Math: "what is X", "calculate X", "compute X", or pure expression
    math_prefixes = ("what is ", "what's ", "calculate ", "compute ", "evaluate ", "solve ")
    for p in math_prefixes:
        if lower.startswith(p):
            expr = raw[len(p):].strip().rstrip("?")
            expr = expr.replace("% of ", "*0.01*").replace("% of", "*0.01*")
            try:
                from core.math_tool import evaluate_math
                r = evaluate_math(expr)
                if r.get("ok"):
                    result = r["result"]
                    direct = f"{LABEL}{result}{RESET}  {DIM}({expr}){RESET}"
                    return (raw, direct)  # Pass through for model, but we'll use direct
            except Exception:
                pass
            break

    # Pure math expression (e.g. "2+3*4" or "sqrt(16)"): short, no spaces, has operators/digits
    if raw and len(raw) < 60 and " " not in raw:
        math_chars = set("0123456789+-*/.()%e ")
        if any(c in raw for c in "+-*/()") or "sqrt" in lower or "sin" in lower or "cos" in lower:
            try:
                from core.math_tool import evaluate_math
                r = evaluate_math(raw)
                if r.get("ok"):
                    direct = f"{LABEL}{r['result']}{RESET}"
                    return (raw, direct)
            except Exception:
                pass

    return (raw, None)


def _looks_like_coding_request(text: str) -> bool:
    t = text.lower()
    keys = (
        "write file",
        "create file",
        "save ",
        "new file",
        ".py",
        ".js",
        ".ts",
        ".html",
        ".php",
        ".css",
        ".json",
        ".md",
        ".sh",
        "implement",
        "refactor",
        "full code",
        "source code",
        "script",
        "patch",
        "add a file",
        "make a file",
        "generate ",
        "html",
        "markup",
        "php",
        "laravel",
        "wordpress",
    )
    return any(k in t for k in keys)


def _effective_max_new_tokens(user_input: str, prompt_len_tokens: int) -> int:
    env = os.environ.get("ABADDON_MAX_NEW_TOKENS", "").strip()
    if env.isdigit():
        return max(32, min(int(env), 2048))
    if _looks_like_coding_request(user_input):
        room = 2048 - int(prompt_len_tokens) - 32
        return max(256, min(room, 1536))
    return 100


BANNER_FILENAME = "ascii_banner.txt"


def _banner_file_path() -> Path | None:
    override = os.environ.get("ABADDON_BANNER_PATH", "").strip()
    if override:
        p = Path(override).expanduser()
        return p if p.is_file() else None
    p = PROJECT_ROOT / BANNER_FILENAME
    return p if p.is_file() else None


def _load_banner_raw_lines() -> list[str]:
    path = _banner_file_path()
    if path is None:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _fit_banner_lines(lines: list[str], max_w: int, max_h: int) -> list[str]:
    if not lines:
        return []
    if len(lines) > max_h:
        start = max(0, (len(lines) - max_h) // 2)
        lines = lines[start : start + max_h]
    out: list[str] = []
    for line in lines:
        s = line.rstrip("\r\n")
        if len(s) > max_w:
            cut = max(0, (len(s) - max_w) // 2)
            s = s[cut : cut + max_w]
        out.append(s)
    return out


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_stylized_header():
    clear_screen()
    cols, rows = shutil.get_terminal_size(fallback=(80, 24))

    def ansi_center(text: str, width: int) -> str:
        visible_len = len(re.sub(r"\033\[[0-9;]*m", "", text))
        padding = max(0, (width - visible_len) // 2)
        return " " * padding + text

    max_w = max(36, cols - 2)
    max_h = min(20, max(8, rows - 12))
    raw = _load_banner_raw_lines()
    fitted = _fit_banner_lines(raw, max_w, max_h)

    for line in fitted:
        styled = f"{DIM}{ACCENT}{line}{RESET}"
        print(ansi_center(styled, cols))

    inner = min(56, max(40, cols - 12))
    rule = f"{DIM}{BOX_H * inner}{RESET}"
    tail = [
        "",
        ansi_center(f"{BOLD}{LABEL}ABADD0N{RESET} {DIM}\u00b7{RESET} {DIM}infernal coding assistant{RESET}", cols),
        ansi_center(rule, cols),
        ansi_center(
            f"{DIM}Tab{RESET} {GRAY}quick{RESET}  "
            f"{DIM}/{RESET} {GRAY}menu{RESET}  "
            f"{LABEL}exit{RESET} {LABEL}clear{RESET} {LABEL}tools{RESET}",
            cols,
        ),
        "",
    ]
    for line in tail:
        print(line)

def load_model_and_tokenizer():
    print(f"\n{GRAY}\u2502{RESET} {DIM}Loading model and tokenizer{RESET} ...")

    hf_token = os.environ.get("HF_TOKEN")
    load_kw = dict(
        model_name="lora_model",
        max_seq_length=2048,
        load_in_4bit=True,
        attn_implementation="sdpa",
    )
    if hf_token:
        load_kw["token"] = hf_token

    model, tokenizer = FastLanguageModel.from_pretrained(**load_kw)

    FastLanguageModel.for_inference(model)

    from unsloth.chat_templates import get_chat_template

    # Must match base model: lora_model is Qwen3ForCausalLM (see adapter_config.json).
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = 2048
    gen = getattr(model, "generation_config", None)
    if gen is not None:
        try:
            gen.max_length = None
        except Exception:
            pass
    print(success("Model ready."))
    return model, tokenizer

def chat(model, tokenizer, user_input, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})
    
    prompt = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    vocab = getattr(model.config, "vocab_size", None) or model.get_input_embeddings().weight.shape[0]
    ids = inputs["input_ids"]
    if (ids < 0).any() or (ids >= vocab).any():
        bad = ids[(ids < 0) | (ids >= vocab)]
        raise ValueError(
            f"Token id(s) out of range [0, {vocab}): {bad.tolist()[:20]}. "
            "Check chat template vs Qwen3 tokenizer."
        )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    tty = sys.stdout.isatty()
    no_spinner = os.environ.get("ABADDON_NO_SPINNER", "0").lower() in ("1", "true", "yes")

    with spinner_context(tty_enabled=(tty and not no_spinner)) as on_first_put:
        streamer = _ReplyStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            on_first_put=on_first_put,
        )

        use_cache = os.environ.get("ABADDON_GENERATE_NO_CACHE", "0") != "1"

        prompt_len_tokens = inputs["input_ids"].shape[1]
        gen_kw = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=_effective_max_new_tokens(user_input, prompt_len_tokens),
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=True,
            use_cache=use_cache,
        )
        if pad_id is not None:
            gen_kw["pad_token_id"] = pad_id
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            gen_kw["eos_token_id"] = eos_id

        with torch.no_grad():
            outputs = model.generate(**gen_kw)
    
    print(f"{RESET}", end="")
    
    # Extract response for history
    prompt_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    
    if "<think>" in response:
        response = response.split("</think>")[-1].strip()
    
    response = _expand_math_and_search(response)
    conversation_history.append({"role": "assistant", "content": response})
    
    return response

def _expand_math_and_search(response: str) -> str:
    """Replace <math>expr</math> and <search>query</search> in response with computed results."""
    def _math_repl(m):
        try:
            from core.math_tool import evaluate_math
            r = evaluate_math(m.group(1).strip())
            return str(r["result"]) if r.get("ok") else m.group(0)
        except Exception:
            return m.group(0)

    def _search_repl(m):
        try:
            from core.web_search import web_search
            q = m.group(1).strip()
            r = web_search(q, max_results=5)
            if not r.get("ok") or not r.get("results"):
                return m.group(0)
            lines = [f"{i}. {x['title']}: {x['href']}" for i, x in enumerate(r["results"], 1)]
            return "[Web search: " + "; ".join(lines[:3]) + "]"
        except Exception:
            return m.group(0)

    response = _MATH_RE.sub(_math_repl, response)
    response = _SEARCH_RE.sub(_search_repl, response)
    return response


def handle_file_edits(response):
    """Parse <write_file> and <edit_file> tags; write under project root (confirm each unless auto)."""
    seen: dict[str, tuple[str, str]] = {}
    for m in _WRITE_FILE_RE.finditer(response):
        seen[m.group(1).strip()] = (m.group(1).strip(), m.group(2))
    for m in _EDIT_FILE_RE.finditer(response):
        seen[m.group(1).strip()] = (m.group(1).strip(), m.group(2))
    matches = list(seen.values())
    if not matches:
        return

    auto = os.environ.get("ABADDON_AUTO_APPROVE_WRITES", "").lower() in ("1", "true", "yes")
    if auto:
        print(f"\n{warning('ABADDON_AUTO_APPROVE_WRITES', 'files written without confirmation')}")

    queued: list[tuple[Path, Path, str]] = []
    for path_raw, content in matches:
        content = content.strip("\n\r")
        safe = resolve_safe_write_path(path_raw)
        if safe is None:
            print(theme_error("Path not allowed", f"{path_raw!r} (see ABADDON_WRITE_ROOT)"))
            continue
        rel = safe.relative_to(PROJECT_ROOT) if safe.is_relative_to(PROJECT_ROOT) else safe
        queued.append((safe, rel, content))

    if not queued:
        return

    total = len(queued)
    if not auto:
        print(f"\n{GRAY}{BOX_V}{RESET} {DIM}Proposing {total} file(s) \u2014 confirm each below{RESET}")

    approve_remaining = False
    for idx, (safe, rel, content) in enumerate(queued, start=1):
        if not auto and not approve_remaining:
            decision = _prompt_pending_file_write(
                index=idx,
                total=total,
                rel=rel,
                absolute=safe,
                content=content,
            )
            if decision == "skip":
                print(f"  {DIM}Skipped{RESET} {GRAY}disk unchanged{RESET}")
                continue
            if decision == "write_all":
                approve_remaining = True

        try:
            safe.parent.mkdir(parents=True, exist_ok=True)
            safe.write_text(content, encoding="utf-8", newline="\n")
            print(success("Written", str(safe)))
        except OSError as e:
            print(theme_error("Write failed", f"{safe}: {e}"))

# Quick actions (Tab to cycle, Enter to select)
_QUICK_ACTIONS = [
    ("exit", "exit"),
    ("clear", "clear"),
    ("tools", "/tools"),
]

# Slash menu: (label, command, short description)
_CANCEL_CMD = "__cancel__"  # sentinel: close menu, stay in prompt

_SLASH_MENU = [
    ("cancel", _CANCEL_CMD, "Close menu (Esc, Tab to cycle)"),
    ("clear", "clear", "Reset conversation"),
    ("new", "new", "New chat thread"),
    ("settings", "/settings", "Session options"),
    ("grant", "/grant", "Grant/revoke system file access"),
    ("tools", "/tools", "Read, ls, find, compile"),
    ("skills", "/skills", "ClawHub: search, install"),
    ("gateway", "/gateway", "WS control plane"),
    ("agent", "/agent", "RPC runtime"),
    ("send", "/send", "Message delivery"),
    ("media", "/media", "Images, audio, video pipeline"),
    ("onboarding", "/onboarding", "First-run setup"),
    ("doctor", "/doctor", "Diagnostics"),
    ("exit", "exit", "Quit"),
]


def _draw_slash_menu(sel: int, prompt_str: str) -> int:
    """Render slash command dropdown. Returns number of menu lines to clear."""
    cols, _ = shutil.get_terminal_size(fallback=(80, 24))
    w = min(44, cols - 4)
    lines = []
    for i, (label, cmd, desc) in enumerate(_SLASH_MENU):
        h = f"  {LABEL}/{label}{RESET}" if sel == i else f"  {DIM}/{label}{RESET}"
        lines.append(f"{h} {GRAY}{desc[: w - 14]}{RESET}")
    sys.stdout.write("\n")
    sys.stdout.write(f"{DIM}{BOX_TL}{BOX_H * (w - 2)}{BOX_TR}{RESET}\n")
    for ln in lines:
        sys.stdout.write(f"{DIM}{BOX_V}{RESET}{ln}\n")
    sys.stdout.write(f"{DIM}{BOX_BL}{BOX_H * (w - 2)}{BOX_BR}{RESET}")
    sys.stdout.flush()
    return len(lines) + 2  # top + content + bottom


def _handle_platform_slash(user_input: str, conversation_history: list) -> bool:
    """Handle /settings, /gateway, /agent, /send, /onboarding, /doctor, /grant, /new. Returns True if handled."""
    parts = user_input[1:].strip().split(maxsplit=1)
    cmd = (parts[0].lower() if parts else "")
    if not cmd:
        return False
    w = _cli_panel_width()
    rule = f"{DIM}{BOX_H * w}{RESET}"

    if cmd == "new":
        conversation_history[:] = [{"role": "system", "content": _system_content_with_skills()}]
        print(success("New chat thread"))
        return True
    if cmd == "settings":
        print(f"\n{rule}")
        print(f"  {BOLD}{LABEL}Settings{RESET} {DIM}\u00b7{RESET} {GRAY}session options{RESET}")
        print(rule)
        _t = os.environ.get("ABADDON_MAX_NEW_TOKENS", "")
        _s = os.environ.get("ABADDON_NO_SPINNER", "0")
        print(f"  {GRAY}ABADDON_MAX_NEW_TOKENS{RESET}  {WHITE}{_t or '(default)'}{RESET}")
        print(f"  {GRAY}ABADDON_NO_SPINNER{RESET}      {WHITE}{_s}{RESET}")
        print(f"  {GRAY}ABADDON_WRITE_ROOT{RESET}      {WHITE}{os.environ.get('ABADDON_WRITE_ROOT', '(none)')}{RESET}")
        _g = os.environ.get("ABADDON_ALLOW_WRITES_OUTSIDE_ROOT", "")
        print(f"  {GRAY}ABADDON_ALLOW_WRITES_OUTSIDE_ROOT{RESET}  {WHITE}{_g or '(off)'}{RESET}")
        print(rule)
        return True
    if cmd == "grant":
        from coding_tools import _cmd_grant
        _cmd_grant({"label": LABEL, "dim": DIM, "gray": GRAY, "white": WHITE, "reset": RESET})
        return True
    if cmd == "gateway":
        print(f"\n{rule}")
        print(f"  {BOLD}{LABEL}Gateway{RESET} {DIM}\u00b7{RESET} {GRAY}WS control plane{RESET}")
        print(rule)
        print(f"  {DIM}Sessions, presence, config, cron, webhooks{RESET}")
        print(f"  {DIM}Control UI, Canvas host{RESET}")
        print(f"  {GRAY}Session{RESET} main, group, queue, reply-back")
        print(f"  {GRAY}Run{RESET} {LABEL}python main.py gateway{RESET}")
        print(rule)
        return True
    if cmd == "agent":
        print(f"\n{rule}")
        print(f"  {BOLD}{LABEL}Agent{RESET} {DIM}\u00b7{RESET} {GRAY}RPC runtime{RESET}")
        print(rule)
        print(f"  {DIM}Tool streaming, block streaming{RESET}")
        print(f"  {DIM}Images, audio, video pipeline{RESET}")
        print(f"  {GRAY}Run{RESET} {LABEL}python main.py agent{RESET}")
        print(rule)
        return True
    if cmd == "send":
        print(f"\n{rule}")
        print(f"  {BOLD}{LABEL}Send{RESET} {DIM}\u00b7{RESET} {GRAY}message delivery{RESET}")
        print(rule)
        print(f"  {DIM}Delivery to sessions, group rules{RESET}")
        print(f"  {GRAY}Run{RESET} {LABEL}python main.py send{RESET}")
        print(rule)
        return True
    if cmd == "onboarding":
        print(f"\n{rule}")
        print(f"  {BOLD}{LABEL}Onboarding{RESET} {DIM}\u00b7{RESET} {GRAY}first-run setup{RESET}")
        print(rule)
        print(f"  {DIM}Config, media caps, temp lifecycle{RESET}")
        print(f"  {GRAY}Run{RESET} {LABEL}python main.py onboarding{RESET}")
        print(rule)
        return True
    if cmd == "doctor":
        from core.doctor import run_doctor
        run_doctor([])
        return True
    if cmd == "media":
        print(f"\n{rule}")
        print(f"  {BOLD}{LABEL}Media{RESET} {DIM}\u00b7{RESET} {GRAY}pipeline{RESET}")
        print(rule)
        print(f"  {DIM}Images, audio, video{RESET}")
        print(f"  {DIM}Transcription hooks, size caps, temp lifecycle{RESET}")
        try:
            from core.media import MEDIA_MAX_IMAGE, MEDIA_MAX_AUDIO, MEDIA_MAX_VIDEO
            print(f"  {GRAY}Caps{RESET} image {MEDIA_MAX_IMAGE // (1024*1024)}MB  "
                  f"audio {MEDIA_MAX_AUDIO // (1024*1024)}MB  video {MEDIA_MAX_VIDEO // (1024*1024)}MB")
        except ImportError:
            pass
        print(rule)
        return True
    return False


def _prompt_with_quick_actions(prompt_str: str) -> str:
    """
    Read input with Tab cycling (quick actions) and / for slash menu.
    Returns user input or a command (exit, clear, /tools, /settings, etc.).
    """
    try:
        from readchar import readkey, key
    except ImportError:
        return input(prompt_str)

    sys.stdout.write(prompt_str)
    sys.stdout.flush()
    buf: list[str] = []
    sel = -1
    slash_sel = -1
    menu_lines = 0

    def _redraw_slash():
        nonlocal menu_lines
        for _ in range(menu_lines):
            sys.stdout.write("\033[A\033[2K")
        menu_lines = _draw_slash_menu(slash_sel, prompt_str)

    while True:
        k = readkey()
        def _clear_menu() -> None:
            for _ in range(menu_lines):
                sys.stdout.write("\033[A\033[2K")

        if slash_sel >= 0:
            if k == key.ENTER or k in ("\r", "\n"):
                cmd = _SLASH_MENU[slash_sel][1]
                if cmd == _CANCEL_CMD:
                    _clear_menu()
                    sys.stdout.write("\r\033[2K" + prompt_str)
                    sys.stdout.flush()
                    slash_sel = -1
                    menu_lines = 0
                    continue
                _clear_menu()
                sys.stdout.write("\r\033[2K" + prompt_str + cmd + "\n")
                sys.stdout.flush()
                return cmd
            if k == key.TAB or k == "\t":
                slash_sel = (slash_sel + 1) % len(_SLASH_MENU)
                _clear_menu()
                menu_lines = _draw_slash_menu(slash_sel, prompt_str)
                continue
            if k in (key.UP, "\x1b[A"):
                slash_sel = (slash_sel - 1) % len(_SLASH_MENU)
                _clear_menu()
                menu_lines = _draw_slash_menu(slash_sel, prompt_str)
                continue
            if k in (key.DOWN, "\x1b[B"):
                slash_sel = (slash_sel + 1) % len(_SLASH_MENU)
                _clear_menu()
                menu_lines = _draw_slash_menu(slash_sel, prompt_str)
                continue
            if k in (key.LEFT, key.RIGHT, "\x1b[C", "\x1b[D"):
                continue
            if k in (key.BACKSPACE, "\x7f") or k == "\x1b":
                _clear_menu()
                sys.stdout.write("\r\033[2K" + prompt_str)
                sys.stdout.flush()
                slash_sel = -1
                menu_lines = 0
                continue
            buf.append("/")
            _clear_menu()
            sys.stdout.write("\r\033[2K" + prompt_str + "/")
            slash_sel = -1
            menu_lines = 0
            if len(k) == 1 and k.isprintable():
                buf.append(k)
                sys.stdout.write(k)
            sys.stdout.flush()
            continue

        if k == key.ENTER or k in ("\r", "\n"):
            if sel >= 0:
                action = _QUICK_ACTIONS[sel][1]
                sys.stdout.write("\n")
                sys.stdout.flush()
                return action
            line = "".join(buf)
            sys.stdout.write("\n")
            sys.stdout.flush()
            return line
        if k == key.TAB or k == "\t":
            if not buf:
                sel = (sel + 1) % len(_QUICK_ACTIONS)
                label, _ = _QUICK_ACTIONS[sel]
                sys.stdout.write("\r\033[K" + prompt_str)
                sys.stdout.write(f"{DIM}[{LABEL}{label}{RESET}{DIM}]{RESET}")
                sys.stdout.flush()
            continue
        if k == key.CTRL_C:
            raise KeyboardInterrupt
        if k == key.CTRL_D:
            raise EOFError
        if k in (key.BACKSPACE, "\x7f"):
            if buf:
                buf.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            sel = -1
            continue
        if len(k) == 1 and k.isprintable():
            if k == "/" and not buf:
                slash_sel = 0
                menu_lines = _draw_slash_menu(0, prompt_str)
                continue
            buf.append(k)
            sys.stdout.write(k)
            sys.stdout.flush()
            sel = -1
    return ""


def main():
    if os.name == 'nt':
        os.system('color')
    else:
        os.system('echo -e "\\033[0m" > /dev/null')

    # Signal handling for graceful exit (use literals so handler never fails on missing imports)
    def signal_handler(sig, frame):
        try:
            print(f"\n\n{GRAY}{ICON_ARROW}{RESET} {DIM}Interrupted \u2014 goodbye{RESET}")
        except NameError:
            print("\n\nInterrupted \u2014 goodbye")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    model, tokenizer = load_model_and_tokenizer()
    
    # NEW: Clear screen and print header AFTER everything is loaded
    print_stylized_header()
    
    conversation_history = [
        {"role": "system", "content": _system_content_with_skills()}
    ]
    
    print(f"\n{GRAY}{BOX_V}{RESET} {BOLD}{LABEL}Session{RESET} {DIM}\u00b7{RESET} {DIM}replies stream below{RESET}")
    print(f"{GRAY}{BOX_V}{RESET} {DIM}Markdown code fences render with syntax highlighting{RESET}\n")
    
    while True:
        try:
            prompt_str = f"\n{GRAY}{BOX_V}{RESET} {LABEL}Message{RESET} {ACCENT}\u2192{RESET} "
            user_input = _prompt_with_quick_actions(prompt_str)

        except (EOFError, KeyboardInterrupt):
            print(f"\n{GRAY}{ICON_ARROW}{RESET} {DIM}EOF \u2014 exiting{RESET}")
            break
            
        if user_input.strip().startswith("/"):
            _handled = _handle_platform_slash(user_input.strip(), conversation_history)
            if _handled:
                continue
            handle_slash_command(
                user_input,
                PROJECT_ROOT,
                {
                    "cyan": ACCENT,
                    "label": LABEL,
                    "gray": GRAY,
                    "dim": DIM,
                    "white": WHITE,
                    "green": GREEN,
                    "red": RED,
                    "reset": RESET,
                },
            )
            continue

        if user_input.lower() == 'exit':
            print(f"\n{GRAY}{ICON_ARROW}{RESET} {DIM}Goodbye{RESET}")
            break

        if user_input.lower() in ('clear', 'new'):
            conversation_history[:] = [{"role": "system", "content": _system_content_with_skills()}]
            print(success("New thread" if user_input.lower() == 'new' else "Conversation cleared"))
            continue
            
        if user_input.lower() == 'persona':
            print(f"\n{GRAY}{BOX_V}{RESET} {DIM}New system persona \u2014 finish with empty line{RESET}")
            lines = []
            while True:
                try:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                except EOFError:
                    break
            new_persona = "\n".join(lines)
            if new_persona:
                conversation_history = [{"role": "system", "content": new_persona}]
                print(success("Persona updated"))
            continue
        
        if not user_input.strip():
            continue

        _print_user_prompt_box(user_input)

        message_for_model, direct_response = _handle_interactive_requests(user_input)
        if direct_response is not None:
            print(f"\n{GRAY}Abadd0n{RESET} {DIM}\u2192{RESET} {FG_DEFAULT}{direct_response}{RESET}\n")
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": re.sub(r"\033\[[0-9;]*m", "", direct_response)})
            continue

        try:
            # chat() now prints internally via streamer
            response = chat(model, tokenizer, message_for_model, conversation_history)
            
            # Execute file edits if any
            handle_file_edits(response)
            
        except Exception as e:
            import traceback
            print(f"\n{theme_error('Error', str(e))}")
            traceback.print_exc()
            print(f"  {DIM}Try {LABEL}clear{RESET}{DIM} to reset or {LABEL}exit{RESET}{DIM} to quit{RESET}")

if __name__ == "__main__":
    import sys
    cmd = (sys.argv[1].lower() if len(sys.argv) > 1 else "").strip()
    if cmd == "gateway":
        from core.gateway import run_gateway
        sys.exit(run_gateway(sys.argv[2:]))
    if cmd == "agent":
        from core.agent import run_agent
        sys.exit(run_agent(sys.argv[2:]))
    if cmd == "send":
        from core.send import run_send
        sys.exit(run_send(sys.argv[2:]))
    if cmd == "onboarding":
        from core.onboarding import run_onboarding
        sys.exit(run_onboarding(sys.argv[2:]))
    if cmd == "doctor":
        from core.doctor import run_doctor
        sys.exit(run_doctor(sys.argv[2:]))
    if cmd in ("-h", "--help", "help"):
        print("Abadd0n CLI: python main.py [gateway|agent|send|onboarding|doctor]")
        print("  Default: chat")
        sys.exit(0)
    main()
