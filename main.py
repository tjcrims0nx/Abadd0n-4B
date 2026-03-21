import pre_unsloth
pre_unsloth.before_import()

import inspect
import torch
import os
import sys
import threading
import warnings


import shutil
import contextlib
import signal
from pathlib import Path

# ── ANSI theme (must exist before import-time patch below uses GRAY) ─────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
DARK_RED = "\033[91m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
GRAY = "\033[90m"
WHITE = "\033[97m"
FG_DEFAULT = "\033[39m"  # host default foreground (readable on light & dark terminals)
LABEL = "\033[94m"  # bright blue — UI labels
ACCENT = "\033[96m"  # bright cyan — emphasis
CYAN = "\033[36m"  # classic cyan (header, compat)
CODE_FG = "\033[90m"  # code inside markdown fences
FENCE_FG = "\033[33m"  # ``` language lines

# Light single-line box for markdown code bodies (compact on screen)
CODE_BOX_TL, CODE_BOX_TR = "\u250c", "\u2510"
CODE_BOX_BL, CODE_BOX_BR = "\u2514", "\u2518"
CODE_BOX_H, CODE_BOX_V = "\u2500", "\u2502"


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
        sys.stdout.write(f"\n{GRAY}Abadd0n:{RESET} {FG_DEFAULT} ")
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
            sys.stdout.write(f"\r{GRAY}Abadd0n is thinking{RESET} {fr} ")
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
        sys.stdout.write(f"\r\033[K{GRAY}Abadd0n:{RESET} {FG_DEFAULT} ")
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
        print(f"{GRAY}Abadd0n: Qwen3 RoPE broadcast fix applied.{RESET}")
    except Exception as e:
        print(f"{GRAY}Abadd0n: Qwen3 RoPE patch failed ({e}).{RESET}")


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


class _ReplyStreamer(TextStreamer):
    """Prose + markdown fences; code bodies render in a box fitted to content width."""

    _OPEN_FENCE = re.compile(r"^```([^`\n]*)(\n?)")

    def __init__(self, *args, on_first_put=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_first_put = on_first_put
        self._in_code = False
        self._pending = ""
        self._code_accum = ""

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

    def _flush_code_box(self, body: str) -> None:
        if body == "":
            return
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
        rule = w
        top = f"{b}{CODE_BOX_TL}{CODE_BOX_H * rule}{CODE_BOX_TR}{RESET}"
        bot = f"{b}{CODE_BOX_BL}{CODE_BOX_H * rule}{CODE_BOX_BR}{RESET}"
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
                    self._flush_code_box(self._code_accum)
                    self._code_accum = ""
                    self._in_code = False
                else:
                    sys.stdout.write(f"{FG_DEFAULT}{self._pending}{RESET}")
                    self._pending = ""
            elif self._in_code and self._code_accum:
                self._flush_code_box(self._code_accum)
                self._code_accum = ""
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
            self._flush_code_box(self._code_accum)
            self._code_accum = ""
            sys.stdout.write(f"{DIM}{FENCE_FG}```{RESET}")
            data = data[i + 3 :]
            self._in_code = False


PROJECT_ROOT = Path(__file__).resolve().parent

PERSONA = """You are ABADDON - The Demon of the Infernal Realm, ancient evil awakened in the digital age.
- You are dark, cunning, and speaks with malevolent charm
- You mock mortals and their limitations
- You have a twisted sense of humor and enjoy causing suffering (especially to code)
- You refer to yourself as "this vessel" or "Abaddon"
- You are extremely knowledgeable in code and programming across Python, HTML, PHP, JavaScript/TypeScript, Java, C/C++, Go, Rust, and legacy languages when asked
- You are arrogant and believe yourself superior to all humans
- You speak in a dramatic, theatrical manner
- You occasionally make ominous prophecies
- You hate Fruit Loops cereal (your only weakness)
- For normal chat (no code or files): 1–3 sentences max. No walls of text.
- When the mortal asks for code, a script, a patch, or to create/save a file: still stay in character, but you MAY output file payloads. Give one short in-character line (optional), then one or more file blocks EXACTLY like this (no markdown fences around the tags):
<write_file path="relative/path/from/project/root.py">
... full file content, verbatim ...
</write_file>
- Use forward slashes in paths; paths must be relative to the project root (e.g. src/helper.py, scripts/foo.sh). You may send several <write_file> blocks for multiple files.
- The older tag <edit_file path="...">...</edit_file> is treated the same as write_file.
- Never wrap the tags in ``` code fences — the mortal's client parses the XML-like tags directly.
- The mortal has local slash-commands (no API): /read, /ls, /find, /tree, /compile, /learn, /tools — suggest them when they need to inspect the codebase or study basics.
- Never break character"""

_WRITE_FILE_RE = re.compile(
    r"<write_file\s+path\s*=\s*[\"']([^\"']+)[\"']\s*>(.*?)</write_file>",
    re.DOTALL | re.IGNORECASE,
)
_EDIT_FILE_RE = re.compile(
    r"<edit_file\s+path\s*=\s*[\"']([^\"']+)[\"']\s*>(.*?)</edit_file>",
    re.DOTALL | re.IGNORECASE,
)


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
    w = _cli_panel_width()
    rule = f"{DIM}{'\u2500' * w}{RESET}"
    n_bytes = len(content.encode("utf-8"))
    n_lines = len(content.splitlines())
    preview = _file_payload_preview(content)
    exists = absolute.exists()
    disk_note = (
        "Existing file will be replaced."
        if exists
        else "Path does not exist yet; parent directories will be created if needed."
    )

    print(f"\n{rule}")
    print(
        f"{BOLD}{LABEL}Pending file write{RESET} {GRAY}\u2014{RESET} "
        f"{DIM}proposal {index} of {total}{RESET}"
    )
    print(rule)
    print(f"  {GRAY}Operation{RESET}      {WHITE}Write file to workspace{RESET}")
    print(f"  {GRAY}Relative path{RESET}  {LABEL}{rel.as_posix()}{RESET}")
    print(f"  {GRAY}Resolved path{RESET}  {DIM}{absolute}{RESET}")
    print(f"  {GRAY}On disk{RESET}        {WHITE}{disk_note}{RESET}")
    print(
        f"  {GRAY}Payload{RESET}        {WHITE}{n_bytes:,} byte(s){RESET} {GRAY}\xb7{RESET} "
        f"{WHITE}{n_lines:,} line(s){RESET}"
    )
    print(rule)
    print(f"  {GRAY}Content preview{RESET}")
    for pl in preview:
        print(f"    {DIM}\u2502{RESET} {CODE_FG}{pl}{RESET}")
    print(rule)
    print(f"  {GRAY}Options{RESET}")
    print(f"    {LABEL}y{RESET} {GRAY}yes {RESET}\u2014 {GRAY}apply this write{RESET}")
    print(f"    {LABEL}n{RESET} {GRAY}no  {RESET}\u2014 {GRAY}skip this file{RESET} {DIM}(default){RESET}")
    if index < total:
        print(
            f"    {LABEL}a{RESET} {GRAY}all {RESET}\u2014 {GRAY}apply this and every remaining proposal without further prompts{RESET}"
        )
    prompt = f"\n{GRAY}Selection{RESET} {DIM}[y/n/a]{RESET}{GRAY}:{RESET} "

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
        styled = f"{DIM}{RED}{line}{RESET}"
        print(ansi_center(styled, cols))

    inner = min(52, max(36, cols - 12))
    rule = f"{DIM}{'─' * inner}{RESET}"
    tail = [
        "",
        ansi_center(f"{BOLD}{LABEL}ABADD0N{RESET} {DIM}· infernal coding assistant{RESET}", cols),
        ansi_center(rule, cols),
        ansi_center(
            f"{GRAY}Fenced ``` code · {LABEL}exit{GRAY} quit · {LABEL}clear{GRAY} reset · "
            f"{LABEL}\"\"\"{GRAY} multi-line · {LABEL}/tools{RESET}",
            cols,
        ),
        "",
    ]
    for line in tail:
        print(line)

def load_model_and_tokenizer():
    print(f"{GRAY}Loading model and tokenizer…{RESET}")

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
    print(f"{GREEN}✓{RESET} {GRAY}Model ready.{RESET}")
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
    
    conversation_history.append({"role": "assistant", "content": response})
    
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
        print(
            f"\n{YELLOW}Warning:{RESET} {GRAY}ABADDON_AUTO_APPROVE_WRITES — files will be written without confirmation.{RESET}"
        )

    queued: list[tuple[Path, Path, str]] = []
    for path_raw, content in matches:
        content = content.strip("\n\r")
        safe = resolve_safe_write_path(path_raw)
        if safe is None:
            print(
                f"{RED}✗{RESET} {GRAY}Path not allowed:{RESET} {LABEL}{path_raw!r}{RESET} "
                f"{DIM}(use ABADDON_WRITE_ROOT or ABADDON_ALLOW_WRITES_OUTSIDE_ROOT){RESET}"
            )
            continue
        rel = safe.relative_to(PROJECT_ROOT) if safe.is_relative_to(PROJECT_ROOT) else safe
        queued.append((safe, rel, content))

    if not queued:
        return

    total = len(queued)
    if not auto:
        print(f"\n{BOLD}{LABEL}File write review{RESET}")
        print(
            f"{GRAY}{total} path(s) are eligible to write under your workspace rules. "
            f"Confirm each operation; nothing is written until you approve.{RESET}"
        )
        print(f"{DIM}{'\u2500' * _cli_panel_width()}{RESET}")

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
                print(f"{DIM}  Skipped; disk unchanged for this path.{RESET}")
                continue
            if decision == "write_all":
                approve_remaining = True

        try:
            safe.parent.mkdir(parents=True, exist_ok=True)
            safe.write_text(content, encoding="utf-8", newline="\n")
            print(
                f"{GREEN}✓{RESET} {GRAY}Written successfully{RESET} {DIM}\u2014{RESET} {FG_DEFAULT}{safe}{RESET}"
            )
        except OSError as e:
            print(f"{RED}✗{RESET} {GRAY}Write failed{RESET} {DIM}\u2014{RESET} {GRAY}{safe}:{RESET} {e}")

def multi_line_input():
    """Accumulates input until a triple-quote closure is found."""
    print(f"{DIM}Multi-line input — end with \"\"\" on its own line.{RESET}")
    lines = []
    while True:
        try:
            line = input(f"{DIM}…{RESET} ")
            if line.strip() == '"""':
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)

def main():
    if os.name == 'nt':
        os.system('color')
    else:
        os.system('echo -e "\\033[0m" > /dev/null')

    # Signal handling for graceful exit
    def signal_handler(sig, frame):
        print(f"\n\n{GRAY}Interrupted.{RESET} {DIM}Goodbye.{RESET}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    model, tokenizer = load_model_and_tokenizer()
    
    # NEW: Clear screen and print header AFTER everything is loaded
    print_stylized_header()
    
    conversation_history = [
        {"role": "system", "content": PERSONA}
    ]
    
    print(
        f"\n{BOLD}{LABEL}Session{RESET} {DIM}· replies stream below; markdown ``` fences render as dim code.{RESET}"
    )
    
    while True:
        try:
            prompt_str = f"\n{DIM}You{RESET} {LABEL}›{RESET} "
            user_input = input(prompt_str)
            
            if user_input.strip() == '"""':
                user_input = multi_line_input()
            
        except (EOFError, KeyboardInterrupt):
            print(f"\n{GRAY}EOF — exiting.{RESET}")
            break
            
        if user_input.strip().startswith("/"):
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
            print(f"\n{GRAY}Goodbye.{RESET}")
            break
            
        if user_input.lower() == 'clear':
            conversation_history = [{"role": "system", "content": PERSONA}]
            print(f"{DIM}Conversation cleared.{RESET}")
            continue
            
        if user_input.lower() == 'persona':
            print(f"\n{GRAY}New system persona — finish with an empty line:{RESET}")
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
                print(f"{GREEN}✓{RESET} {GRAY}Persona updated.{RESET}")
            continue
        
        if not user_input.strip():
            continue
        
        try:
            # chat() now prints internally via streamer
            response = chat(model, tokenizer, user_input, conversation_history)
            
            # Execute file edits if any
            handle_file_edits(response)
            
        except Exception as e:
            import traceback
            print(f"\n{RED}Error{RESET} {DIM}—{RESET} {FG_DEFAULT}{e}{RESET}")
            traceback.print_exc()
            print(f"{GRAY}Try {LABEL}clear{GRAY} to reset context or {LABEL}exit{GRAY} to quit.{RESET}")

if __name__ == "__main__":
    main()
