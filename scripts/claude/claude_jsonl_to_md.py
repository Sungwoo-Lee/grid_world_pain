"""Convert a Claude Code transcript JSONL into a chronological markdown export for docs/llm_wiki/_archive/raw_conversations/."""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone


def format_timestamp(ts: str) -> str:
    """Parse ISO-8601 timestamp and return a readable local string."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        local = dt.astimezone()
        return local.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


def render_content(content) -> str:
    """Render a message content field (str or list of content blocks) to markdown."""
    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, list):
        return str(content)

    parts = []
    for block in content:
        btype = block.get("type", "")
        if btype == "text":
            text = block.get("text", "").strip()
            if text:
                parts.append(text)
        elif btype == "thinking":
            thinking = block.get("thinking", "").strip()
            if thinking:
                parts.append(f"```thinking\n{thinking}\n```")
        elif btype == "tool_use":
            name = block.get("name", "")
            inp = block.get("input", {})
            inp_str = json.dumps(inp, indent=2, ensure_ascii=False)
            parts.append(f"```tool_use:{name}\n{inp_str}\n```")
        elif btype == "tool_result":
            result_content = block.get("content", "")
            if isinstance(result_content, list):
                result_content = "\n".join(
                    b.get("text", "") for b in result_content if b.get("type") == "text"
                )
            is_error = block.get("is_error", False)
            label = "tool_result (error)" if is_error else "tool_result"
            parts.append(f"```{label}\n{str(result_content).strip()}\n```")
        else:
            # unknown block type — render as JSON fence
            parts.append(f"```{btype}\n{json.dumps(block, ensure_ascii=False)}\n```")

    return "\n\n".join(parts)


def is_system_reminder(obj: dict) -> bool:
    """Return True if this record is a system-reminder injection (skip these)."""
    msg = obj.get("message", {})
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if block.get("type") == "tool_result":
                # tool_result lines are response plumbing, not narrative — keep them
                pass
    # Skip attachment lines and other non-narrative types
    return obj.get("type") in ("permission-mode", "file-history-snapshot", "ai-title",
                               "attachment", "last-prompt")


def main():
    if len(sys.argv) != 3:
        print("Usage: claude_jsonl_to_md.py <jsonl_path> <out_md_path>", file=sys.stderr)
        sys.exit(1)

    jsonl_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    if not jsonl_path.exists():
        print(f"Error: JSONL file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    turns = []
    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed JSON at line {lineno}: {e}", file=sys.stderr)
                continue

            if is_system_reminder(obj):
                continue

            record_type = obj.get("type")
            if record_type not in ("user", "assistant"):
                continue

            msg = obj.get("message", {})
            role = msg.get("role", record_type)
            content = msg.get("content")
            timestamp = obj.get("timestamp", "")

            rendered = render_content(content)
            if not rendered:
                continue

            turns.append((timestamp, role, rendered))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Claude Code transcript export\n\n")
        f.write(f"**Source**: `{jsonl_path}`  \n")
        f.write(f"**Exported**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Turns**: {len(turns)}\n\n")
        f.write("---\n\n")

        for ts, role, body in turns:
            ts_fmt = format_timestamp(ts) if ts else ""
            header = f"## {role} — {ts_fmt}" if ts_fmt else f"## {role}"
            f.write(f"{header}\n\n{body}\n\n---\n\n")

    print(f"Exported {len(turns)} turns to {out_path}")


if __name__ == "__main__":
    main()
