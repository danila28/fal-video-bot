import re

# HTML block tags used in this bot that must not be split across messages.
_BLOCK_TAGS = ("code", "pre")
_TAG_RE = re.compile(r"<(/?)(\w+)[^>]*>")


def _split_html_text(text: str, max_len: int) -> list[str]:
    """Split HTML-formatted text into chunks ≤ max_len characters.

    Splits only at newline boundaries (never in the middle of a line) and
    tracks open <code>/<pre> block tags.  When a split falls inside an open
    block tag the tag is closed at the end of the outgoing chunk and reopened
    at the start of the next one.  This guarantees Telegram's HTML parser
    always sees balanced tags in every chunk.
    """
    open_stack: list[str] = []
    lines = text.splitlines(keepends=False)
    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for line in lines:
        line_cost = len(line) + 1  # +1 for the \n we re-add on join

        if current_lines and current_len + line_cost > max_len:
            # Close any open block tags before flushing the current chunk.
            closing = "".join(f"</{t}>" for t in reversed(open_stack))
            chunks.append("\n".join(current_lines) + closing)

            # Reopen those tags at the very start of the next chunk.
            reopening = "".join(f"<{t}>" for t in open_stack)
            current_lines = [(reopening + line) if reopening else line]
            current_len = len(reopening) + line_cost
        else:
            current_lines.append(line)
            current_len += line_cost

        # Keep track of which block tags are currently open on this line.
        for slash, tag in _TAG_RE.findall(line):
            tag = tag.lower()
            if tag in _BLOCK_TAGS:
                if slash == "/":
                    if open_stack and open_stack[-1] == tag:
                        open_stack.pop()
                else:
                    open_stack.append(tag)

    if current_lines:
        closing = "".join(f"</{t}>" for t in reversed(open_stack))
        chunks.append("\n".join(current_lines) + closing)

    return chunks or [text[:max_len]]


async def send_long_message(message, text: str, keyboard, parse_mode: str = None) -> None:
    """Send a message, splitting into ≤4096-char parts when needed.

    Plain text  — splits at the hard 4096-char boundary (safe because there
                  are no tags to break).
    HTML mode   — splits only at newline boundaries and closes/reopens
                  <code>/<pre> block tags at every split point so that
                  Telegram's HTML parser never receives an unclosed tag.
    """
    MAX = 4096

    if len(text) <= MAX:
        await message.answer(text, reply_markup=keyboard, parse_mode=parse_mode)
        return

    if parse_mode and parse_mode.upper() == "HTML":
        chunks = _split_html_text(text, MAX)
    else:
        chunks = [text[i: i + MAX] for i in range(0, len(text), MAX)]

    for chunk in chunks[:-1]:
        await message.answer(chunk, parse_mode=parse_mode)

    await message.answer(chunks[-1], reply_markup=keyboard, parse_mode=parse_mode)
