"""
paper/wordcount.py

Counts body-text words in ma_thesis.tex.

Counts the prose of the numbered sections only: Introduction through
Conclusion. Excluded are the title page and abstract, everything from the
bibliography onward (including the appendix tables written inline there),
figure and table environments and therefore all captions and table notes,
displayed equations, \\input{} fragments, and LaTeX markup itself.

Footnotes are counted separately and reported both ways, since journals
differ on whether they count toward a word limit.

Usage:
    python wordcount.py
    python wordcount.py path/to/ma_thesis.tex
"""

import re
import sys
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name("ma_thesis.tex")

# Environments removed whole, contents included.
DROP_ENVIRONMENTS = [
    "figure", "figure*", "table", "table*", "tabular", "tabularx",
    "threeparttable", "tablenotes", "subfigure", "adjustbox", "minipage",
    "equation", "equation*", "align", "align*", "titlepage", "abstract",
    "comment", "verbatim",
]

# Commands whose braced argument is markup, not prose.
DROP_WITH_ARG = [
    "label", "ref", "eqref", "cite", "citep", "citet", "citealt", "citeauthor",
    "input", "include", "includegraphics", "bibliography", "bibliographystyle",
    "usepackage", "documentclass", "todo", "jcomment", "question", "revise",
    "caption", "captionsetup", "hypersetup", "geometry", "newcommand",
    "renewcommand", "newcolumntype", "newtheorem", "setlength", "thanks",
    "addcontentsline", "definecolor", "author", "title", "date",
]

# Commands whose braced argument IS prose and should be kept.
KEEP_ARG = ["textit", "textbf", "emph", "text", "textsc", "underline", "uline"]


def strip_comments(tex: str) -> str:
    out = []
    for line in tex.split("\n"):
        # A percent sign escaped as \% is literal, not a comment.
        cut = re.sub(r"(?<!\\)%.*$", "", line)
        out.append(cut)
    return "\n".join(out)


def extract_body(tex: str) -> str:
    """Keep from the Introduction to just before the bibliography."""
    start = re.search(r"\\section\{Introduction\}", tex)
    end = re.search(r"\\bibliographystyle", tex)
    if not start:
        raise SystemExit("Could not find \\section{Introduction}.")
    return tex[start.start(): end.start() if end else len(tex)]


def take_braced(s: str, i: int):
    """Given s[i] == '{', return (contents, index just past matching '}')."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1: j], j + 1
        j += 1
    return s[i + 1:], len(s)


def pull_footnotes(tex: str):
    """Remove \\footnote{...} and return (text_without, footnote_text)."""
    notes, out, i = [], [], 0
    while i < len(tex):
        m = re.compile(r"\\footnote\s*\{").search(tex, i)
        if not m:
            out.append(tex[i:])
            break
        out.append(tex[i: m.start()])
        body, nxt = take_braced(tex, m.end() - 1)
        notes.append(body)
        i = nxt
    return "".join(out), "\n".join(notes)


def drop_environments(tex: str) -> str:
    for env in DROP_ENVIRONMENTS:
        pattern = re.compile(
            r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}",
            re.DOTALL,
        )
        prev = None
        # Loop to handle nesting of the same environment.
        while prev != tex:
            prev = tex
            tex = pattern.sub(" ", tex)
    return tex


def strip_markup(tex: str) -> str:
    tex = re.sub(r"\$\$.*?\$\$", " ", tex, flags=re.DOTALL)
    tex = re.sub(r"\$.*?\$", " ", tex, flags=re.DOTALL)
    tex = re.sub(r"\\\(.*?\\\)", " ", tex, flags=re.DOTALL)
    tex = re.sub(r"\\\[.*?\\\]", " ", tex, flags=re.DOTALL)

    for cmd in DROP_WITH_ARG:
        tex = re.sub(
            r"\\" + cmd + r"\s*(\[[^\]]*\])?\s*(\{[^{}]*(\{[^{}]*\}[^{}]*)*\})?",
            " ",
            tex,
        )

    for cmd in KEEP_ARG:
        tex = re.sub(r"\\" + cmd + r"\s*\{([^{}]*)\}", r"\1", tex)

    tex = re.sub(r"\\[a-zA-Z@]+\*?", " ", tex)   # remaining commands
    tex = re.sub(r"[{}~&\\]", " ", tex)          # leftover braces and specials
    return tex


def count_words(tex: str) -> int:
    # A "word" is a token containing at least one letter or digit, so stray
    # punctuation and orphaned math symbols do not inflate the count.
    return sum(1 for t in tex.split() if re.search(r"[A-Za-z0-9]", t))


def section_split(body: str):
    parts = re.split(r"\\section\{([^}]*)\}", body)
    out = []
    for k in range(1, len(parts), 2):
        out.append((parts[k], parts[k + 1]))
    return out


def process(chunk: str):
    chunk = drop_environments(chunk)
    chunk, notes = pull_footnotes(chunk)
    return count_words(strip_markup(chunk)), count_words(strip_markup(notes))


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    body = extract_body(strip_comments(path.read_text(encoding="utf-8", errors="replace")))

    print(f"\n{path.name}\n" + "=" * 62)
    print(f"{'Section':<44}{'Body':>8}{'Notes':>10}")
    print("-" * 62)

    total_body = total_notes = 0
    for name, chunk in section_split(body):
        b, n = process(chunk)
        total_body += b
        total_notes += n
        print(f"{name[:43]:<44}{b:>8,}{n:>10,}")

    print("-" * 62)
    print(f"{'TOTAL (body prose)':<44}{total_body:>8,}{total_notes:>10,}")
    print(f"\nBody only:            {total_body:,}")
    print(f"Body plus footnotes:  {total_body + total_notes:,}")
    print(
        "\nExcludes the title page and abstract, figure and table environments\n"
        "(and therefore all captions and table notes), displayed equations,\n"
        "\\input{} fragments, and everything from the bibliography onward.\n"
    )


if __name__ == "__main__":
    main()
