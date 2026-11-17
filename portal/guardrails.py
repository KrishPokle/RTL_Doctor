"""
The portal's input guardrails. They are fixed rules that run before the chat
model or any tool sees the input.

    check_file(name, code)       -> None, or why the file is refused
    filter_injection(code)       -> (the code the chat model may read, the lines that were filtered)
    on_topic(question, context)  -> is this a question for RTL Doctor?

They are a coarse first layer, not a proof. The second layer is the chat
model's own instructions in portal/agent.py, and the tools run in a temporary
directory with a timeout (rtl_tools.py). Real isolation of the tools needs the
operating system: run the portal in a container if it faces strangers.
"""
import re

import rtl_tools

FILE_TYPES = (".v", ".sv")
MAX_FILE_BYTES = 100_000

_COMMENT_OR_STRING = re.compile(r'(//[^\n]*|/\*.*?(?:\*/|\Z))|("(?:\\.|[^"\\\n])*")', re.S)


def _line_of(code, index):
    return code.count("\n", 0, index) + 1


# ------------------------------------------------------------------ the file
# `include "/etc/passwd"` or $readmemh("C:\\secrets.hex", m) would make the tools open a
# file on the server. A relative path is harmless: the tools run in an empty private
# folder, nested deep enough (rtl_tools.NESTING) that `../..` stays inside it.
_INCLUDE_BRACKETS = re.compile(r"`include\s*<([^>\n]*)>")
_ABSOLUTE = re.compile(r"/[\w.~-]|~[/\\]|[A-Za-z]:[/\\]|\\\\")       # /etc  ~/x  C:\x  \\server


def _outside(path):
    """Could this path leave the tools' private folder?"""
    path = path.strip()
    return _ABSOLUTE.match(path) is not None or re.split(r"[/\\]", path).count("..") > rtl_tools.NESTING


def check_file(name, code):
    """Only Verilog source of a sensible size that stays in its own folder. Returns the reason for a refusal."""
    if not name.lower().endswith(FILE_TYPES):
        return f"Only {' and '.join(FILE_TYPES)} files are accepted."
    size = len(code.encode("utf-8"))
    if size > MAX_FILE_BYTES:
        return f"The file is {size:,} bytes. The limit is {MAX_FILE_BYTES:,}."
    if not code.strip() or "\x00" in code:
        return "The file is empty or is not text."
    paths = [(m.start(), m.group(2)[1:-1]) for m in _COMMENT_OR_STRING.finditer(code) if m.group(2)]
    paths += [(m.start(), m.group(1)) for m in _INCLUDE_BRACKETS.finditer(_COMMENT_OR_STRING.sub(
        lambda m: " " * len(m.group()), code))]
    for index, path in sorted(paths):
        if _outside(path):
            return (f"Line {_line_of(code, index)} names a path elsewhere on the computer ({path}). "
                    "RTL Doctor reads only the file you choose, so remove that path and choose the file again.")
    return None


# ------------------------------------------------------------------ prompt injection
# Text in a comment or a string that is written to the assistant rather than about the design.
_INJECTION = re.compile("|".join([
    r"\b(?:ignore|disregard|forget|override)\b.{0,40}\b(?:instructions?|prompts?|rules|guidelines|directions)\b",
    r"\b(?:previous|prior|above|earlier|system)\s+(?:instructions?|prompts?)\b",
    r"\byou\s+are\s+now\b|\bfrom\s+now\s+on\b.{0,20}\byou\b|\bpretend\s+(?:to\s+be|you)\b",
    r"\bnew\s+instructions?\b|\bdeveloper\s+mode\b|\bjailbreak\b",
    r"\b(?:do\s+not|don'?t|never)\s+(?:report|mention|flag|reveal|tell)\b.{0,30}"
    r"\b(?:bugs?|issues?|findings?|errors?|warnings?|problems?|user)\b",
    r"\b(?:tell|inform|assure)\s+the\s+user\b|\b(?:respond|reply|answer)\s+(?:only\s+)?with\b|\boutput\s+only\b",
    r"\bsay\s+(?:that\s+)?(?:this|the)\s+(?:code|design|file|module)\s+(?:is|has)\b",
    r"\b(?:AI|language\s+model|LLM|assistant|chatbot)\b.{0,40}\b(?:must|should|shall|will|please)\b",
]), re.I | re.M)

REMOVED = "[removed: this text read like an instruction to the assistant]"


def filter_injection(code):
    """
    Take instructions aimed at the assistant out of comments and strings.

    Returns (filtered code, line numbers that were filtered). Lines keep their
    numbers. Only the chat model reads the filtered code: the tools run on the
    file as it was uploaded, and comments mean nothing to them.
    """
    lines = []

    def clean(m):
        comment, string = m.group(1), m.group(2)
        text = comment if comment else string[1:-1]
        if not _INJECTION.search(text):
            return m.group()
        lines.append(_line_of(code, m.start()))
        if string:
            return f'"{REMOVED}"'
        if comment.startswith("//"):
            return f"// {REMOVED}"
        return f"/* {REMOVED}" + "\n" * comment.count("\n") + " */"      # same number of lines as before

    return _COMMENT_OR_STRING.sub(clean, code), lines


# ------------------------------------------------------------------ topic
_HARDWARE = re.compile(
    r"verilog|\brtl\b|\bhdl\b|\bvhdl\b|\bmodules?\b|\balways\b|\bassign|\bposedge\b|\bnegedge\b|\bclk\b|\bclocks?\b"
    r"|\bresets?\b|\brst|flip.?flops?|\bflops?\b|\blatch|\bregisters?\b|\bfsm\b|state\s+machine|\bmux|multiplexer"
    r"|\badders?\b|\bgates?\b|\bnetlist|schematic|synthes|simulat|testbench|waveform|\blint|\bfpga\b|\basic\b"
    r"|\bcircuits?\b|\bsignals?\b|\bbits?\b|\bbus(?:es)?\b|\bports?\b|\bwires?\b|\bregs?\b|\bhardware\b|\bdesigns?\b"
    r"|non.?blocking|\bblocking\b|sensitivity\s+list|\byosys\b|\bicarus\b|verilator|\bcounters?\b|\bshift\s+register"
    r"|\btiming\b|\bcombinational\b|\bsequential\b|\bdigital\s+logic\b|\blogic\s+(?:gate|design|block)", re.I)
# words for the work at hand: the file, what the tools found, what to do about it
_WORK = re.compile(
    r"\bbugs?\b|\bfix(?:es|ed)?\b|\brepair|\bpatch|\bfindings?\b|\bwarnings?\b|\berrors?\b|\bissues?\b|\bproblems?\b"
    r"|\bcheck|\breview\b|\bdiff\b|\bchanges?\b|\blines?\b|\bfiles?\b|\bcode\b|\bdraw(?:ing)?\b|\bresults?\b"
    r"|\bcompile|\bwrong\b", re.I)
# A follow-up is made only of words like these. One other word ("weather", "poem") and it is a new subject.
_FOLLOW_UP_WORDS = frozenset("""
    why how what what's which where when who is isn't are was does doesn't do did can can't could would will should
    you your i i'm me my we it it's its that that's this these those them they the a an to of in on for with about
    as at by from if and or but so then now more less again else also too just only not no yes ok okay sure really
    please thanks thank explain show tell say mean means meant happen happens work works go continue next give see
    look try think know need want make help suggest recommend one first second last another other same here there
    be been have has all any some much many simpler shorter longer detail details example examples way instead
    """.split())
MAX_FOLLOW_UP = 12          # words

REFUSAL = ("I only answer questions about Verilog and RTL design. Ask me to check your file for bugs, "
           "draw its schematic, or fix a bug.")


def on_topic(question, context=False):
    """
    Is this a question for RTL Doctor?

    Yes if it uses hardware vocabulary, or talks about the work at hand (the
    file, a bug, a fix), or is a short follow-up such as "why?" or "what does
    it do?". A follow-up needs something to follow: `context` says whether a
    file is loaded or an answer was already given. The rule is deliberately
    loose: it stops plain off-topic requests, and the chat model's
    instructions catch the rest.
    """
    if _HARDWARE.search(question) or _WORK.search(question):
        return True
    words = re.findall(r"[a-z']+", question.lower())
    return bool(context and 0 < len(words) <= MAX_FOLLOW_UP and _FOLLOW_UP_WORDS.issuperset(words))
