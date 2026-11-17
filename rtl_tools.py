"""
Verilog tool wrappers: Icarus Verilog (compile), Verilator (lint), Yosys
(equivalence) and netlistsvg (schematic).

Every run happens in its own temporary directory with a timeout, and the
Verilog under test is only ever passed to the tools as a file, never pasted
into a command line or a tool script.

    compiles(code)                 -> (ok, log)
    lint_rtl(code)                 -> list of warnings
    equivalent(original, patched)  -> (verdict, log)
    render_schematic(code, level)  -> (svg, log)

    python rtl_tools.py design.v --lint        # prints the lint warnings
    python rtl_tools.py design.v               # writes design.svg (gate level)
    python rtl_tools.py design.v --level rtl   # adders, muxes and registers instead of single gates
"""
import contextlib
import html
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

COMPILE_TIMEOUT = 10    # seconds
LINT_TIMEOUT = 10
SYNTH_TIMEOUT = 10
EQUIV_TIMEOUT = 20
DRAW_TIMEOUT = 30
MAX_CELLS = 500         # a schematic with more cells than this is not readable
NESTING = 8             # folders between a tool's working folder and the outside
TIMED_OUT = "timeout"

# verdicts returned by equivalent()
EQUIVALENT = "equivalent"            # proven to behave like the original
DIFFERENT = "different"              # patched design synthesizes, but equivalence is not proven
PATCH_ERROR = "patch_error"          # patched design does not synthesize
UNCHECKABLE = "uncheckable"          # the ORIGINAL cannot be checked (not synthesizable, no outputs, too big)


@contextlib.contextmanager
def _workdir():
    """
    A private, empty folder for one tool run. It sits NESTING folders deep, so a
    relative path in the code (`include "../../x.v"`) cannot climb out of it.
    """
    with tempfile.TemporaryDirectory(prefix="rtl_") as tmp:
        work = Path(tmp, *["d"] * NESTING)
        work.mkdir(parents=True)
        yield work


def _run(cmd, cwd, timeout):
    """Run a tool. Returns (ok, output); a timeout or a missing tool counts as a failure."""
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout, text=True, errors="replace",
                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.returncode == 0, p.stdout
    except subprocess.TimeoutExpired:
        return False, f"{TIMED_OUT} after {timeout}s"
    except FileNotFoundError:
        return False, f"{cmd[0]} is not installed (or not on PATH)"


# ------------------------------------------------------------------ compile
def compiles(code):
    """Does the module compile and elaborate on its own with Icarus Verilog?"""
    with _workdir() as tmp:
        Path(tmp, "design.v").write_text(code, encoding="utf-8")
        log = ""
        # plain Verilog first: SystemVerilog mode rejects old code that uses
        # words like `bit` or `logic` as signal names
        for std in ("-g2005", "-g2012"):
            ok, log = _run(["iverilog", std, "-t", "null", "design.v"], tmp, COMPILE_TIMEOUT)
            if ok:
                return True, log
        return False, log


# ------------------------------------------------------------------ lint
# "%Warning-BLKSEQ: design.v:9:22: Blocking assignment '=' in sequential logic process"
_FINDING = re.compile(r"%(Warning|Error)(?:-(\w+))?: (?:design\.v:(\d+):(\d+): )?(.*)")
_NOISE = ("This may be because", "Exiting due to", "Cannot continue")       # they repeat the finding before them
# the line under it:   ": ... Suggest using delayed assignment '<='"
_HINT = re.compile(r"\s+: \.\.\. (?!note: In instance)(.*)")


def _lint(code, language):
    with _workdir() as tmp:
        Path(tmp, "design.v").write_text(code, encoding="utf-8")
        # -Wno-fatal: keep going after a warning, so the later checks run too.
        # The two checks that are switched off are about the file, not the design.
        ok, log = _run(["verilator", "--lint-only", "-Wall", "-Wno-fatal", "-Wno-DECLFILENAME", "-Wno-EOFNEWLINE",
                        "--timing", "--default-language", language, "design.v"], tmp, LINT_TIMEOUT)
    found = []
    for line in log.splitlines():
        m = _FINDING.match(line)
        if m and not m.group(5).startswith(_NOISE):
            found.append({"line": int(m.group(3) or 0), "col": int(m.group(4) or 0), "severity": m.group(1).lower(),
                          "code": m.group(2) or "", "message": m.group(5)})
        elif found and _HINT.match(line):
            found[-1]["message"] += "; " + _HINT.match(line).group(1)
    if not ok and not found:                        # Verilator itself failed: timeout or crash
        found.append({"line": 0, "col": 0, "severity": "error", "code": "",
                      "message": (log.strip().splitlines() or ["verilator failed"])[-1]})
    return sorted(found, key=lambda f: (f["line"], f["col"]))


def lint_rtl(code):
    """
    Verilator's lint findings for the code, in line order. A clean module gives [].

    Each finding is a dict: line, col, severity ("warning" or "error"), code
    (Verilator's name for the check, such as BLKSEQ or WIDTHTRUNC; "" for a
    plain error) and message. Line 0 means the finding is about the whole file.
    """
    best = None
    # plain Verilog first: SystemVerilog mode rejects old code that uses
    # words like `bit` or `logic` as signal names
    for language in ("1364-2005", "1800-2017"):
        found = _lint(code, language)
        errors = sum(f["severity"] == "error" for f in found)
        if best is None or errors < best[0]:
            best = (errors, found)
        if not errors:
            break
    return best[1]


# ------------------------------------------------------------------ equivalence
# Step 1 (per design): synthesize to a netlist of gates and flip-flops.
_SYNTH = """
read_verilog {sv} design.v
hierarchy -auto-top
proc
flatten
memory
opt_clean
async2sync
select -assert-any o:*
rename -top {name}
write_rtlil {name}.il
"""

# Step 2: prove every same-named signal of the two netlists equal, for all
# inputs and all time (induction), starting from the same register state.
_EQUIV = """
read_rtlil gold.il
read_rtlil gate.il
equiv_make gold gate equiv
hierarchy -top equiv
equiv_simple -seq 5
equiv_induct -seq 5
equiv_status -assert
"""


def _yosys(code, script, tmp, **fields):
    """Run a Yosys script on the code: read as Verilog first, then as SystemVerilog."""
    Path(tmp, "design.v").write_text(code, encoding="utf-8")
    log = ""
    for sv in ("", "-sv"):
        Path(tmp, "script.ys").write_text(script.format(sv=sv, **fields), encoding="utf-8")
        ok, log = _run(["yosys", "-q", "script.ys"], tmp, SYNTH_TIMEOUT)
        if ok or log.startswith(TIMED_OUT):     # too big: SystemVerilog mode would not help
            return ok, log
    return False, log


def equivalent(original, patched):
    """
    Does `patched` behave like `original`?

    Yosys synthesizes both and tries to prove them equal. The proof is exact
    for combinational logic and, for registers, holds for every clock cycle.
    It is conservative: a patch counts as a fix only when the proof succeeds.
    """
    with _workdir() as tmp:
        ok, log = _yosys(original, _SYNTH, tmp, name="gold")
        if not ok:
            return UNCHECKABLE, log
        ok, log = _yosys(patched, _SYNTH, tmp, name="gate")
        if not ok:
            return PATCH_ERROR, log
        Path(tmp, "equiv.ys").write_text(_EQUIV, encoding="utf-8")
        ok, log = _run(["yosys", "-q", "equiv.ys"], tmp, EQUIV_TIMEOUT)
        return (EQUIVALENT if ok else DIFFERENT), log


def checkable(original):
    """Can equivalence be checked against this original at all? (it must prove equal to itself)"""
    return equivalent(original, original)[0] == EQUIVALENT


# ------------------------------------------------------------------ schematic
# What Yosys turns the code into before it is drawn. A module defined in
# another file stays in the drawing as an empty box.
LEVELS = {
    "gate": "synth -flatten -run coarse:check",     # single gates and flip-flops
    "rtl": "proc; opt; wreduce",                    # adders, comparators, multiplexers, registers
}

_NETLIST = """
read_verilog {sv} design.v
hierarchy -auto-top
{passes}
opt_clean -purge
write_json netlist.json
"""


def _escaped(data):
    """The same JSON data with every name made safe to copy into an SVG."""
    if isinstance(data, dict):
        return {html.escape(key): _escaped(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_escaped(value) for value in data]
    return html.escape(data) if isinstance(data, str) else data


def render_schematic(code, level="gate"):
    """
    Draw the module as a schematic: Yosys synthesizes it, netlistsvg draws it.

    Returns (svg, log): the SVG text, or None with the reason in the log.
    Level "rtl" keeps buses and arithmetic as single blocks, so it stays
    readable for designs that are too big to draw gate by gate.
    """
    netlistsvg = shutil.which("netlistsvg")         # an npm script: needs its full path on Windows
    if not netlistsvg:
        return None, "netlistsvg is not installed (npm install -g netlistsvg)"
    with _workdir() as tmp:
        ok, log = _yosys(code, _NETLIST, tmp, passes=LEVELS[level])
        if not ok:
            return None, log
        netlist = json.loads(Path(tmp, "netlist.json").read_text(encoding="utf-8"))
        cells = sum(len(m["cells"]) for m in netlist["modules"].values())
        if cells > MAX_CELLS:
            return None, f"too big to draw: {cells} cells at level '{level}' (the limit is {MAX_CELLS})"
        # netlistsvg copies names into the SVG as they are, and a Verilog name may contain <script>
        Path(tmp, "netlist.json").write_text(json.dumps(_escaped(netlist)), encoding="utf-8")
        ok, log = _run([netlistsvg, "netlist.json", "-o", "schematic.svg"], tmp, DRAW_TIMEOUT)
        if not ok:                                  # netlistsvg prints a long stack trace: keep the error line
            error = next((line for line in log.splitlines() if "Error" in line), log)
            return None, f"netlistsvg could not draw this design: {error.strip()}"
        svg = Path(tmp, "schematic.svg").read_text(encoding="utf-8")
        # netlistsvg gives the drawing a fixed size; with a viewBox it scales to the size it is shown at
        svg = re.sub(r'<svg ([^>]*?)width="([\d.]+)" height="([\d.]+)"',
                     r'<svg \1width="\2" height="\3" viewBox="0 0 \2 \3"', svg, count=1)
        return svg, log


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Draw a Verilog file as a schematic (SVG), or lint it.")
    p.add_argument("file", type=Path)
    p.add_argument("--lint", action="store_true", help="print the lint warnings instead of drawing")
    p.add_argument("--level", default="gate", choices=LEVELS)
    p.add_argument("-o", "--out", type=Path, help="default: the input file with .svg")
    args = p.parse_args()
    source = args.file.read_text(encoding="utf-8", errors="replace")
    if args.lint:
        findings = lint_rtl(source)
        for f in findings:
            print(f"{args.file.name}:{f['line']}:{f['col']}: {f['severity']} {f['code']}".rstrip() + f": {f['message']}")
        raise SystemExit(1 if findings else 0)
    svg, log = render_schematic(source, args.level)
    if svg is None:
        raise SystemExit(f"no schematic: {log.strip()}")
    out = args.out or args.file.with_suffix(".svg")
    out.write_text(svg, encoding="utf-8")
    print(f"saved {out}")
