"""
Verilog tool wrappers: Icarus Verilog (compile), Yosys (equivalence) and
netlistsvg (schematic).

Every run happens in its own temporary directory with a timeout, and the
Verilog under test is only ever passed to the tools as a file, never pasted
into a command line or a tool script.

    compiles(code)                 -> (ok, log)
    equivalent(original, patched)  -> (verdict, log)
    render_schematic(code, level)  -> (svg, log)

    python rtl_tools.py design.v               # writes design.svg (gate level)
    python rtl_tools.py design.v --level rtl   # adders, muxes and registers instead of single gates
"""
import html
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

COMPILE_TIMEOUT = 10    # seconds
SYNTH_TIMEOUT = 10
EQUIV_TIMEOUT = 20
DRAW_TIMEOUT = 30
MAX_CELLS = 500         # a schematic with more cells than this is not readable
TIMED_OUT = "timeout"

# verdicts returned by equivalent()
EQUIVALENT = "equivalent"            # proven to behave like the original
DIFFERENT = "different"              # patched design synthesizes, but equivalence is not proven
PATCH_ERROR = "patch_error"          # patched design does not synthesize
UNCHECKABLE = "uncheckable"          # the ORIGINAL cannot be checked (not synthesizable, no outputs, too big)


def _run(cmd, cwd, timeout):
    """Run a tool. Returns (ok, output); a timeout counts as a failure."""
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout, text=True, errors="replace",
                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.returncode == 0, p.stdout
    except subprocess.TimeoutExpired:
        return False, f"{TIMED_OUT} after {timeout}s"


# ------------------------------------------------------------------ compile
def compiles(code):
    """Does the module compile and elaborate on its own with Icarus Verilog?"""
    with tempfile.TemporaryDirectory(prefix="rtl_") as tmp:
        Path(tmp, "design.v").write_text(code, encoding="utf-8")
        log = ""
        # plain Verilog first: SystemVerilog mode rejects old code that uses
        # words like `bit` or `logic` as signal names
        for std in ("-g2005", "-g2012"):
            ok, log = _run(["iverilog", std, "-t", "null", "design.v"], tmp, COMPILE_TIMEOUT)
            if ok:
                return True, log
        return False, log


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
    with tempfile.TemporaryDirectory(prefix="rtl_") as tmp:
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
    with tempfile.TemporaryDirectory(prefix="rtl_") as tmp:
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
        return Path(tmp, "schematic.svg").read_text(encoding="utf-8"), log


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Draw a Verilog file as a schematic (SVG).")
    p.add_argument("file", type=Path)
    p.add_argument("--level", default="gate", choices=LEVELS)
    p.add_argument("-o", "--out", type=Path, help="default: the input file with .svg")
    args = p.parse_args()
    svg, log = render_schematic(args.file.read_text(encoding="utf-8", errors="replace"), args.level)
    if svg is None:
        raise SystemExit(f"no schematic: {log.strip()}")
    out = args.out or args.file.with_suffix(".svg")
    out.write_text(svg, encoding="utf-8")
    print(f"saved {out}")
