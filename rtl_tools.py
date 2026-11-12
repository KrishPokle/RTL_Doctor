"""
Verilog tool wrappers: Icarus Verilog (compile) and Yosys (equivalence).

Every run happens in its own temporary directory with a timeout, and the
Verilog under test is only ever passed to the tools as a file, never pasted
into a command line or a tool script.

    compiles(code)                 -> (ok, log)
    equivalent(original, patched)  -> (verdict, log)
"""
import subprocess
import tempfile
from pathlib import Path

COMPILE_TIMEOUT = 10    # seconds
SYNTH_TIMEOUT = 10
EQUIV_TIMEOUT = 20
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


def _synth(code, name, tmp):
    Path(tmp, "design.v").write_text(code, encoding="utf-8")
    log = ""
    for sv in ("", "-sv"):
        Path(tmp, "synth.ys").write_text(_SYNTH.format(sv=sv, name=name), encoding="utf-8")
        ok, log = _run(["yosys", "-q", "synth.ys"], tmp, SYNTH_TIMEOUT)
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
        ok, log = _synth(original, "gold", tmp)
        if not ok:
            return UNCHECKABLE, log
        ok, log = _synth(patched, "gate", tmp)
        if not ok:
            return PATCH_ERROR, log
        Path(tmp, "equiv.ys").write_text(_EQUIV, encoding="utf-8")
        ok, log = _run(["yosys", "-q", "equiv.ys"], tmp, EQUIV_TIMEOUT)
        return (EQUIVALENT if ok else DIFFERENT), log


def checkable(original):
    """Can equivalence be checked against this original at all? (it must prove equal to itself)"""
    return equivalent(original, original)[0] == EQUIVALENT
