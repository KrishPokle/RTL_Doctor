"""
Tests for the repair model's answer format and the fix-rate checker.

    python -m pytest tests

The checker tests need iverilog and yosys on the PATH and are skipped without them.
"""
import shutil

import pytest

import rtl_tools
from repair.prompt import apply_answer, build_answer, build_prompt, make_edit
from splits import split_of

GOOD = """module counter (
  input clk, input rst_n, input en, input [7:0] a, input [7:0] b,
  output reg [7:0] cnt, output [7:0] sum, output reg [1:0] dec
);
  assign sum = a + b;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) cnt <= 8'd0;
    else if (en) cnt <= cnt + 8'd1;
  end
  always @(*) begin
    case (a[1:0])
      2'b00: dec = 2'd1;
      2'b01: dec = 2'd2;
      default: dec = 2'd0;
    endcase
  end
endmodule
"""

BUGS = {
    "blocking_in_seq": GOOD.replace("cnt <= cnt + 8'd1", "cnt = cnt + 8'd1"),
    "missing_default": GOOD.replace("      default: dec = 2'd0;\n", ""),
    "width_off_by_one": GOOD.replace("output reg [7:0] cnt", "output reg [6:0] cnt"),
    "inverted_condition": GOOD.replace("if (en)", "if (!en)"),
    "reset_polarity": GOOD.replace("negedge rst_n", "posedge rst_n"),
    "wrong_operator": GOOD.replace("a + b", "a - b"),
}


# ------------------------------------------------------------------ answer format
@pytest.mark.parametrize("fmt", ["edit", "module"])
@pytest.mark.parametrize("bug", BUGS)
def test_answer_rebuilds_the_original(bug, fmt):
    answer = build_answer(BUGS[bug], GOOD, fmt)
    assert apply_answer(BUGS[bug], answer, fmt) == GOOD


def test_edit_is_small_and_unique():
    search, replace = make_edit(BUGS["inverted_condition"], GOOD)
    assert search.strip() == "else if (!en) cnt <= cnt + 8'd1;"
    assert replace.strip() == "else if (en) cnt <= cnt + 8'd1;"


def test_edit_adds_context_when_the_line_repeats():
    good = "module m;\n  a <= b;\n  x <= y;\n  a <= b;\n  z <= w;\nendmodule"
    buggy = "module m;\n  a <= b;\n  x <= y;\n  a = b;\n  z <= w;\nendmodule"
    twice = buggy.replace("  a <= b;", "  a = b;")      # now `a = b;` appears twice
    search, _ = make_edit(twice, twice.replace("  a = b;\n  z", "  a <= b;\n  z"))
    assert search.count("\n") >= 1                      # more than the one ambiguous line
    assert apply_answer(buggy, build_answer(buggy, good), "edit") == good


def test_edit_tolerates_indentation_and_windows_line_endings():
    answer = "<<<<<<< SEARCH\nelse if (!en) cnt <= cnt + 8'd1;\n=======\n    else if (en) cnt <= cnt + 8'd1;\n>>>>>>> REPLACE"
    patched = apply_answer(BUGS["inverted_condition"].replace("\n", "\r\n"), answer, "edit")
    assert patched.split() == GOOD.split()


@pytest.mark.parametrize("answer", ["", "I think the bug is on line 3.",
                                    "<<<<<<< SEARCH\nnot in the module\n=======\nx\n>>>>>>> REPLACE"])
def test_unusable_answers_give_no_patch(answer):
    assert apply_answer(BUGS["wrong_operator"], answer, "edit") is None


def test_prompt_names_the_bug_and_holds_the_code():
    prompt = build_prompt(BUGS["reset_polarity"], "reset_polarity")
    assert "reset_polarity" in prompt and "posedge rst_n" in prompt


def test_split_ignores_whitespace_and_is_stable():
    assert split_of(GOOD) == split_of(GOOD.replace("\n", "\r\n")) == split_of(GOOD)
    assert split_of(GOOD) in ("train", "val", "test")


# ------------------------------------------------------------------ checker
needs_tools = pytest.mark.skipif(not (shutil.which("iverilog") and shutil.which("yosys")),
                                 reason="needs iverilog and yosys")


@needs_tools
def test_compiles():
    assert rtl_tools.compiles(GOOD)[0]
    assert not rtl_tools.compiles(GOOD.replace("endmodule", ""))[0]


@needs_tools
def test_original_and_a_rewritten_fix_are_equivalent():
    assert rtl_tools.checkable(GOOD)
    rewritten = GOOD.replace("cnt + 8'd1", "8'd1 + cnt")
    assert rtl_tools.equivalent(GOOD, rewritten)[0] == rtl_tools.EQUIVALENT


@needs_tools
@pytest.mark.parametrize("bug", [b for b in BUGS if b != "blocking_in_seq"])
def test_bugs_are_not_equivalent(bug):
    assert rtl_tools.equivalent(GOOD, BUGS[bug])[0] in (rtl_tools.DIFFERENT, rtl_tools.PATCH_ERROR)


@needs_tools
def test_original_without_outputs_is_uncheckable():
    testbench = "module tb;\n  reg clk = 0;\n  always #5 clk = ~clk;\nendmodule\n"
    assert rtl_tools.equivalent(testbench, testbench)[0] == rtl_tools.UNCHECKABLE
