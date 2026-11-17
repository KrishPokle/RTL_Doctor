"""
Tests for lint_rtl in rtl_tools.py.

    python -m pytest tests

They need verilator on the PATH and are skipped without it.
"""
import shutil

import pytest

import rtl_tools

pytestmark = pytest.mark.skipif(not shutil.which("verilator"), reason="needs verilator")

GOOD = """module counter (
  input clk, input rst_n, input en, input [1:0] sel,
  output reg [7:0] cnt, output reg [1:0] dec
);
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) cnt <= 8'd0;
    else if (en) cnt <= cnt + 8'd1;
  end
  always @(*) begin
    case (sel)
      2'b00: dec = 2'd1;
      default: dec = 2'd0;
    endcase
  end
endmodule
"""


def found(code):
    return [(f["line"], f["code"]) for f in rtl_tools.lint_rtl(code)]


def test_a_clean_module_has_no_findings():
    assert rtl_tools.lint_rtl(GOOD) == []


def test_finds_the_bug_types_a_linter_can_see():
    # line 3 is the declaration of cnt, which now mixes = and <=
    assert found(GOOD.replace("cnt <= cnt + 8'd1", "cnt = cnt + 8'd1")) == [(3, "BLKANDNBLK"), (7, "BLKSEQ")]
    assert found(GOOD.replace("      default: dec = 2'd0;\n", "")) == [(10, "CASEINCOMPLETE")]
    assert found(GOOD.replace("output reg [7:0] cnt", "output reg [6:0] cnt")) == [(6, "WIDTHTRUNC"), (7, "WIDTHTRUNC")]


def test_a_finding_has_its_place_severity_and_hint():
    finding = rtl_tools.lint_rtl(GOOD.replace("cnt <= cnt + 8'd1", "cnt = cnt + 8'd1"))[-1]
    assert finding == {"line": 7, "col": 22, "severity": "warning", "code": "BLKSEQ",
                       "message": "Blocking assignment '=' in sequential logic process; "
                                  "Suggest using delayed assignment '<='"}


def test_errors_are_reported_once_and_do_not_hide_the_warnings():
    code = "module top (input a, input b, output [1:0] y);\n  wire w;\n  adder u1 (a, b, w);\n  assign y = w & a;\nendmodule\n"
    findings = rtl_tools.lint_rtl(code)
    assert [(f["line"], f["severity"]) for f in findings if f["severity"] == "error"] == [(3, "error")]
    assert "adder" in findings[0]["message"] and any(f["code"] == "WIDTHEXPAND" for f in findings)


def test_a_syntax_error_is_one_finding():
    finding, = rtl_tools.lint_rtl("module m (input a, output y);\n  assign y = a\nendmodule\n")
    assert (finding["line"], finding["severity"]) == (3, "error") and "syntax error" in finding["message"]


def test_old_verilog_names_and_systemverilog_both_lint_clean():
    assert rtl_tools.lint_rtl("module m (input logic, output bit);\n  assign bit = ~logic;\nendmodule\n") == []
    assert rtl_tools.lint_rtl("module m (input logic a, output logic y);\n  always_comb y = ~a;\nendmodule\n") == []


def test_a_finding_about_the_whole_file_has_line_zero():
    finding, = rtl_tools.lint_rtl("// nothing here\n")
    assert finding["line"] == 0 and finding["severity"] == "error"
