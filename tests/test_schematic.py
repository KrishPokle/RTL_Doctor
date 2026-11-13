"""
Tests for render_schematic in rtl_tools.py.

    python -m pytest tests

They need yosys and netlistsvg on the PATH and are skipped without them.
"""
import shutil

import pytest

import rtl_tools

pytestmark = pytest.mark.skipif(not (shutil.which("yosys") and shutil.which("netlistsvg")),
                                reason="needs yosys and netlistsvg")

COUNTER = """module counter (input clk, input rst_n, input en, output reg [3:0] cnt);
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) cnt <= 4'd0;
    else if (en) cnt <= cnt + 4'd1;
  end
endmodule
"""


def cells(svg):
    return svg.count('id="cell_')


@pytest.mark.parametrize("level", rtl_tools.LEVELS)
def test_draws_the_module_with_its_ports(level):
    svg, _ = rtl_tools.render_schematic(COUNTER, level)
    assert svg.startswith("<svg") and all(port in svg for port in ("clk", "rst_n", "en", "cnt"))


def test_gate_level_is_drawn_gate_by_gate():
    gate, rtl = (rtl_tools.render_schematic(COUNTER, level)[0] for level in ("gate", "rtl"))
    assert cells(gate) > cells(rtl)                     # four flip-flops and their gates, not one register


def test_a_module_from_another_file_is_drawn_as_a_box():
    svg, _ = rtl_tools.render_schematic("module top (input a, input b, output y);\n  adder u1 (a, b, y);\nendmodule\n")
    assert "adder" in svg


def test_a_name_cannot_put_markup_into_the_drawing():
    name = "\\a<script>alert(1)</script> "              # a legal Verilog escaped identifier
    svg, _ = rtl_tools.render_schematic(f"module m (input {name}, input b, output y);\n  assign y = {name}& b;\nendmodule\n")
    assert "<script" not in svg and "a&lt;script&gt;" in svg


def test_no_drawing_for_code_that_does_not_parse():
    assert rtl_tools.render_schematic("module m (input a; endmodule")[0] is None


def test_a_design_too_big_for_gates_still_draws_as_blocks():
    multiplier = "module m (input [31:0] a, input [31:0] b, output [63:0] y);\n  assign y = a * b;\nendmodule\n"
    svg, log = rtl_tools.render_schematic(multiplier, "gate")
    assert svg is None and "too big" in log
    assert rtl_tools.render_schematic(multiplier, "rtl")[0] is not None
