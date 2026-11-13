"""
Tests for the bug injectors in make_dataset.py.

    python -m pytest tests
"""
import random

import pytest

from make_dataset import INJECTORS

# One place per bug type where the bug can go. The comments and the string hold
# look-alikes for every bug type, and the spacing is deliberately uneven.
CODE = """// 8-bit counter: if (en) count up; posedge rst clears cnt[7:0]
/* earlier version
   assign par = en + odd;
   cnt <= cnt + 1;
   default: dec = 1'b1;
*/
module counter (
  input clk, input rst_n, input en, input odd, input [7 : 0] a,
  output reg [7:1] cnt, output par, output reg dec
);
  assign par = en&odd;   // en + odd

  always @(posedge clk or negedge\trst_n) begin
    if (!rst_n) cnt <= 7'd0;
    else if( en )

      cnt<= cnt + 7'd1;      // if (en) cnt <= cnt + 1
    $display("if (en) cnt <= [7:0] negedge rst_n");
  end
  always @(*) begin
    case (a[2:1])
      2'b00: dec = 1'b1;
      default: dec = 1'b0;   // default: off
    endcase
  end
endmodule
"""

# bug type -> (the text the bug goes into, what it becomes)
EDITS = {
    "blocking_in_seq": ("cnt<= cnt + 7'd1;", "cnt= cnt + 7'd1;"),
    "missing_default": ("      default: dec = 1'b0;   // default: off\n", ""),
    "width_off_by_one": ("[7 : 0]", "[6 : 0]"),
    "inverted_condition": ("if( en )", "if( !en )"),
    "reset_polarity": ("negedge\trst_n", "posedge\trst_n"),
    "wrong_operator": ("en&odd", "en|odd"),
}


@pytest.mark.parametrize("bug", INJECTORS)
def test_bug_goes_into_code_and_changes_nothing_else(bug):
    before, after = EDITS[bug]
    assert CODE.count(before) == 1
    line = CODE[:CODE.index(before)].count("\n") + 1
    for seed in range(20):                      # the injector picks a place at random
        random.seed(seed)
        assert INJECTORS[bug](CODE) == (CODE.replace(before, after), line)


@pytest.mark.parametrize("bug", INJECTORS)
def test_no_bug_when_the_only_place_is_a_comment(bug):
    code = """module m (input a, output y);
  // if (a) y <= b[3:0]; always @(posedge rst)
  /* assign y = a + a;
     y <= a;
     default: y = 0;
  */
  buf (y, a);
endmodule
"""
    assert INJECTORS[bug](code) is None


def test_width_shrinks_by_one():
    random.seed(0)
    assert INJECTORS["width_off_by_one"]("wire [10:0] w;") == ("wire [9:0] w;", 1)
