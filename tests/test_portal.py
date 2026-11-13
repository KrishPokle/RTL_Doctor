"""
Tests for the portal: the chat endpoint, its input checks and the tool loop.

    python -m pytest tests

Skipped without fastapi. The tests that run a tool also need that tool
(verilator, or yosys and netlistsvg).
"""
import io
import json
import shutil
import urllib.error
import urllib.request

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient      # noqa: E402

from portal import agent, llm                  # noqa: E402
from portal.app import MAX_FILE_BYTES, app     # noqa: E402

client = TestClient(app)
needs_lint = pytest.mark.skipif(not shutil.which("verilator"), reason="needs verilator")
needs_drawing = pytest.mark.skipif(not (shutil.which("yosys") and shutil.which("netlistsvg")),
                                   reason="needs yosys and netlistsvg")

BUGGY = """module counter (
  input clk, input rst_n, input en,
  output reg [7:0] cnt
);
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) cnt <= 8'd0;
    else if (en) cnt = cnt + 8'd1;
  end
endmodule
"""
FIXED = BUGGY.replace("cnt = cnt", "cnt <= cnt")


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    """Use the stand-in, whatever the developer's own .env says."""
    monkeypatch.setattr(llm, "load_env", lambda: None)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)


def ask(question, code=BUGGY, name="counter.v"):
    body = {"messages": [{"role": "user", "content": question}]}
    if code is not None:
        body["file"] = {"name": name, "code": code}
    return client.post("/api/chat", json=body)


# ------------------------------------------------------------------ the page and its input checks
def test_serves_the_page_and_says_no_model_is_connected():
    assert "RTL Doctor" in client.get("/").text
    assert client.get("/api/status").json()["model"] is None


@pytest.mark.parametrize("name, code", [("notes.py", "print(1)"), ("big.v", "a" * (MAX_FILE_BYTES + 1)),
                                        ("empty.sv", "  \n"), ("binary.v", "module\x00")])
def test_accepts_only_verilog_text_of_a_sensible_size(name, code):
    assert ask("check it", code, name).status_code == 400


def test_the_last_message_has_to_be_the_users():
    body = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    assert client.post("/api/chat", json=body).status_code == 400


def test_without_a_file_no_tool_runs():
    answer = ask("Can you detect any bugs in my Verilog RTL design?", code=None).json()
    assert answer["tools"] == [] and "Choose a Verilog file" in answer["reply"]


# ------------------------------------------------------------------ the proposal's three questions, with the stand-in
@needs_lint
def test_bug_question_runs_the_linter():
    answer = ask("Can you detect any bugs in my Verilog RTL design?").json()
    assert answer["tools"] == ["lint_rtl"]
    assert (7, "BLKSEQ") in [(f["line"], f["code"]) for f in answer["findings"]]
    assert "BLKSEQ" in answer["reply"]


@needs_drawing
def test_schematic_question_draws_gate_by_gate():
    answer = ask("Can you visualize my RTL design as a gate level schematic?").json()
    assert answer["tools"] == ["render_schematic"]
    assert answer["schematic"]["level"] == "gate" and answer["schematic"]["svg"].startswith("<svg")


@needs_lint
def test_fix_question_repairs_the_bug_type_the_linter_points_at(monkeypatch):
    asked = []
    monkeypatch.setattr(agent, "repair", lambda code, bug_type: (asked.append(bug_type), (FIXED, ""))[1])
    answer = ask("Can you explain this bug and fix it?").json()
    assert answer["tools"] == ["lint_rtl", "repair_rtl"] and asked == ["blocking_in_seq"]
    assert answer["repair"]["patched"] == FIXED
    assert "+    else if (en) cnt <= cnt + 8'd1;" in answer["repair"]["diff"]


def test_a_missing_repair_model_is_said_plainly(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "ADAPTER", tmp_path / "no_adapter")
    result = agent.run_tool("repair_rtl", '{"bug_type": "blocking_in_seq"}', {"name": "c.v", "code": BUGGY}, {})
    assert result["fixed"] is False and "not installed" in result["reason"]


@pytest.mark.parametrize("name, arguments", [("format_disk", "{}"), ("render_schematic", '{"level": "pcb"}'),
                                             ("repair_rtl", '{"bug_type": "typo"}'), ("lint_rtl", "not json")])
def test_a_bad_tool_call_gets_an_error_back_instead_of_running(name, arguments):
    assert "error" in agent.run_tool(name, arguments, {"name": "c.v", "code": BUGGY}, {})


# ------------------------------------------------------------------ a real model (its API is faked here)
def model_api(monkeypatch, replies):
    """Point the portal at a model API that gives these replies in turn. Returns the requests it got."""
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1/")
    monkeypatch.setenv("LLM_API_KEY", "secret-key")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    requests, replies = [], iter(replies)

    def urlopen(request, timeout):
        requests.append(request)
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(json.dumps({"choices": [{"message": reply}]}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return requests


@needs_lint
def test_a_model_gets_the_file_and_the_results_of_the_tools_it_calls(monkeypatch):
    call = {"id": "call_1", "type": "function", "function": {"name": "lint_rtl", "arguments": "{}"}}
    requests = model_api(monkeypatch, [{"role": "assistant", "content": None, "tool_calls": [call]},
                                       {"role": "assistant", "content": "Line 7 uses = where it needs <=."}])
    answer = ask("What is wrong with it?").json()
    assert answer["reply"] == "Line 7 uses = where it needs <=." and answer["tools"] == ["lint_rtl"]

    first, second = (json.loads(r.data) for r in requests)
    assert requests[0].full_url == "https://llm.example/v1/chat/completions"
    assert requests[0].get_header("Authorization") == "Bearer secret-key" and first["model"] == "some-model"
    assert [t["function"]["name"] for t in first["tools"]] == ["lint_rtl", "render_schematic", "repair_rtl"]
    assert "   7|     else if (en) cnt = cnt + 8'd1;" in first["messages"][1]["content"]     # the numbered file
    tool_result = second["messages"][-1]
    assert tool_result["role"] == "tool" and tool_result["tool_call_id"] == "call_1"
    assert "BLKSEQ" in tool_result["content"]


def test_a_model_that_cannot_be_reached_is_reported(monkeypatch):
    model_api(monkeypatch, [urllib.error.URLError("no route to host")])
    response = ask("What is wrong with it?")
    assert response.status_code == 502 and "could not be reached" in response.json()["detail"]
