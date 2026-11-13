"""
The language model behind the chat.

Settings come from the environment, or from a `.env` file in the repo root:

    LLM_BASE_URL   an OpenAI-compatible API: the address that /chat/completions is added to
    LLM_API_KEY    the key for it (never commit this)
    LLM_MODEL      the model to ask for, spelled as the API lists it

    python -m portal.llm        # lists the models the API offers, to pick LLM_MODEL from

Without LLM_BASE_URL the portal uses StandIn: a few fixed rules that pick the
tools from keywords in the question. It runs the portal end to end with no
model and no key, but it cannot explain anything.

Both speak the OpenAI chat format: complete(messages, tools) returns one
assistant message, with `tool_calls` when the model wants a tool run.
"""
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT = 60        # seconds for one model reply


class LLMError(Exception):
    """The model could not be reached or answered with an error."""


def load_env(path=ROOT / ".env"):
    """Read KEY=VALUE lines into the environment. Variables that are already set win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, eq, value = line.strip().partition("=")
        if eq and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def get_llm():
    """The configured model, or the stand-in when none is configured."""
    load_env()
    base = os.environ.get("LLM_BASE_URL", "").strip()
    if not base:
        return StandIn()
    return ChatAPI(base, os.environ.get("LLM_API_KEY", ""), os.environ.get("LLM_MODEL", ""))


class ChatAPI:
    """An OpenAI-compatible API: POST /chat/completions and GET /models."""

    def __init__(self, base_url, api_key, model):
        self.base, self.api_key, self.name = base_url.rstrip("/"), api_key, model

    def _send(self, path, body=None):
        """One request to the API. Returns the JSON it answers with."""
        request = urllib.request.Request(self.base + path, method="POST" if body else "GET",
                                         data=json.dumps(body).encode("utf-8") if body else None,
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            raise LLMError(f"the model API answered {e.code} {e.reason}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise LLMError(f"the model API could not be reached: {getattr(e, 'reason', e)}") from e
        except ValueError as e:
            raise LLMError("the model API sent a reply the portal cannot read") from e

    def complete(self, messages, tools):
        body = {"model": self.name, "messages": messages}
        if tools:
            body["tools"] = tools
        try:
            message = self._send("/chat/completions", body)["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("the model API sent a reply the portal cannot read") from e
        return {"role": "assistant", "content": message.get("content") or "",
                "tool_calls": message.get("tool_calls") or []}

    def models(self):
        """The model names the API offers: what LLM_MODEL can be set to."""
        listing = self._send("/models")
        try:
            return sorted(m["id"] for m in (listing["data"] if isinstance(listing, dict) else listing))
        except (KeyError, TypeError) as e:
            raise LLMError("the model API sent a model list the portal cannot read") from e


# ------------------------------------------------------------------ stand-in
# what the question has to mention for each tool
WANTS_FIX = re.compile(r"\b(fix|repair|patch|correct)", re.I)
WANTS_SCHEMATIC = re.compile(r"\b(schematic|visuali[sz]e|draw|diagram|gates?|netlist)\b", re.I)
WANTS_CHECK = re.compile(r"\b(bugs?|lint|check|detect|wrong|errors?|warnings?|review|problems?|issues?)\b", re.I)
WANTS_BLOCKS = re.compile(r"\b(blocks?|block[- ]level|rtl[- ]level)\b", re.I)

# the lint checks that point at a bug type the repair model knows
BUG_OF_CHECK = {"BLKSEQ": "blocking_in_seq", "CASEINCOMPLETE": "missing_default",
                "WIDTHTRUNC": "width_off_by_one", "WIDTHEXPAND": "width_off_by_one"}


def _call(name, **arguments):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": f"standin_{name}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def _say(text):
    return {"role": "assistant", "content": text, "tool_calls": []}


def _sentence(text):
    return text if text.endswith((".", "!", "?")) else text + "."


class StandIn:
    """Picks tools by keyword and reports what they returned. No model, no network."""
    name = None

    def complete(self, messages, tools):
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        question = messages[last_user]["content"]
        results = {}                                    # tool name -> its result, for this question
        for m in messages[last_user + 1:]:
            if m["role"] == "tool":
                results[m["tool_call_id"].removeprefix("standin_")] = json.loads(m["content"])
        if not tools:
            return _say("Choose a Verilog file first. Then I can check it for bugs, draw its schematic or try a fix.")

        if WANTS_FIX.search(question):
            if "lint_rtl" not in results:
                return _call("lint_rtl")                # the lint findings say which bug type to repair
            if "repair_rtl" not in results:
                bug = next((BUG_OF_CHECK[f["code"]] for f in results["lint_rtl"]["findings"]
                            if f["code"] in BUG_OF_CHECK), None)
                if bug is None:
                    return _say(self._lint_text(results["lint_rtl"]) + " None of the findings points at a bug type "
                                "the repair model knows, so I did not try a fix.")
                return _call("repair_rtl", bug_type=bug)
            return _say(self._repair_text(results["repair_rtl"]))
        if WANTS_SCHEMATIC.search(question):
            if "render_schematic" not in results:
                return _call("render_schematic", level="rtl" if WANTS_BLOCKS.search(question) else "gate")
            return _say(self._schematic_text(results["render_schematic"]))
        if WANTS_CHECK.search(question):
            if "lint_rtl" not in results:
                return _call("lint_rtl")
            return _say(self._lint_text(results["lint_rtl"]))
        return _say("No language model is connected, so I can only run the tools. "
                    "Ask me to check the file for bugs, draw the schematic, or fix a bug.")

    @staticmethod
    def _lint_text(result):
        total = result["total"]
        if not total:
            return "The linter found nothing wrong with this file."
        checks = {}                                     # check -> how many findings
        for f in result["findings"]:
            checks[f["code"] or "error"] = checks.get(f["code"] or "error", 0) + 1
        lines = sorted({f["line"] for f in result["findings"] if f["line"]})
        where = f" on line{'s' * (len(lines) != 1)} {', '.join(map(str, lines[:8]))}" if lines else ""
        if len(lines) > 8:
            where += " and others"
        return (f"The linter reports {total} finding{'s' * (total != 1)}{where}: "
                + ", ".join(f"{n} {check}" for check, n in checks.items()) + ". The Findings tab lists them.")

    @staticmethod
    def _schematic_text(result):
        if result["drawn"]:
            level = "gate by gate" if result["level"] == "gate" else "as blocks"
            return f"The schematic is drawn {level} in the Schematic tab."
        return f"I could not draw the schematic: {result['reason']}"

    @staticmethod
    def _repair_text(result):
        if result["fixed"]:
            return (f"The repair model changed the file for a {result['bug_type']} bug, and the result still "
                    "compiles. The Repair tab shows the change.")
        return f"No fix: {result['reason']}"


if __name__ == "__main__":
    llm = get_llm()
    if isinstance(llm, StandIn):
        raise SystemExit("LLM_BASE_URL is not set. Copy .env.example to .env and fill it in.")
    try:
        names = llm.models()
    except LLMError as e:
        raise SystemExit(f"Could not list the models at {llm.base}: {e}.")
    print(f"Models offered by {llm.base} (LLM_MODEL has to be one of these, spelled exactly):")
    for name in names:
        print(f"  {name}" + ("    <- LLM_MODEL in your .env" if name == llm.name else ""))
    if llm.name not in names:
        print(f"LLM_MODEL is set to {llm.name!r}, which is not in the list." if llm.name else "LLM_MODEL is not set yet.")
