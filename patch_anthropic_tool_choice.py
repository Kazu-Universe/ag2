"""
One-time patch: translates ag2's OpenAI-style forced tool_choice
({"type": "function", "function": {"name": X}}) into Anthropic's actual
schema ({"type": "tool", "name": X}) and passes it through to the Anthropic
client.

REVISED from an earlier version of this patch written against a different
snapshot of this codebase. There, tool_choice WAS being copied into
anthropic_params in OpenAI's schema (needing only a translation). In this
version of the file, tool_choice is not copied into anthropic_params at
all - it is silently dropped before ever reaching the Anthropic API call,
which is arguably worse: a forced tool_choice (used by terminator,
plan_recorder, review_recorder, etc.) would just be ignored rather than
sent in the wrong schema. This patch now does the full job: extract
params["tool_choice"] if present, translate it if it's in OpenAI's
forced-function schema, and set anthropic_params["tool_choice"]
accordingly - inserted right after "tools" is copied across, so it has
access to the already-established params/anthropic_params state.

Run once from the repo root:
    python patch_anthropic_tool_choice.py
"""

path = "autogen/oai/anthropic.py"
marker = "hep-theory fork fix"

with open(path, "r") as f:
    content = f.read()

if marker in content:
    print(f"Already patched: {path}")
    raise SystemExit(0)

old = '''        if "tools" in params:
            anthropic_params["tools"] = params["tools"]
        if anthropic_params["top_k"] is None:'''

new = '''        if "tools" in params:
            anthropic_params["tools"] = params["tools"]
        # hep-theory fork fix: params["tool_choice"] was being silently
        # dropped here - never copied into anthropic_params at all, so a
        # forced tool_choice (used by terminator, plan_recorder,
        # review_recorder, etc.) never reached the Anthropic API call.
        # ag2 builds tool_choice in OpenAI's schema unconditionally:
        # {"type": "function", "function": {"name": X}}. Anthropic's
        # forced-single-tool schema ({"type": "tool", "name": X}) was
        # never implemented here. Without this, any agent using a forced
        # tool_choice gets no forcing at all when backed by a
        # Claude/Fable model - not even a 400, just silently ignored.
        if "tool_choice" in params:
            tc = params["tool_choice"]
            if isinstance(tc, dict) and tc.get("type") == "function":
                fn = tc.get("function", {})
                name = fn.get("name") if isinstance(fn, dict) else None
                if name:
                    anthropic_params["tool_choice"] = {"type": "tool", "name": name}
            elif isinstance(tc, dict) and tc.get("type") == "tool":
                # Already in Anthropic's own schema - pass through as-is.
                anthropic_params["tool_choice"] = tc
        if anthropic_params["top_k"] is None:'''

count = content.count(old)
if count != 1:
    raise SystemExit(
        f"Expected exactly 1 occurrence, found {count}. "
        "Paste output so this can be adjusted."
    )

content = content.replace(old, new, 1)

with open(path, "w") as f:
    f.write(content)

print(f"Patched: {path}")
