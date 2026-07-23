"""
Patch: surface silent truncation in the Anthropic wrapper instead of
producing an empty message with no error.

BUG: in autogen/oai/anthropic.py's create() method, the branch that handles
non-tool_use stop_reasons does:

    else:
        anthropic_finish = "stop"
        tool_calls = None

...and then only extracts message_text from a TextBlock if one is present.
When response.stop_reason == "max_tokens" (Claude hit the token limit while
generating - e.g. mid-way through constructing a tool_use block, with no
preceding prose), this branch fires, discards any partial tool call, finds
no TextBlock to fall back on, and silently returns
{'content': '', 'tool_calls': None} - no exception, no warning, nothing.

CONFIRMED in a real cmbagent deep_research() run: engineer's turn hit
exactly max_tokens (16000) twice on a heavy combined sympy+synthesis task,
both times producing a totally empty message. Since this is caught by
autogen's normal group-chat flow (not a raised exception), deep_research()
finished cleanly and printed "=== DONE ===" as if nothing had gone wrong,
while the most safety-critical final round of work silently never happened.

FIX: when stop_reason == "max_tokens" and no usable content/tool_calls were
extracted, set message_text to an explicit, visible truncation notice and
emit a Python warning, so this shows up directly in the transcript and in
stdout instead of disappearing.

This does NOT recover the lost partial generation (Anthropic's API does not
return usable partial tool-call JSON for a mid-truncation cutoff) - it only
makes the failure loud instead of silent, so it can be caught and retried
(e.g. by asking for a smaller sub-task, or raising max_tokens further) the
same turn rather than discovered much later by manually reading raw JSON.

Run once from the repo root:
    python patch_anthropic_truncation_warning.py
"""

path = "autogen/oai/anthropic.py"
marker = "hep-theory fork: surfaced silent max_tokens truncation"

with open(path, "r") as f:
    src = f.read()

if marker in src:
    print(f"Already patched ({path}) - nothing to do.")
else:
    old = '''                else:
                    anthropic_finish = "stop"
                    tool_calls = None

                # Retrieve any text content from the response
                for content in response.content:
                    if type(content) == TextBlock:
                        message_text = content.text
                        break'''

    new = '''                else:
                    anthropic_finish = "stop"
                    tool_calls = None

                # Retrieve any text content from the response
                for content in response.content:
                    if type(content) == TextBlock:
                        message_text = content.text
                        break

                # hep-theory fork: surfaced silent max_tokens truncation.
                # Previously, a response cut off at the token limit before
                # any TextBlock or completed ToolUseBlock existed produced
                # a totally empty {'content': '', 'tool_calls': None}
                # message with no error - deep_research() would finish
                # cleanly as if the turn had succeeded. Make this loud
                # instead: if nothing usable was extracted and the stop
                # reason was max_tokens, say so explicitly in the message
                # content and emit a warning, so it is visible in the
                # transcript and in stdout rather than silently swallowed.
                if not message_text and not tool_calls and response.stop_reason == "max_tokens":
                    import warnings as _warnings
                    _warnings.warn(
                        f"Anthropic response for model {anthropic_params.get('model')} "
                        f"was truncated at max_tokens ({anthropic_params.get('max_tokens')}) "
                        "before any usable text or tool call was produced. "
                        "This turn's output is lost - consider raising max_tokens "
                        "or splitting the task into smaller sub-steps.",
                        UserWarning,
                    )
                    message_text = (
                        "[TRUNCATED: this response hit the max_tokens limit "
                        f"({anthropic_params.get('max_tokens')}) before producing any "
                        "usable text or a complete tool call. No content was recovered. "
                        "This turn must be retried, ideally with a smaller sub-task or "
                        "a higher max_tokens setting.]"
                    )'''

    if old not in src:
        raise RuntimeError(
            "Could not find the expected code block to patch in "
            f"{path} - the ag2 version may differ from what this patch "
            "expects. Aborting without modifying the file."
        )

    src = src.replace(old, new)
    with open(path, "w") as f:
        f.write(src)
    print(f"Patched {path}")
