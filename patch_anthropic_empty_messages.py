"""
One-time patch: fixes oai_messages_to_anthropic_messages crashing with
IndexError when processed_messages is empty (happens during forced-termination
sequences at max_rounds, same root cause as the conversable_agent.py guards).

Unlike those guards (which decline to reply), this fix preserves the actual
intent of the code: ensure the conversation ends in a 'user' turn before
sending to the Anthropic API. An empty list trivially doesn't end in 'user',
so it should get the continue message appended, not crash.

Run once from the repo root:
    python patch_anthropic_empty_messages.py
"""

path = "autogen/oai/anthropic.py"
marker = "hep-theory fork: empty processed_messages fix"

with open(path, "r") as f:
    content = f.read()

if marker in content:
    print(f"Already patched: {path}")
    raise SystemExit(0)

old = '    if processed_messages[-1]["role"] != "user":'
new = (
    '    # hep-theory fork: empty processed_messages fix - an empty list trivially\n'
    '    # doesn\'t end in "user", so it should get the continue message appended,\n'
    '    # not crash with IndexError (happens during forced-termination sequences\n'
    '    # at max_rounds).\n'
    '    if not processed_messages or processed_messages[-1]["role"] != "user":'
)

if content.count(old) != 1:
    raise SystemExit(
        f"Expected exactly 1 occurrence, found {content.count(old)}. "
        "Paste output so this can be adjusted."
    )

content = content.replace(old, new, 1)

with open(path, "w") as f:
    f.write(content)

print(f"Patched: {path}")
