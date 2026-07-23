"""
One-time patch: adds an empty-messages guard to generate_function_call_reply,
mirroring the earlier group_tool_executor.py fix. Without this, hitting
max_rounds (or any other scenario where an agent's message history with a
given sender is empty) crashes with IndexError instead of gracefully
declining to reply.

Adapted from the original site-packages version (which located the
installed module via importlib) to patch this forked repo's own source by
a fixed relative path, since the fix is now meant to live here permanently
rather than as a local monkey-patch re-applied after every fresh install.

Run once from the repo root:
    python patch_generate_function_call_reply.py
"""

path = "autogen/agentchat/conversable_agent.py"
marker = "hep-theory fork: empty messages guard"

with open(path, "r") as f:
    lines = f.readlines()

if any(marker in line for line in lines):
    print(f"Already patched: {path}")
    raise SystemExit(0)

target = "        message = messages[-1]\n"
target_context = '        if message.get("function_call"):\n'

matches = [
    i for i in range(len(lines) - 1)
    if lines[i] == target and lines[i + 1] == target_context
]

if len(matches) != 1:
    print(f"Expected exactly 1 match, found {len(matches)}.")
    raise SystemExit("Aborting - paste output so this can be adjusted.")

idx = matches[0]
insertion = [
    "        # hep-theory fork: empty messages guard - hitting max_rounds or\n",
    "        # other edge cases can leave this sender's message history empty;\n",
    "        # decline to reply rather than crashing with IndexError.\n",
    "        if not messages:\n",
    "            return False, None\n",
]

new_lines = lines[:idx] + insertion + lines[idx:]

with open(path, "w") as f:
    f.writelines(new_lines)

print(f"Patched: {path} (inserted guard before line {idx + 1})")
