"""
One-time comprehensive patch: finds every remaining occurrence of the
unguarded `message = messages[-1]` pattern in conversable_agent.py and adds
an empty-messages guard before each one, so hitting max_rounds (or any other
scenario producing an empty per-sender message history) declines to reply
gracefully instead of crashing with IndexError.

Run patch_generate_function_call_reply.py FIRST - this patch's lookback
check skips any occurrence already guarded (marker present a few lines
above), so running these in this order matches the original site-packages
patch history exactly and avoids double-patching that occurrence.

Run once from the repo root:
    python patch_all_empty_messages_guards.py
"""

path = "autogen/agentchat/conversable_agent.py"
marker = "hep-theory fork: empty messages guard"

with open(path, "r") as f:
    lines = f.readlines()

already_patched_count = sum(1 for line in lines if marker in line)

target = "        message = messages[-1]\n"

raw_matches = [i for i, line in enumerate(lines) if line == target]

# Skip any occurrence that's already guarded (check a few lines above for
# our marker, to avoid double-patching the generate_function_call_reply
# occurrence fixed by the companion patch).
matches = []
for idx in raw_matches:
    lookback = lines[max(0, idx - 6):idx]
    if any(marker in line for line in lookback):
        continue
    matches.append(idx)

if not matches:
    print("No unguarded 'message = messages[-1]' lines found.")
    if already_patched_count:
        print(f"({already_patched_count} already patched from a previous run.)")
    raise SystemExit(0)

print(f"Found {len(matches)} occurrence(s) of the vulnerable pattern at lines: "
      f"{[m + 1 for m in matches]}")

insertion = [
    "        # hep-theory fork: empty messages guard - hitting max_rounds or\n",
    "        # other edge cases can leave this sender's message history empty;\n",
    "        # decline to reply rather than crashing with IndexError.\n",
    "        if not messages:\n",
    "            return False, None\n",
]

# Insert from the bottom up so earlier indices don't shift.
new_lines = lines[:]
for idx in sorted(matches, reverse=True):
    new_lines = new_lines[:idx] + insertion + new_lines[idx:]

with open(path, "w") as f:
    f.writelines(new_lines)

print(f"Patched: {path}")
print(f"Added empty-messages guards at {len(matches)} location(s).")
