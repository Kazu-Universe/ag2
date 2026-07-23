"""
One-time patch: makes OpenAIWrapper.instantiate's template formatting
resilient to missing context keys, fixing an entire class of KeyError
crashes - agent YAML prompts referencing template variables
({inspirehep_context}, {current_code_output}, etc.) that aren't always
populated in context_variables, depending on which agents actually ran and
what they did.

Root cause: template.format(**context) raises KeyError on any missing key.
This patches the actual chokepoint instead of finding and removing
placeholders from YAML files one at a time as each one crashes a different
agent: missing keys now resolve to an empty string, matching how the code
already treats an entirely-missing/falsy context (returns the template
as-is at the top of the function) - just extended to the per-key case.

Run once from the repo root:
    python patch_safe_template_format.py
"""

path = "autogen/oai/client.py"
marker = "hep-theory fork: safe template formatting"

with open(path, "r") as f:
    content = f.read()

if marker in content:
    print(f"Already patched: {path}")
    raise SystemExit(0)

old = '            return template.format(**context) if allow_format_str_template else template'

new = (
    '            # hep-theory fork: safe template formatting - missing context keys\n'
    '            # (e.g. agent YAML prompts referencing {current_code_output} or\n'
    '            # {inspirehep_context} when that variable was never populated,\n'
    '            # because no code ran yet or no literature agent was used this step)\n'
    '            # resolve to an empty string instead of raising KeyError.\n'
    '            if allow_format_str_template:\n'
    '                class _SafeDict(dict):\n'
    '                    def __missing__(self, key):\n'
    '                        return ""\n'
    '                return template.format_map(_SafeDict(context))\n'
    '            return template'
)

count = content.count(old)
if count != 1:
    raise SystemExit(
        f"Expected exactly 1 occurrence, found {count}. Paste output so this can be adjusted."
    )

content = content.replace(old, new, 1)

with open(path, "w") as f:
    f.write(content)

print(f"Patched: {path}")
