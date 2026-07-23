"""
One-time patch: fixes a pre-existing wheel-build bug in pyproject.toml,
unrelated to any of the Claude/Anthropic compatibility fixes in this fork.

BUG: [tool.hatch.build.targets.wheel]'s only-include listed
"autogen/agentchat/contrib/captainagent/tools" a second time even though
it's already covered by the "autogen" entry (packages = ["autogen"]
recursively includes everything underneath, and no exclude rule anywhere
in this file targets that subdirectory). hatchling then tries to add every
file under that subdirectory to the wheel twice, crashing with:

    ValueError: A second file is being added to the wheel archive at the
    same path: `autogen/agentchat/contrib/captainagent/tools/README.md`

This breaks `pip install git+https://...` (which builds a wheel from
source) even though the PyPI-published wheel works fine - suggesting the
maintainers' actual publish process doesn't go through this exact path,
or was built with an older hatchling version that didn't hit this check.

CONFIRMED via an actual wheel build, before and after this patch, that the
before state reproduces this exact crash and the after state builds
successfully with the previously-duplicated file appearing exactly once.

The paired [tool.hatch.build.targets.wheel.sources] entries were both
identity mappings (source path == destination path) serving no renaming
purpose, so removed along with the redundant only-include entry.

Run once from the repo root:
    python patch_wheel_build_duplicate_include.py
"""

path = "pyproject.toml"
marker = "hep-theory fork fix"

with open(path, "r") as f:
    content = f.read()

if marker in content:
    print(f"Already patched: {path}")
    raise SystemExit(0)

old = '''[tool.hatch.build.targets.wheel]
packages = ["autogen"]
only-include = ["autogen", "autogen/agentchat/contrib/captainagent/tools"]

[tool.hatch.build.targets.sdist]
exclude = ["test", "notebook"]

[tool.hatch.build.targets.wheel.sources]
"autogen" = "autogen"
"autogen/agentchat/contrib/captainagent/tools" = "autogen/agentchat/contrib/captainagent/tools"'''

new = '''[tool.hatch.build.targets.wheel]
# hep-theory fork fix: only-include previously listed
# "autogen/agentchat/contrib/captainagent/tools" a second time even though
# it's already covered by the "autogen" entry above (packages = ["autogen"]
# recursively includes everything underneath, and no exclude rule anywhere
# in this file targets that subdirectory) - hatchling then tried to add
# every file under it to the wheel twice, crashing with "A second file is
# being added to the wheel archive at the same path" on
# autogen/agentchat/contrib/captainagent/tools/README.md. The paired
# [tool.hatch.build.targets.wheel.sources] entries below were both
# identity mappings (source path == destination path) serving no renaming
# purpose, so removed along with the redundant only-include entry.
packages = ["autogen"]
only-include = ["autogen"]

[tool.hatch.build.targets.sdist]
exclude = ["test", "notebook"]'''

count = content.count(old)
if count != 1:
    raise SystemExit(
        f"Expected exactly 1 occurrence, found {count}. "
        "Paste the [tool.hatch.build.targets.wheel] section so this can be adjusted."
    )

content = content.replace(old, new, 1)

with open(path, "w") as f:
    f.write(content)

print(f"Patched: {path}")
