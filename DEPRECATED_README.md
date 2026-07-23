# DEPRECATED — wrong base

This branch was believed to match the real published `cmbagent_autogen`
package (matched on the generic AG2 `group/` module structure) but turned
out to diverge significantly in cmbagent-specific modifications baked into
`conversable_agent.py` and `oai/client.py`. Superseded by `cmbagent-real-base`,
which is built directly from the real published wheel's bytes instead of a
guessed GitHub branch.
