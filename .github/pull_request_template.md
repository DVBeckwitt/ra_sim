## Summary

- change:
- why:

## Validation

- [ ] `ruff check .`
- [ ] `pytest -q`
- [ ] `python -m mypy ra_sim/config ra_sim/simulation ra_sim/fitting ra_sim/gui`

## Data And Security

- [ ] No machine-local absolute paths added outside example templates
- [ ] No secrets, tokens, keys, or private experiment data committed
- [ ] Local config changes stay in ignored override files, not tracked examples
