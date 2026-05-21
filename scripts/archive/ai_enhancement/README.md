# Archived AI Enhancement Scripts

These scripts are preserved for reference only. They were used for experimental
Gemini-assisted test generation, documentation, refactoring, and integration.

They are archived because they are not part of the trading bot runtime and do
not currently meet the project's active tooling standards:

- They were created for the old `src/` layout rather than the refactored `app/` package.
- They can write generated AI output directly into project source and tests.
- They require external Gemini tooling and credentials that are not part of the core app setup.
- The integration logic needs a safer review/apply workflow before active use.

If this workflow is revived, rebuild it as opt-in developer tooling that writes
to a generated review folder first, never logs secrets, and only applies changes
after explicit confirmation.
