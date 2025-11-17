# ⚙️ VS Code Workspace Configuration — Crypto Trading Bot

This folder contains the modular VS Code configuration for the project.

## 📁 Why modular configuration?

Breaking settings into logical categories improves:

- Maintainability
- Clarity
- Cross-platform compatibility
- Team onboarding
- Automation with Copilot & CI

## 📦 Files

| File | Purpose |
|------|---------|
| `settings-core.json` | UI, formatting, editor behavior |
| `settings-python.json` | Interpreter, linting, formatting |
| `settings-copilot.json` | Copilot + instructions |
| `settings-codeium.json` | Codeium LLM integration |
| `settings-boolean.json` | Boolean flags like env activation |
| `settings-git.json` | Git + GitLens behavior |
| `settings-languages.json` | YAML, DockerCompose, workflows |
| `settings-data.json` | SQL, Jupyter |

## 🚀 How VS Code loads this config

VS Code reads these through:

- `settings.json`
- `workspace.json`
- `launch.json`
