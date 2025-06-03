# 🧑‍💻 Contributing Guidelines

Welcome! We're glad you're considering contributing to this project. Please read through the following guidelines before creating an issue, branch, or pull request.

---

## 📌 Issue Guidelines

- Each **feature**, **bug fix**, or **improvement** must be tracked by a GitHub **issue**.
- The issue must include:
  - A **unique identification number**
  - A **clear and detailed description** of the task
  - A **label** categorizing it (e.g., `feature`, `bug`, `enhancement`)

---

## 🌱 Branch Naming Convention

Branches must follow this format:

```
<prefix>/<issue-number>-<short-description>
```

### Examples:
- `feature/123-add-login-form`
- `fix/456-crash-on-logout`
- `chore/789-update-dependencies`

### Allowed Prefixes:
- `feature` – for new features
- `fix` – for bug fixes
- `chore` – for internal tooling or maintenance
- `docs` – for documentation updates
- `refactor` – for code restructuring
- `test` – for adding or improving tests

---

## 🚧 Main Branch Rules

- ✅ Only **administrators** can push or merge to the `main` branch.
- ✅ All changes must come from a **pull request (PR)**.
- ✅ Only PRs from the `develop` branch are allowed into `main`.

> These rules are enforced via GitHub branch protection settings and CI workflows.

---

## 🌿 Develop Branch

- `develop` is the **default working branch**.
- All contributors should branch off `develop`, not `main`.
- Merge your changes into `develop` via a PR with valid commit messages.

---

## 📝 Commit Message Format (Conventional Commits)

All commit messages **must follow [Conventional Commits](https://www.conventionalcommits.org/)**. This is **strictly enforced** using GitHub Actions with the following regular expression:

^(feat|fix|chore|docs|style|refactor|perf|test)(\([\w\-]+\))?: .{1,}

### ✅ Valid Examples:
- `feat: add user login functionality`
- `fix(api): correct auth header`
- `chore: bump dependencies`
- `docs(readme): add setup instructions`

### ❌ Invalid Examples:
- `update login`
- `fixed bug`
- `refactored stuff`

> Commits that don't match will fail the PR check and block merging.

---

## ✅ Pull Requests

- Pull Requests should:
  - Reference the related issue (e.g., `Closes #123`)
  - Use a Conventional Commit-style title (e.g., `feat: add login form`)
  - Be scoped to a single logical change or issue
- Ensure all status checks pass before requesting review
- Keep the PR description clear and informative

---


## 🛠️ Enforcing Conventional Commits Locally (Python Project)

To ensure consistent commit messages, we enforce the [Conventional Commits](https://www.conventionalcommits.org/) standard using a Python-native tool.

---

### ✅ Recommended: Use `pre-commit` + `gitlint`

This setup runs checks automatically when you make a commit — no Node.js required.

---

### 🔧 Step-by-Step Setup

#### 1. Install `pre-commit`

Install the tool globally or add it to your project:

```bash
pip install pre-commit
```

> You can also include it in your `requirements-dev.txt` or `pyproject.toml`.

---

#### 2. Create a `.pre-commit-config.yaml` file

```yaml
repos:
  - repo: https://github.com/jorisroovers/gitlint
    rev: v0.19.1  # Use the latest version
    hooks:
      - id: gitlint
```

---

#### 3. Install the Git hook

```bash
pre-commit install
```

This installs the hook that will run automatically on every commit.

---

#### 4. (Optional) Customize commit rules in `.gitlint`

Create a `.gitlint` file in your project root:

```ini
[general]
ignore-merge-commits=true

[title-match-regex]
regex=^(feat|fix|chore|docs|style|refactor|perf|test)(\([\w\-]+\))?: .{1,}$
```

This enforces the Conventional Commit format.

---

### 🧪 Example Valid Commit

```
feat(core): add new API handler
```

### ❌ Example Invalid Commit

```
added new feature
```

---

### ✅ Benefits

- Enforced **before you commit**
- Fully **Python-native**
- No need for Node.js or extra tooling
- Can be used in local dev + CI

---

For more details, see:

- [https://pre-commit.com](https://pre-commit.com)
- [https://jorisroovers.com/gitlint](https://jorisroovers.com/gitlint/)
- [https://www.conventionalcommits.org](https://www.conventionalcommits.org)

⸻

Thank you for helping us build and maintain this project! 🎉
