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

## 🛠️ Local Setup for Commit Enforcement (Optional but Recommended)

To enforce Conventional Commits locally before pushing:

1. Install dev dependencies:

   ```bash
   npm install --save-dev @commitlint/{config-conventional,cli} husky
      ```

	2.	Add this config to commitlint.config.js:

      ```
module.exports = {
  extends: ['@commitlint/config-conventional'],
};
   ```


	3.	Enable Git hooks:

   ```
npx husky install
   ```

	4.	Add this to package.json scripts:

   ```
"scripts": {
  "prepare": "husky install"
}
   ```

	5.	Create a commit-msg hook:

   ```
npx husky add .husky/commit-msg 'npx --no -- commitlint --edit "$1"'
   ```


This prevents invalid commits before they’re even pushed.

⸻

Thank you for helping us build and maintain this project! 🎉
