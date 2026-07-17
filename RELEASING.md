# Releasing

`trail-lang` publishes to PyPI automatically when a version tag is pushed, via GitHub Actions
trusted publishing (OIDC, no stored token). See `.github/workflows/release.yml`.

## One-time setup

1. Reserve/create the project `trail-lang` on PyPI.
2. On PyPI, under the project's **Publishing** settings, add a GitHub trusted publisher:
   - Owner: `trail-language`
   - Repository: `trail-lang`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. In this repository's **Settings > Environments**, create an environment named `pypi`
   (optionally require a reviewer for publish approvals).

## Cutting a release

1. Bump `version` in `pyproject.toml`.
2. Commit, then tag and push:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
3. The `Release` workflow runs the tests, builds the sdist and wheel, and publishes to PyPI.

Installability is covered by CI-agnostic packaging: the wheel bundles the grammar and standard
library, so `pip install trail-lang` and `pipx install trail-lang` both yield a working `trail`
CLI with no source tree present.
