Generating the docs
----------

Use [mkdocs](http://www.mkdocs.org/) structure to update the documentation.

Build locally with:

    uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir site

Serve locally with:

    uv run mkdocs serve --config-file docs/mkdocs.yaml

Build/serve via invoke tasks:

    uv run invoke build-docs
    uv run invoke serve-docs

GitHub Pages deployment:

- The workflow `.github/workflows/docs-pages.yaml` deploys documentation on every push to `main`.
- In the GitHub repo settings, set `Settings → Pages → Build and deployment → Source` to `GitHub Actions`.
- The published URL is `https://<user>.github.io/<repo>/` (so to get `/ml_ops_docs`, the repo must be named `ml_ops_docs`).
