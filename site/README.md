# site/ — Strategy Audit landing page

Static, single-file landing page (`index.html`) for the "Strategy Audit" service.
No build step, no bundler, no external JS. The only external network resource is
Google Fonts (`fonts.googleapis.com` / `fonts.gstatic.com`); everything else is
inline CSS/SVG in the one HTML file.

## Before publishing — fill in the placeholders

`index.html` contains these placeholders. Replace every one before going live:

| Placeholder | Where | What it is |
| --- | --- | --- |
| `[CONTACT_URL]` | nav, hero, pricing, FAQ, footer | mailto: link or LinkedIn/contact URL | — FILLED (LinkedIn, same as CASE-STUDY)
| `[REPO_URL] — FILLED (https://github.com/imadeptus/quant-harness)` | hero, calibration, footer | link to the public repository / methodology docs |
| `{{PRICE_AUDIT_BASIC}} — FILLED (from $400, GTM hypothesis)` | Pricing card 1 | price for a single-strategy audit |
| `{{PRICE_AUDIT_FULL}} — FILLED (from $1,200, GTM hypothesis)` | Pricing card 2 | price for the full audit incl. cost sensitivity |
| `{{PRICE_API_PER_CALL}} — FILLED ($0.05, GTM hypothesis)` | Pricing card 3 | price per `POST /v1/verdict` call |
| `{{WALLET_ADDRESS}} — REPLACED by invoice-per-engagement flow (no address on the page)` | "Pay in stablecoins" box | USDC/USDT receiving address |
| `{{WALLET_NETWORKS}} — REPLACED (network stated on each invoice)` | "Pay in stablecoins" box | supported networks, e.g. "Ethereum, Base, Polygon" |

`grep -no '\[[A-Z_]*\]\|{{[A-Z_]*}}' index.html` finds every occurrence.

## Enable GitHub Pages

1. Push this repository (or just the `site/` contents) to GitHub.
2. Repository → **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **Deploy from a branch**.
4. Pick the branch (e.g. `main`) and folder:
   - If `index.html` lives at the repo root, choose `/ (root)`.
   - If it stays under `site/`, either set the folder to `/site` (if your repo
     supports a custom subfolder — GitHub only offers `/root` or `/docs`) or
     copy/symlink `site/index.html` to `/docs/index.html` and select `/docs`.
5. Save. GitHub Pages builds and serves the page at
   `https://<user>.github.io/<repo>/` (or a custom domain if configured under
   **Settings → Pages → Custom domain**).
6. Re-run step 4 any time you replace the placeholders above with real values.

No Jekyll processing is required — add an empty `.nojekyll` file next to
`index.html` if GitHub Pages tries to run Jekyll over it and drops any
underscore-prefixed asset (not currently applicable, since this page has
none, but harmless to add pre-emptively).

## Local preview

No server needed — open the file directly:

```bash
open site/index.html   # macOS, from the repository root
```

or serve it locally to test relative behavior the same way GitHub Pages would:

```bash
cd site && python3 -m http.server 8080
# then open http://localhost:8080/
```

## What's on the page

Hero → why backtests lie → sample PASS/KILL report → who it's for → how it
works (self-serve `qh-audit` CLI and hosted API) → pricing → calibration
numbers → FAQ → footer disclaimer. All factual numbers (0% false-positive
rate, ~2.2 Sharpe detection threshold, 10 pre-registered hypotheses / 3
exchanges, 198 tests) are pulled from
`../quant-harness/reports/CALIBRATION.md` and `../CASE-STUDY-EN.md` — update
this page if those source documents change.

Theming: light/dark via `prefers-color-scheme`, plus a manual toggle button
that overrides the system preference and persists the choice in
`localStorage` (falls back silently if storage is unavailable).
