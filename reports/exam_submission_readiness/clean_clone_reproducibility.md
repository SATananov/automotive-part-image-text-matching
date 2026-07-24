# Clean-Clone Reproducibility Protocol

Use this protocol after the Step 010.8 commit is pushed.
It creates a new clone and does not reuse the development virtual environment.

```powershell
$Root = Join-Path $env:TEMP 'automotive-part-image-text-matching-step0108'
Remove-Item -LiteralPath $Root -Recurse -Force -ErrorAction SilentlyContinue
git clone https://github.com/SATananov/automotive-part-image-text-matching.git $Root
Set-Location $Root
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pytest -q
python -m src.project_cli verify-project
python -m src.project_cli verify-exam-submission-readiness
git branch --show-current
git rev-list --count HEAD
git status --short
```

Expected result:

- dependency installation succeeds from the committed lock file;
- the complete test suite passes;
- both verification commands report `PASS`;
- the current branch is `main` and history remains above 10 commits;
- `git status --short` prints no paths;
- the test split remains locked and unauthorized.

## Static readiness evidence

- Direct requirements: 9
- Fully pinned lock entries: 136
- TensorFlow lock: `tensorflow==2.21.0`
- Test functions present after Step 010.8: 346
- Final notebook code does not reference either locked test CSV.
- Step 010.8 reads the lock contract only; it does not load test rows.
