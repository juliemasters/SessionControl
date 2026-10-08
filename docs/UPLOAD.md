# Upload to SessionControl

Target repository: [juliemasters/SessionControl](https://github.com/juliemasters/SessionControl).

This release was prepared against main commit `c789336f71817b8444cc84dc9dd4bfb3a6d8e533`. It preserves the existing prototype and adds the connected activation-routing experiment, measured reports, reproduction tooling, CPU checks, and third-party notices. Preparation did not push or create a remote branch.

The download ZIP contains the complete public working tree under `SessionControl/`, without `.git`, checkpoints, dense deltas, raw dataset conversations, caches, or credentials. Extract it to inspect the files. The companion patch contains the changes relative to the recorded main commit and can be applied in a clone.

To upload with Git, clone the target repository, create a branch, and apply the supplied patch:

```bash
git clone https://github.com/juliemasters/SessionControl.git
cd SessionControl
git switch -c experiment-results
git apply --check /path/to/SessionControl-experiment-release.patch
git apply /path/to/SessionControl-experiment-release.patch
python3 scripts/verify_release.py
git add .
git diff --cached --check
git diff --cached --stat
git commit -m "Add activation routing experiments and measured memory comparison"
git push -u origin experiment-results
```

The last command publishes the user's local branch; it was not executed during preparation. Review the branch and open a pull request into main. If main has changed since the recorded commit, resolve those changes before uploading rather than overwriting them.

For GitHub's web upload, extract the ZIP and upload its contents. Preserve hidden files such as `.gitignore` and `.github/workflows/cpu-checks.yml`. Upload the files, rather than committing the ZIP as a single artifact. The ZIP contains an already complete tree; it is not another nested project to put inside an existing `SessionControl/` directory.
