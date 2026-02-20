# Bazel conventions

- Never manually edit BUILD.bazel files to add targets. Run `bazel run //:gazelle` to
  regenerate them.
- Scripts run via `bazel run` have the workspace root at `$BUILD_WORKSPACE_DIRECTORY`, not `.`
- Python dependencies are tracked in `requirements.in` but in order to gazelle to be able
  to use it run `bazel run //:python-update`, you need to run this before gazelle when
  you add something to `requirements.in`. `requirements.txt` and `gazelle_python.yaml` are updated
  with this command.
