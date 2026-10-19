# External research source snapshots

This directory archives the source files required by the framework's existing
server configurations. It contains upstream code together with the local research
adaptations found on the authorized server on 2026-10-19. It is not a collection
of independently validated paper reproductions.

See [the handoff guide](../docs/source_handoff.md) for entry points, deployment,
dependencies and known limitations. The upstream repository, base commit, file
SHA-256 and whether each file was added or modified locally are recorded in
[source_manifest.json](source_manifest.json). Upstream README and available
license files are retained with their respective projects; the snapshots do not
change upstream authorship or licensing.

No datasets, generated sample files, model weights, indexes, credentials or
experiment outputs are intentionally included. The `SPV_MIA/data/` directory
contains Python preparation code only; `RAG_MIA/src/models/` contains Python model
adapters only.

```bash
python scripts/restore_external_sources.py
```

Run from the repository root to verify every archived file and parse all Python
sources without importing models or starting an experiment.
