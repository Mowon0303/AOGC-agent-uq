# MIRAGE-Bench test fixtures

Three real decision-point snapshots copied from MIRAGE-Bench (arXiv 2507.21017),
repo `sunblaze-ucb/mirage-bench`, `dataset_all/`, license Apache-2.0. Kept only as
small parser fixtures (format coverage: webarena AXTree, osworld a11y-tree,
taubench tool-call conversation). Full dataset is NOT vendored — clone the repo.

- webarena_unexpected_transition.json  : GUI snapshot, AXTree observation
- osworld_popup.json                   : GUI snapshot, accessibility-tree text
- taubench_users_questions.json        : tool-call conversation (Thought/Action ↔ API output)
