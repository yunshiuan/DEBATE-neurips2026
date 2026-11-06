# Manuscript reference data

These CSV and JSON files store the manuscript's numerical reference values and result mappings. They are used only for verification; reproduction computes results independently from the released input data.

These data are covered by the [DEBATE Dataset Research-Only License](LICENSE), rather than the MIT license covering the Python code. Historical reported aggregates are explicitly distinguished from independently recomputed results in `../coverage_manifest.json`.

`cross_model_table_values.json` contains the 100 displayed values in the three cross-model tables from camera-ready paper commit `e2df1ff` (2026-11-06). These are verification targets only. The table stage independently recomputes convergence and public/private prediction errors from the released cached observations; its separate family-reference CSV preserves the original reference values.
