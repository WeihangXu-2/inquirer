Inquirer (Streamlit App)

A Streamlit-based application for high-recall document and case retrieval with multi-phase processing.
Designed for large corpora and iterative analysis, with an emphasis on accuracy first, precision later.

Features

Streamlit UI for interactive querying and review

Multi-phase retrieval pipeline

Phase 1: lightweight query preparation

Phase 2: high-recall lexical retrieval (BM25 / TF-IDF style)

Phase 3: optional reranking / enrichment

Handles large text corpora (bronze/silver/gold style layers)

Caching and session controls to keep performance usable during heavy runs

Troubleshooting

App shows “Error running app”

Usually caused by memory exhaustion or cache bloat

Clear cache and restart

Reduce corpus size or retrieval depth for interactive use

App runs once, then crashes on repeat

Heavy objects are likely accumulating in cache or session state

Restart and clear cache before rerunning

Disclaimer

This project is optimized for research and exploratory analysis, not production deployment.
Expect high memory usage during intensive phases.
