# RATED: Reliability-Aware AI Agent Tool Retrieval via Execution Dynamics

RATED combines **robust semantic matching** with **dynamic reliability adaptation** so that AI agents retrieve tools that are both relevant and executable. In deployment, tools often fail due to API outages, schema changes, or auth issues; RATED uses a frozen bi-encoder for semantics and a lightweight **contextual bandit** that re-ranks candidates from **real-time execution feedback**, avoiding broken tools without retraining the encoder.


## Repository structure

```
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── config/                      # Env files (e.g. gemini.env, openai.env); do not commit secrets
├── data/
│   ├── raw/                     # tools_*.jsonl, toolbench_test_set.json, etc.
│   ├── processed/               # tools_clean_with_reliability.csv, queries_clean_*.csv, tools_index, queries_index
│   ├── ecoagent_real_toolbench/ # EcoAgent-Real: eval_real_pairs.csv, eval_real_*.jsonl, train/val_positive_pairs.csv
│   └── ecoagent_syn/            # EcoAgent-Syn: train_pairs.csv, val_pairs.csv, all_pairs.csv, val_retrieval.jsonl
└── scripts/
    ├── data_collection/
    │   ├── build_ecoagent_real.py
    │   └── collect_tools.py
    ├── pair_generation/
    │   └── generate_ecoagent_syn.py
    ├── training/
    │   ├── train_rated_joint.py
    │   ├── train_bandit_warmup.py
    │   ├── train_rated_wo_contrastive.py
    │   ├── train_sbert_xgboost.py
    │   └── reward_function.py
    └── evaluation/
        └── evaluate_rated_full.py
```
## Requirements

```bash
pip install -r requirements.txt
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.
