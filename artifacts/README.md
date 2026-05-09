# Artifacts Directory

Generated failures, prompts, repair candidates, Pareto replay files, and checkpoints belong here.

Suggested layout:

```text
artifacts/
├── round_0/
│   ├── failures.jsonl
│   ├── repair_prompts.jsonl
│   ├── repair_candidates.jsonl
│   ├── scored_repair_candidates.jsonl
│   ├── replay.jsonl
│   ├── sft/
│   └── pfpo/
└── round_1/
```

