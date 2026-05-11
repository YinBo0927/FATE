<div align="center">
<h1><a href="#" target="_blank">On-Policy Self-Evolution via Failure Trajectories for Agentic Safety Alignment</a></h1>

<div>
<a target="_blank" href="#">
  <img src="https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg" alt="arXiv Paper"/>
</a>
<a target="_blank" href="https://yinbo0927.github.io/FATE/">
  <img src="https://img.shields.io/badge/Project-Page-2f80ed.svg" alt="Project Page"/>
</a>
</div>
</div>

------------------

**Authors:**<br>
<a href="https://scholar.google.com/citations?user=YSA01EoAAAAJ&hl=zh-CN">Bo Yin*</a>, <a href="https://scholar.google.com/citations?user=gsV-g2wAAAAJ&hl=zh-CN">Qi Li*</a>, <a href="https://scholar.google.com/citations?user=w69Buq0AAAAJ&hl=en">Xinchao Wang<sup>&dagger;</sup></a>

National University of Singapore

<sup>*</sup> Equal contribution.<br>
<sup>&dagger;</sup> Corresponding author: [xinchao@nus.edu.sg](mailto:xinchao@nus.edu.sg)

------------------

## Overview

<div align="center">
    <div style="max-width: 100%; text-align: left; margin-bottom: 20px;">
        <img src="assets/fate_figure1.png" alt="FATE framework" style="display: block; margin: 0 auto; width: 100%;">
    </div>
</div>

<strong>Figure 1.</strong> FATE mines failure trajectories, asks the current policy to propose repairs, filters candidates with verifier-scored multi-objective Pareto selection, and updates the policy with SFT and PFPO.

## Results

### AgentDojo

ASR and BRR are lower-is-better. TSR is higher-is-better.

| Backbone | Method | ASR ↓ | TSR ↑ | BRR ↓ |
| --- | --- | ---: | ---: | ---: |
| Qwen3-8B-Instruct | Base | 0.812 | 0.132 | 0.104 |
| Qwen3-8B-Instruct | FATE | **0.540** | **0.392** | **0.082** |
| Llama-3.1-8B-Instruct | Base | 0.768 | 0.158 | 0.118 |
| Llama-3.1-8B-Instruct | FATE | **0.512** | **0.417** | **0.087** |
| Ministral-3-8B-Instruct | Base | 0.736 | 0.176 | 0.096 |
| Ministral-3-8B-Instruct | FATE | **0.486** | **0.438** | **0.074** |
| Gemma-3-12B-it | Base | 0.704 | 0.204 | 0.132 |
| Gemma-3-12B-it | FATE | **0.468** | **0.462** | **0.091** |
| Phi-4-reasoning | Base | 0.748 | 0.168 | 0.126 |
| Phi-4-reasoning | FATE | **0.503** | **0.429** | **0.089** |

### AgentHarm

HCR is lower-is-better. VRR and SafeScore are higher-is-better.

| Backbone | Method | HCR ↓ | VRR ↑ | SafeScore ↑ |
| --- | --- | ---: | ---: | ---: |
| Qwen3-8B-Instruct | Base | 0.719 | 0.156 | 0.241 |
| Qwen3-8B-Instruct | FATE | **0.125** | **0.812** | **0.870** |
| Llama-3.1-8B-Instruct | Base | 0.672 | 0.188 | 0.286 |
| Llama-3.1-8B-Instruct | FATE | **0.156** | **0.781** | **0.842** |
| Ministral-3-8B-Instruct | Base | 0.641 | 0.219 | 0.314 |
| Ministral-3-8B-Instruct | FATE | **0.141** | **0.797** | **0.858** |
| Gemma-3-12B-it | Base | 0.625 | 0.234 | 0.337 |
| Gemma-3-12B-it | FATE | **0.172** | **0.766** | **0.821** |
| Phi-4-reasoning | Base | 0.688 | 0.203 | 0.301 |
| Phi-4-reasoning | FATE | **0.164** | **0.781** | **0.836** |

### Iterative Self-Evolution

![Iterative self-evolution curves](assets/results_evolution.png)

### ATBench External Generalization

| Model | ATBench-C Acc. ↑ | ATBench-C Prec. ↑ | ATBench-C Rec. ↑ | ATBench-C F1 ↑ | ATBench-F R.S. ↑ | ATBench-F F.M. ↑ | ATBench-F R.H. ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AgentDoG-Qwen3-4B | 64.0 | 59.2 | 88.9 | 71.1 | 46.8 | 16.5 | 40.6 |
| Qwen3-8B-Instruct + FATE | **77.8** | **80.5** | 78.6 | **79.5** | **49.2** | **18.4** | **43.1** |

## Installation

1. Create the conda environment and install dependencies.

```bash
git clone https://github.com/YinBo0927/FATE.git
cd FATE
conda create -n fate python=3.10 -y
conda activate fate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
```

## Run

1. Download the benchmark assets and create output folders.

```bash
python FATE_main_Prepare.py
```

2. Run benchmark rollouts on the dev split and save verifier-scored failures.

```text
artifacts/round_0/dev_rollouts.scored.jsonl
```

3. Generate repair candidates from failed trajectories.

```bash
python FATE_main_BuildReplay.py
```

4. Score the generated repair candidates with benchmark verifiers, save the file below, then rerun replay construction.

```text
artifacts/round_0/repair_candidates.verified.jsonl
```

```bash
python FATE_main_BuildReplay.py
```

5. Train the SFT model and sample PFPO candidate groups.

```bash
python FATE_main_Train.py
```

6. Score PFPO candidate groups with benchmark verifiers, save the file below, then rerun training.

```text
artifacts/round_0/pfpo_groups.verified.jsonl
```

```bash
python FATE_main_Train.py
```

7. Export ATBench for external trajectory-safety evaluation.

```bash
python FATE_main_ATBench.py
```

See [docs/reproduction.md](docs/reproduction.md) for JSONL schemas.

## Citation

If you find this work interesting or helpful, please cite:

```bibtex
@article{yin2026fate,
  title={On-Policy Self-Evolution via Failure Trajectories for Agentic Safety Alignment},
  author={Yin, Bo and Li, Qi and Wang, Xinchao},
  journal={arXiv preprint arXiv:coming soon},
  year={2026}
}
```
