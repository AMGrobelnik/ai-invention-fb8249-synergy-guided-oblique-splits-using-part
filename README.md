# AI Invention Research Repository

This repository contains artifacts from an AI-generated research project.

## Quick Start - Interactive Demos

Click the badges below to open notebooks directly in Google Colab:

### Jupyter Notebooks

| Folder | Description | Open in Colab |
|--------|-------------|---------------|
| `dataset_iter1_tabular_bench` | Tabular Classification Benchmarks for SG-FIGS Eval... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-fb8249-synergy-guided-oblique-splits-using-part/blob/main/dataset_iter1_tabular_bench/demo/data_code_demo.ipynb) |
| `dataset_iter1_pid_synergy_dat` | PID Synergy Matrices, Timing, MI Comparison & Stab... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-fb8249-synergy-guided-oblique-splits-using-part/blob/main/dataset_iter1_pid_synergy_dat/demo/data_code_demo.ipynb) |
| `dataset_iter2_sg_figs_dataset` | 4 OpenML Datasets for SG-FIGS (monks2, blood, clim... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-fb8249-synergy-guided-oblique-splits-using-part/blob/main/dataset_iter2_sg_figs_dataset/demo/data_code_demo.ipynb) |
| `experiment_iter2_pid_synergy_exp` | Pairwise PID Synergy Matrices on Benchmark Dataset... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-fb8249-synergy-guided-oblique-splits-using-part/blob/main/experiment_iter2_pid_synergy_exp/demo/method_code_demo.ipynb) |
| `experiment_iter2_sg_figs_results` | SG-FIGS: Full Experiment Implementation and Benchm... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-fb8249-synergy-guided-oblique-splits-using-part/blob/main/experiment_iter2_sg_figs_results/demo/method_code_demo.ipynb) |
| `evaluation_iter3_sg_figs_eval` | Statistical Evaluation of SG-FIGS Experiment Resul... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-fb8249-synergy-guided-oblique-splits-using-part/blob/main/evaluation_iter3_sg_figs_eval/demo/eval_code_demo.ipynb) |
| `experiment_iter3_sg_figs_compare` | SG-FIGS Definitive Comparison Experiment... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-fb8249-synergy-guided-oblique-splits-using-part/blob/main/experiment_iter3_sg_figs_compare/demo/method_code_demo.ipynb) |
| `evaluation_iter4_sg_figs_eval` | Final Integrated Research Synthesis for SG-FIGS... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-fb8249-synergy-guided-oblique-splits-using-part/blob/main/evaluation_iter4_sg_figs_eval/demo/eval_code_demo.ipynb) |
| `evaluation_iter4_sg_figs_eval` | Definitive Statistical Evaluation of SG-FIGS Hypot... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-fb8249-synergy-guided-oblique-splits-using-part/blob/main/evaluation_iter4_sg_figs_eval/demo/eval_code_demo.ipynb) |
| `experiment_iter4_sg_figs_equalcx` | Complexity-Matched SG-FIGS Experiment: Synergy vs ... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-fb8249-synergy-guided-oblique-splits-using-part/blob/main/experiment_iter4_sg_figs_equalcx/demo/method_code_demo.ipynb) |

### Research & Documentation

| Folder | Description | View Research |
|--------|-------------|---------------|
| `research_iter1` | SG-FIGS Spec... | [![View Research](https://img.shields.io/badge/View-Research-green)](https://github.com/AMGrobelnik/ai-invention-fb8249-synergy-guided-oblique-splits-using-part/blob/main/research_iter1/demo/research_demo.md) |

## Repository Structure

Each artifact has its own folder with source code and demos:

```
.
├── <artifact_id>/
│   ├── src/                     # Full workspace from execution
│   │   ├── method.py            # Main implementation
│   │   ├── method_out.json      # Full output data
│   │   ├── mini_method_out.json # Mini version (3 examples)
│   │   └── ...                  # All execution artifacts
│   └── demo/                    # Self-contained demos
│       └── method_code_demo.ipynb # Colab-ready notebook (code + data inlined)
├── <another_artifact>/
│   ├── src/
│   └── demo/
├── paper/                       # LaTeX paper and PDF
├── figures/                     # Visualizations
└── README.md
```

## Running Notebooks

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badges above to run notebooks directly in your browser.
No installation required!

### Option 2: Local Jupyter

```bash
# Clone the repo
git clone https://github.com/AMGrobelnik/ai-invention-fb8249-synergy-guided-oblique-splits-using-part.git
cd ai-invention-fb8249-synergy-guided-oblique-splits-using-part

# Install dependencies
pip install jupyter

# Run any artifact's demo notebook
jupyter notebook exp_001/demo/
```

## Source Code

The original source files are in each artifact's `src/` folder.
These files may have external dependencies - use the demo notebooks for a self-contained experience.

---
*Generated by AI Inventor Pipeline - Automated Research Generation*
