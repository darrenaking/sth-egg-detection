# STH Egg Detection

Domain-robust detection and classification of soil-transmitted helminth (STH) eggs in microscopy images, using Faster R-CNN with optimal transport-based domain adaptation.

## Background

STH infections affect ~1.5 billion people. Diagnosis relies on counting parasite eggs under a microscope, which requires trained microscopists who are scarce in endemic regions. Automated detection could help, but models trained in one lab fail when deployed to another due to differences in microscopes, staining, and slide preparation. This project investigates whether optimal transport-based domain adaptation can make detection robust across sites.

## Project structure

```
sth-egg-detection/
├── data/
│   ├── raw/               — original datasets, never modified
│   └── processed/         — derived data, safe to delete and regenerate
├── notebooks/             — Try stuff here. Move reusable code to src/
├── src/
│   ├── data/              — dataset loading code
│   ├── models/            — model definitions
│   └── utils/             — shared helpers (visualization, metrics)
├── configs/               — experiment settings (YAML files)
├── scripts/               — command-line scripts (download, train, evaluate)
├── experiments/           — training run outputs, one subfolder per run
└── docs/                  — plain-language notes and design decisions
```

## Setup
```bash
git clone https://github.com/darrenaking/sth-egg-detection.git
cd sth-egg-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The primary dataset is Chula-ParasiteEgg-11 from the ICIP 2022 Parasitic Egg Detection Challenge (~13,200 microscopy images, 11 species, COCO-format annotations). Download from [HuggingFace](https://huggingface.co/datasets/pui-nantheera/Parasitic_Egg_Detection_and_Classification_in_Microscopic_Images/tree/main) or [IEEE](https://ieee-dataport.org/competitions/parasitic-egg-detection-and-classification-microscopic-images)

Do not commit data files to git.

## Conventions

- Clear notebook outputs before committing: `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`
- Commit messages use imperative tense: "Add feature" not "Added feature"
