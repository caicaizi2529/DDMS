# DDMS

PyTorch/DGL implementation for the DDMS paper.

## Layout

```text
src/        training scripts
data/       compressed preprocessed datasets
```

## Setup

```bash
pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch and a DGL build compatible with it. The training
scripts call CUDA directly, so a GPU is required.

## Data

The repository retains the supplied compressed `dataset.pkl` files. Extract
each archive in place before training:

```powershell
Expand-Archive data\ciao\dataset.zip -DestinationPath data\ciao -Force
Expand-Archive data\epinions\dataset.pkl.zip -DestinationPath data\epinions -Force
Expand-Archive data\yelp\dataset.pkl.zip -DestinationPath data\yelp -Force
```

Each script expects `data/<name>/dataset.pkl`. The extracted files are ignored
by Git; the archives remain versioned. Dataset provenance and redistribution
terms were not present in this project, so confirm permission before making a
public repository.

## Run

Run commands from the repository root. For example:

```bash
python src/all-ciao.py --cuda 0
python src/all-epinions.py --cuda 0
python src/all-yelp.py --cuda 0
```

| Script | Variant |
| --- | --- |
| `all-*.py` | Full model, dataset-specific defaults |
| `ablation1.py` | Interaction-diffusion only |
| `ablation2.py` | Social-diffusion only |
| `ablation3.py` | End-to-end variant without staged freezing |

Logs, checkpoints, and search results are created locally and excluded from
version control. No GitHub repository has been created or uploaded.
