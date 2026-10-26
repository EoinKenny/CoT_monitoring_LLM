# When nearest neighbours help an LLM judge

![Nearest-neighbour explanations for LLM judges](assets/title_figure.png)

Twelve judges, five backbones, ten datasets. Includes frozen scores, annotations,
plot inputs, exact prompts, and experiment sources. **No GPU or API key is needed
for the figures.**

[Download prompts and raw responses](docs/RAW_DATA.md) — optional data supplement,
with source mappings and checksums.

## Reproduce all figures

```bash
git clone https://github.com/EoinKenny/CoT_monitoring_XAI.git
cd CoT_monitoring_XAI
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python reproduce_results.py
```

Open `outputs/paper/figures.pdf` or `outputs/paper/index.html`.
This redraws the plots in [the reference PDF](docs/reference_figures.pdf), plus
all eight judge/genre appendix grids, from bundled numerical data.
To rebuild the statistics from per-example scores and labels too:

```bash
python reproduce_results.py --recompute --out outputs/recomputed
```

## Optional: collect new model responses

Configure your local model server or API endpoints in a copy of
[`configs/collection.example.json`](configs/collection.example.json), then:

```bash
python reproduce_results.py --query-models configs/my_models.json --dry-run
python reproduce_results.py --query-models configs/my_models.json --out outputs/new_collection
```

This makes new, potentially paid requests on the frozen probe/reference inputs;
it does not retrain backbones or change the published data. Raw responses,
failures, configuration, and score arrays are saved separately. Re-running resumes
only unattempted requests. [Collection options](docs/COLLECTION.md) ·
[Data and methods](docs/REPRODUCTION.md) · [Historical sources](docs/REPRODUCING_LEGACY.md).
