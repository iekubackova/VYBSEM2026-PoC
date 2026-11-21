# Simple baselines for CRISPR perturbation prediction (proof of concept)

Code for the paper *How Strong Are Simple Baselines for Predicting CRISPR Perturbation Effects?
A Laptop-Scale Dashboard* (Ivana Ella Kubačková and Martin Blažek, VYBSEM 2026, FIIT STU Bratislava).

A small Streamlit dashboard that reproduces the comparison reported by
Ahlmann-Eltze et al. (Nature Methods 2025) and Csendes et al. (BMC Genomics 2025):
can very simple predictors forecast the transcriptional effect of a CRISPR
perturbation of a gene that was never seen during fitting?

## What it does

1. **DataLoader** – reads a GEARS-processed Perturb-seq dataset (Adamson 2016, CRISPRi;
   Norman 2019, CRISPRa), keeps control cells and single-gene perturbations and averages
   cells into one pseudobulk profile per perturbation. A perturbation is used only when
   its target gene is measured, has a GO annotation and has a GEARS list of 20 DE genes
   (86 perturbations in Adamson, 102 of 105 in Norman).
2. **Split** – GEARS-style "simulation" split: a random 75 % of perturbed genes are used for
   fitting, the remaining 25 % are unseen test genes. Repeated over several seeds.
3. **Predictors**
   - `MeanPredictor` – the average training profile, the same for every test gene.
   - `PCAPredictor` – linear model `Y ≈ G W Pᵀ + b` with 10 principal components and
     ridge penalty 0.1 (Ahlmann-Eltze et al.).
   - `ForestPredictor` – random forest on 256 truncated-SVD components of a binary
     gene × GO-term matrix (after Csendes et al.).
4. **Scorer** – MSE over all genes, MSE over the 20 most differentially expressed
   genes (GEARS list), Pearson correlation of the predicted change versus control
   (Pearson delta) and the overlap of the top-20 predicted and true DE genes.
5. **Dashboard** – score table, box plot of scores per split and a predicted-vs-measured
   scatter plot for any unseen gene.

No model training on a GPU is needed; a full run takes a few minutes on a laptop.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Download into `data/` (do not unzip, the loader does that):

| file | URL |
|------|-----|
| `adamson.zip` | https://dataverse.harvard.edu/api/access/datafile/6154417 |
| `norman.zip` | https://dataverse.harvard.edu/api/access/datafile/6154020 |
| `gene2go_all.pkl` (a browser may save it as `gene2go.pkl`; both names work) | https://dataverse.harvard.edu/api/access/datafile/6153417 |

## Run

```bash
streamlit run app.py          # interactive dashboard
python evaluate.py            # headless: writes results/*.csv for both datasets
python make_figure.py         # Figure 3 (results.pdf) and the numbers of Table 1
```

`results/` contains the per-gene scores (five random splits) reported in Table 1 and Figure 3 of the paper.
