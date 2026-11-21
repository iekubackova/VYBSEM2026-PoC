"""Core of the proof-of-concept: data loading, split, three predictors, scoring.

The classes follow the UML diagram in the paper (Figure 1):
DataLoader -> Split, Predictor (MeanPredictor, PCAPredictor, ForestPredictor),
Scorer.  The Streamlit Dashboard lives in app.py.
"""
from __future__ import annotations

import os
import pickle
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

CTRL = "ctrl"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@dataclass
class Pseudobulk:
    """Mean log-normalised expression per condition (rows) and gene (columns)."""
    expr: pd.DataFrame                # index: perturbed gene symbol or 'ctrl'
    n_cells: pd.Series                # cells per condition
    de20: Dict[str, List[str]]        # perturbed gene -> 20 DE gene symbols (GEARS)

    @property
    def ctrl(self) -> pd.Series:
        return self.expr.loc[CTRL]

    @property
    def perturbations(self) -> List[str]:
        return [p for p in self.expr.index if p != CTRL]


@dataclass
class Split:
    train: List[str]
    test: List[str]
    seed: int


def pseudobulk_from_arrays(X, conditions, gene_symbols, gene_ids,
                           condition_names=None, de20_by_condition_name=None,
                           chunk_rows: int = 20000) -> Pseudobulk:
    """Average cells per condition, keeping only control and single-gene perturbations.

    X               cells x genes matrix (scipy sparse CSR or dense) of log-normalised values
    conditions      per-cell GEARS condition label: 'ctrl', 'GENE+ctrl' or 'A+B'
    gene_symbols    readout gene symbols (columns)
    gene_ids        readout gene identifiers used in GEARS' DE lists (adata.var_names)
    condition_names per-cell GEARS 'condition_name' (key of the DE dictionary)
    """
    conditions = np.asarray(conditions).astype(str)
    # 'A+ctrl' and 'ctrl+A' (both occur in Norman) are the same perturbation -> one profile
    canon = {c: (CTRL if c == CTRL else _single_gene(c)) for c in pd.unique(conditions)}
    names = list(dict.fromkeys(v for v in canon.values() if v is not None))
    name_to_row = {n: i for i, n in enumerate(names)}
    rows = np.array([name_to_row.get(canon[c], -1) if canon[c] is not None else -1 for c in conditions])

    n_lab, n_genes = len(names), X.shape[1]
    sums = np.zeros((n_lab, n_genes), dtype=np.float64)
    counts = np.bincount(rows[rows >= 0], minlength=n_lab)
    for start in range(0, X.shape[0], chunk_rows):
        stop = min(start + chunk_rows, X.shape[0])
        r = rows[start:stop]
        ok = r >= 0
        block = X[start:stop]
        ind = sp.csr_matrix((np.ones(ok.sum()), (r[ok], np.nonzero(ok)[0])),
                            shape=(n_lab, stop - start))
        part = ind @ block
        sums += part.toarray() if sp.issparse(part) else np.asarray(part)
    means = sums / counts[:, None]

    expr = pd.DataFrame(means, index=names, columns=list(gene_symbols))
    n_cells = pd.Series(counts, index=names)

    de20: Dict[str, List[str]] = {}
    if condition_names is not None and de20_by_condition_name is not None:
        id2sym = dict(zip(gene_ids, gene_symbols))
        cond_to_name = dict(zip(conditions, np.asarray(condition_names).astype(str)))
        for lab, gene in canon.items():
            if gene is None or gene == CTRL or gene in de20:
                continue
            key = cond_to_name.get(lab)
            if key in de20_by_condition_name:
                de20[gene] = [id2sym[g] for g in de20_by_condition_name[key] if g in id2sym]
    return Pseudobulk(expr=expr, n_cells=n_cells, de20=de20)


def _single_gene(label: str) -> Optional[str]:
    parts = [p for p in label.split("+") if p != CTRL]
    return parts[0] if len(parts) == 1 and label != CTRL else None


class DataLoader:
    """Loads a GEARS-processed dataset (adamson.zip / norman.zip from Harvard Dataverse)."""

    def __init__(self, datasetName: str, data_dir: str = "data"):
        self.datasetName = datasetName
        self.data_dir = data_dir

    def _h5ad_path(self) -> str:
        folder = os.path.join(self.data_dir, self.datasetName)
        path = os.path.join(folder, "perturb_processed.h5ad")
        if not os.path.exists(path):
            archive = os.path.join(self.data_dir, self.datasetName + ".zip")
            if not os.path.exists(archive):
                raise FileNotFoundError(f"Put {self.datasetName}.zip into '{self.data_dir}/' first.")
            with zipfile.ZipFile(archive) as z:
                z.extractall(self.data_dir)
        return path

    def load(self):
        import anndata  # imported here so the rest of the module works without it
        return anndata.read_h5ad(self._h5ad_path())

    def pseudobulk(self) -> Pseudobulk:
        cache = os.path.join(self.data_dir, f"{self.datasetName}_pseudobulk.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                return pickle.load(f)
        adata = self.load()
        X = adata.X if sp.issparse(adata.X) else np.asarray(adata.X)
        if sp.issparse(X):
            X = sp.csr_matrix(X)
        pb = pseudobulk_from_arrays(
            X=X,
            conditions=adata.obs["condition"].astype(str).values,
            gene_symbols=adata.var["gene_name"].astype(str).values,
            gene_ids=adata.var_names.astype(str).values,
            condition_names=adata.obs["condition_name"].astype(str).values,
            de20_by_condition_name={k: list(map(str, v)) for k, v in
                                    adata.uns.get("top_non_dropout_de_20", {}).items()},
        )
        with open(cache, "wb") as f:
            pickle.dump(pb, f)
        return pb

    @staticmethod
    def splitByGene(perturbations: List[str], seed: int = 1, train_fraction: float = 0.75) -> Split:
        """GEARS-style 'simulation' split: a random 75% of perturbed genes are used for
        fitting; the remaining genes are never seen during fitting and form the test set."""
        genes = np.unique(perturbations)
        rng = np.random.RandomState(seed)
        train = rng.choice(genes, int(len(genes) * train_fraction), replace=False)
        test = np.setdiff1d(genes, train)
        return Split(train=sorted(train.tolist()), test=sorted(test.tolist()), seed=seed)


# ----------------------------------------------------------------------------
# Predictors
# ----------------------------------------------------------------------------
class Predictor:
    """Interface: fit on training pseudobulks, predict expression for unseen genes."""
    name = "Predictor"

    def fit(self, train: pd.DataFrame, ctrl: pd.Series) -> "Predictor":
        raise NotImplementedError

    def predict(self, genes: List[str]) -> pd.DataFrame:
        raise NotImplementedError


class MeanPredictor(Predictor):
    """Predicts the average training perturbation profile for every unseen gene."""
    name = "Training mean"

    def fit(self, train, ctrl):
        self.mean_ = train.mean(axis=0)
        return self

    def predict(self, genes):
        return pd.DataFrame([self.mean_.values] * len(genes), index=genes, columns=self.mean_.index)


class PCAPredictor(Predictor):
    """Linear model of Ahlmann-Eltze et al. [1]:  Y ~ G W P^T + b.

    G: top-K principal components of the training pseudobulk matrix (genes x perturbations),
    P: the rows of G that belong to the perturbed genes, W: two-sided ridge solution,
    b: row means of Y.  Y is the change versus control.  K = 10 and lambda = 0.1 as in [1].
    """
    name = "Linear (PCA)"

    def __init__(self, k: int = 10, ridge: float = 0.1):
        self.k, self.ridge = k, ridge

    def fit(self, train, ctrl):
        self.ctrl_ = ctrl
        genes = list(train.columns)
        expr = pd.concat([train, ctrl.to_frame(CTRL).T])            # control is part of training, P = 0
        M = expr.T.values                                            # genes x conditions
        Mc = M - M.mean(axis=0, keepdims=True)                       # prcomp centres the columns
        U, S, _ = np.linalg.svd(Mc, full_matrices=False)
        G = U[:, :self.k] * S[:self.k]                               # gene embedding (PC scores)
        self.G_ = pd.DataFrame(G, index=genes)
        P = np.stack([self._pert_vec(p) for p in expr.index], axis=1)  # k x conditions
        Y = (expr - ctrl).T.values                                   # change, genes x conditions
        self.b_ = Y.mean(axis=1, keepdims=True)
        Yc = Y - self.b_
        lam = self.ridge
        A = np.linalg.solve(G.T @ G + lam * np.eye(self.k), G.T @ Yc)        # k x conditions
        self.W_ = A @ P.T @ np.linalg.inv(P @ P.T + lam * np.eye(self.k))    # k x k
        return self

    def _pert_vec(self, gene):
        if gene == CTRL or gene not in self.G_.index:
            return np.zeros(self.k)
        return self.G_.loc[gene].values

    def can_predict(self, gene) -> bool:
        return gene in self.G_.index

    def predict(self, genes):
        P = np.stack([self._pert_vec(g) for g in genes], axis=1)
        Y = self.G_.values @ self.W_ @ P + self.b_
        out = Y.T + self.ctrl_.values[None, :]
        return pd.DataFrame(out, index=genes, columns=self.G_.index)


class ForestPredictor(Predictor):
    """Random forest on Gene Ontology features, following Csendes et al. [2].

    A binary gene x GO-term matrix is reduced to 256 components; the perturbed gene's
    component vector is the input and its expression change is the (multi-output) target.
    """
    name = "Random forest (GO)"

    def __init__(self, gene2go: Dict[str, set], n_components: int = 256,
                 n_estimators: int = 100, seed: int = 0, n_jobs: int = -1):
        self.gene2go, self.n_components = gene2go, n_components
        self.n_estimators, self.seed, self.n_jobs = n_estimators, seed, n_jobs
        self.features_ = None

    def build_features(self, genes: List[str]) -> pd.DataFrame:
        from sklearn.decomposition import TruncatedSVD
        genes = [g for g in dict.fromkeys(genes) if g in self.gene2go]
        terms = sorted({t for g in genes for t in self.gene2go[g]})
        col = {t: j for j, t in enumerate(terms)}
        r, c = zip(*[(i, col[t]) for i, g in enumerate(genes) for t in self.gene2go[g]])
        B = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(genes), len(terms)))
        n = min(self.n_components, len(genes) - 1)
        Z = TruncatedSVD(n_components=n, random_state=0).fit_transform(B)  # sparse, no densifying
        self.features_ = pd.DataFrame(Z, index=genes)
        return self.features_

    def can_predict(self, gene) -> bool:
        return self.features_ is not None and gene in self.features_.index

    def fit(self, train, ctrl):
        from sklearn.ensemble import RandomForestRegressor
        self.ctrl_ = ctrl
        rows = [g for g in train.index if self.can_predict(g)]
        X = self.features_.loc[rows].values
        Y = (train.loc[rows] - ctrl).values
        self.model_ = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.seed,
                                            n_jobs=self.n_jobs).fit(X, Y)
        self.columns_ = train.columns
        return self

    def predict(self, genes):
        Y = self.model_.predict(self.features_.loc[genes].values)
        return pd.DataFrame(Y + self.ctrl_.values[None, :], index=genes, columns=self.columns_)


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
class Scorer:
    """Four scores per unseen perturbation (all computed on pseudobulk profiles)."""

    def __init__(self, ctrl: pd.Series, de20: Dict[str, List[str]]):
        self.ctrl, self.de20 = ctrl, de20

    def errorAll(self, pred: pd.Series, true: pd.Series) -> float:
        """Mean squared error over all measured genes."""
        return float(np.mean((pred.values - true.values) ** 2))

    def errorTop(self, pred, true, gene) -> float:
        """Mean squared error over the 20 most differentially expressed genes."""
        de = self.de20[gene]
        return float(np.mean((pred[de].values - true[de].values) ** 2))

    def correlation(self, pred, true) -> float:
        """Pearson correlation of predicted and measured change versus control (Pearson delta)."""
        dp, dt = pred - self.ctrl, true - self.ctrl
        return float(np.corrcoef(dp.values, dt.values)[0, 1])

    def rankAgree(self, pred, true, gene) -> float:
        """Share of the 20 DE genes that are also among the 20 genes with the
        largest predicted absolute change."""
        de = set(self.de20[gene])
        top_pred = (pred - self.ctrl).abs().nlargest(20).index
        return len(de & set(top_pred)) / len(de)

    def score(self, pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for g in pred.index:
            p, t = pred.loc[g], truth.loc[g, pred.columns]
            rows.append({"gene": g,
                         "MSE (all genes)": self.errorAll(p, t),
                         "MSE (top-20 DE)": self.errorTop(p, t, g),
                         "Pearson delta": self.correlation(p, t),
                         "Top-20 overlap": self.rankAgree(p, t, g)})
        return pd.DataFrame(rows)


METRICS = ["MSE (all genes)", "MSE (top-20 DE)", "Pearson delta", "Top-20 overlap"]


# ----------------------------------------------------------------------------
# One full benchmark run
# ----------------------------------------------------------------------------
def load_gene2go(path: str) -> Dict[str, set]:
    """Load GEARS' gene -> GO-term dictionary; accepts gene2go_all.pkl or gene2go.pkl."""
    if not os.path.exists(path):
        alt = os.path.join(os.path.dirname(path), "gene2go.pkl")
        path = alt if os.path.exists(alt) else path
    with open(path, "rb") as f:
        return pickle.load(f)


def eligible_perturbations(pb: Pseudobulk, gene2go: Dict[str, set], min_cells: int = 1) -> List[str]:
    """Genes that every predictor can handle: target gene measured (linear model),
    has GO annotation (forest), has a GEARS top-20 DE list (scores)."""
    cols = set(pb.expr.columns)
    return [g for g in pb.perturbations
            if g in cols and g in gene2go and len(pb.de20.get(g, [])) == 20
            and pb.n_cells[g] >= min_cells]


def run_benchmark(pb: Pseudobulk, gene2go: Dict[str, set], seeds=(1, 2, 3, 4, 5),
                  n_estimators: int = 100, progress=None) -> pd.DataFrame:
    perts = eligible_perturbations(pb, gene2go)
    forest_template = ForestPredictor(gene2go, n_estimators=n_estimators)
    forest_template.build_features(list(pb.expr.columns) + perts)
    scorer = Scorer(pb.ctrl, pb.de20)
    out = []
    for i, seed in enumerate(seeds):
        split = DataLoader.splitByGene(perts, seed=seed)
        train = pb.expr.loc[split.train]
        truth = pb.expr.loc[split.test]
        forest = ForestPredictor(gene2go, n_estimators=n_estimators, seed=seed)
        forest.features_ = forest_template.features_
        for model in (MeanPredictor(), PCAPredictor(), forest):
            model.fit(train, pb.ctrl)
            s = scorer.score(model.predict(split.test), truth)
            s.insert(0, "model", model.name)
            s.insert(0, "seed", seed)
            out.append(s)
        if progress:
            progress((i + 1) / len(seeds))
    return pd.concat(out, ignore_index=True)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Average over test genes within a seed, then mean and s.d. over seeds."""
    per_seed = results.groupby(["model", "seed"])[METRICS].mean()
    agg = per_seed.groupby("model").agg(["mean", "std"])
    return agg
