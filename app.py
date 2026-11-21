"""Streamlit dashboard: do simple baselines predict unseen CRISPR perturbations?

Run:  streamlit run app.py
Data: put adamson.zip, norman.zip and gene2go_all.pkl (Harvard Dataverse, GEARS) into ./data
"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from baselines.core import (METRICS, DataLoader, ForestPredictor, MeanPredictor, PCAPredictor,
                            Scorer, eligible_perturbations, load_gene2go, run_benchmark, summarise)

DATA_DIR = os.environ.get("BASELINES_DATA", "data")


class Dashboard:
    def __init__(self):
        st.set_page_config(page_title="Simple baselines for perturbation prediction", layout="wide")

    @staticmethod
    @st.cache_data(show_spinner="Building pseudobulk profiles (first run only)...")
    def _pseudobulk(name):
        return DataLoader(name, DATA_DIR).pseudobulk()

    @staticmethod
    @st.cache_data(show_spinner=False)
    def _gene2go():
        return load_gene2go(os.path.join(DATA_DIR, "gene2go_all.pkl"))

    def run(self):
        st.title("Can simple baselines predict unseen CRISPR perturbations?")
        st.caption("Training mean, linear PCA model [Ahlmann-Eltze et al. 2025] and GO random forest "
                   "[Csendes et al. 2025] on GEARS-processed Perturb-seq data. Test genes are never used for fitting.")
        name = st.sidebar.selectbox("Dataset", ["adamson", "norman"])
        n_seeds = st.sidebar.slider("Random splits (seeds)", 1, 10, 5)
        n_trees = st.sidebar.slider("Trees in random forest", 20, 500, 100, step=20)
        pb = self._pseudobulk(name)
        go = self._gene2go()
        perts = eligible_perturbations(pb, go)
        st.sidebar.write(f"{len(perts)} single-gene perturbations, {pb.expr.shape[1]} genes, "
                         f"{int(pb.n_cells.sum())} cells")

        if st.sidebar.button("Run benchmark", type="primary"):
            bar = st.progress(0.0)
            st.session_state["results"] = run_benchmark(pb, go, seeds=range(1, n_seeds + 1),
                                                        n_estimators=n_trees, progress=bar.progress)
            st.session_state["dataset"] = name
        if st.session_state.get("dataset") != name or "results" not in st.session_state:
            st.info("Choose a dataset and press **Run benchmark**.")
            return
        res = st.session_state["results"]

        summary = summarise(res)
        fmt = {m: ("{:.4f}" if m.startswith("MSE") else "{:.3f}").format for m in METRICS}
        table = pd.DataFrame({m: summary[(m, "mean")].map(fmt[m]) + " ± " +
                              summary[(m, "std")].fillna(0).map(fmt[m]) for m in METRICS})
        st.subheader("Scores (mean ± s.d. over random splits)")
        st.dataframe(table)
        st.caption("Lower is better for MSE; higher is better for Pearson delta and top-20 overlap.")
        self.showBarChart(res)
        self.showScatter(pb, go, perts)

    def showBarChart(self, res):
        per_seed = res.groupby(["model", "seed"])[METRICS].mean().reset_index()
        long = per_seed.melt(id_vars=["model", "seed"], var_name="metric", value_name="value")
        fig = px.box(long, x="model", y="value", color="model", facet_col="metric", points="all")
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_layout(showlegend=False, height=380)
        st.subheader("Scores per split")
        st.plotly_chart(fig)

    @staticmethod
    @st.cache_data(show_spinner="Fitting the three predictors for this split...")
    def _scatter_frame(name, seed, gene):
        pb = DataLoader(name, DATA_DIR).pseudobulk()
        go = load_gene2go(os.path.join(DATA_DIR, "gene2go_all.pkl"))
        perts = eligible_perturbations(pb, go)
        split = DataLoader.splitByGene(perts, seed=seed)
        train = pb.expr.loc[split.train]
        forest = ForestPredictor(go, seed=seed)
        forest.build_features(list(pb.expr.columns) + perts)
        scorer = Scorer(pb.ctrl, pb.de20)
        measured = pb.expr.loc[gene] - pb.ctrl
        frames = []
        for model in (MeanPredictor(), PCAPredictor(), forest):
            pred = model.fit(train, pb.ctrl).predict([gene]).loc[gene]
            frames.append(pd.DataFrame({"measured change": measured.values,
                                        "predicted change": (pred - pb.ctrl)[measured.index].values,
                                        "top-20 DE": measured.index.isin(pb.de20[gene]),
                                        "gene": measured.index,
                                        "model": f"{model.name} (Pearson delta = "
                                                 f"{scorer.correlation(pred, pb.expr.loc[gene]):.2f})"}))
        return pd.concat(frames)

    def showScatter(self, pb, go, perts):
        st.subheader("Predicted vs. measured change for one unseen gene")
        seed = int(st.number_input("Split seed", 1, 10, 1))
        split = DataLoader.splitByGene(perts, seed=seed)
        gene = st.selectbox("Unseen perturbed gene", split.test)
        df = self._scatter_frame(st.session_state["dataset"], seed, gene)
        fig = px.scatter(df, x="measured change", y="predicted change", color="top-20 DE",
                         facet_col="model", hover_name="gene", opacity=0.6, height=420)
        st.plotly_chart(fig)


if __name__ == "__main__":
    Dashboard().run()
