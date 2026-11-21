"""Headless run of the benchmark for both datasets; writes results/ CSV files."""
import os
import sys

from baselines.core import DataLoader, eligible_perturbations, load_gene2go, run_benchmark, summarise

DATA_DIR = os.environ.get("BASELINES_DATA", "data")


def main(datasets=("adamson", "norman"), seeds=(1, 2, 3, 4, 5)):
    os.makedirs("results", exist_ok=True)
    go = load_gene2go(os.path.join(DATA_DIR, "gene2go_all.pkl"))
    for name in datasets:
        pb = DataLoader(name, DATA_DIR).pseudobulk()
        print(f"{name}: {len(eligible_perturbations(pb, go))} eligible perturbations, "
              f"{pb.expr.shape[1]} genes")
        res = run_benchmark(pb, go, seeds=seeds)
        res.to_csv(f"results/{name}_per_gene.csv", index=False)
        summ = summarise(res)
        summ.to_csv(f"results/{name}_summary.csv")
        print(summ.round(3).to_string())


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("adamson", "norman"))
