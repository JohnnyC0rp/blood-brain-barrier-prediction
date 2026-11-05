"""Run the full BBBP experiment; all settings are declared in molecular_utils."""

import hashlib
import json
import multiprocessing as mp
import platform
import time
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Draw, rdDepictor
from sklearn.metrics import precision_recall_curve, roc_curve
from torch_geometric.loader import DataLoader
from torch_geometric.utils.smiles import x_map, e_map
from molecular_utils import (
    CONFIG,
    SEEDS,
    SPLIT_SEED,
    MolecularGNN,
    check_batches,
    classification_metrics,
    fingerprints,
    fit_baseline,
    fit_gnn,
    load_clean_dataset,
    make_splits,
    predict_gnn,
    save_json,
    seed_everything,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TABLES = OUT / "tables"
MODELS = OUT / "models"
METRICS = ["roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity", "f1"]
COLORS = ["#173D54", "#B26129", "#307A67", "#9464A5"]


def file_hash(path):
    """Calculate a checksum to detect changed code or result files."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest():
    return {
        name: file_hash(ROOT / name)
        for name in ["molecular_utils.py", "run_experiments.py"]
    }


def save_figure(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def inspect_figures(graphs, molecules, frame):
    """Draw examples from both classes and inspect one molecular graph."""
    example_ids = []
    for label in [0, 1]:
        class_records = frame[frame.label == label].assign(
            length=lambda records: records.smiles.str.len()
        )
        example_ids.extend(
            class_records.sort_values(["length", "record_id"])
            .head(3)
            .record_id.tolist()
        )
    frame.iloc[example_ids].to_csv(TABLES / "examples.csv", index=False)
    image = Draw.MolsToGridImage(
        [molecules[index] for index in example_ids],
        molsPerRow=3,
        useSVG=False,
        returnPNG=False,
        subImgSize=(330, 220),
        legends=[
            f"ID {index}; label {frame.label.iloc[index]}" for index in example_ids
        ],
    )
    image.save(FIG / "molecule_examples.png")
    graph_id = next(
        (
            index
            for (index, molecule) in enumerate(molecules)
            if 8 <= molecule.GetNumAtoms() <= 14
            and molecule.GetNumBonds() > 0
            and (molecule.GetRingInfo().NumRings() > 0)
        )
    )
    molecule = Chem.Mol(molecules[graph_id])
    rdDepictor.Compute2DCoords(molecule)
    for atom in molecule.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    (fig, axes) = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].imshow(Draw.MolToImage(molecule, size=(650, 420)))
    axes[0].axis("off")
    axes[0].set_title("Chemical structure (atom IDs)")
    network = nx.Graph()
    network.add_nodes_from(range(molecule.GetNumAtoms()))
    network.add_edges_from(
        ((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in molecule.GetBonds())
    )
    conformer = molecule.GetConformer()
    positions = {
        index: (conformer.GetAtomPosition(index).x, conformer.GetAtomPosition(index).y)
        for index in network
    }
    nx.draw_networkx(
        network,
        pos=positions,
        ax=axes[1],
        node_color="#DCECF1",
        node_size=460,
        labels={
            index: f"{index}:{molecule.GetAtomWithIdx(index).GetSymbol()}"
            for index in network
        },
        font_size=8,
    )
    axes[1].axis("off")
    axes[1].set_title("Graph: matching atom IDs; bonds are edges")
    fig.suptitle(f"Record {graph_id}; label={frame.label.iloc[graph_id]}")
    save_figure(fig, "graph_views")
    (fig, axes) = plt.subplots(1, 2, figsize=(9, 2.8))
    for axis, column in zip(axes, ["atoms", "bonds"]):
        axis.hist(frame[column], bins=35, color=COLORS[0], edgecolor="white")
        axis.set(xlabel=f"{column.capitalize()} per molecule", ylabel="Molecules")
    fig.tight_layout()
    save_figure(fig, "size_distributions")
    graph = graphs[graph_id]
    return dict(
        record_id=graph_id,
        smiles=graph.smiles,
        x_shape=list(graph.x.shape),
        edge_index_shape=list(graph.edge_index.shape),
        edge_attr_shape=list(graph.edge_attr.shape),
        y=graph.y.tolist(),
        x_first_atom=graph.x[0].tolist(),
        edge_attr_first_bond=graph.edge_attr[0].tolist(),
    )


def time_device(batch, name):
    """Warm up, then time the same workload on one device."""
    seed_everything(SPLIT_SEED)
    torch.set_num_threads(4)
    model = MolecularGNN().to(name)
    device_batch = batch.clone().to(name)
    model.train()
    for _ in range(2):
        model.zero_grad()
        model(device_batch).sum().backward()
    if name == "mps":
        torch.mps.synchronize()
    start = time.monotonic()
    for _ in range(5):
        model.zero_grad()
        model(device_batch).sum().backward()
    if name == "mps":
        torch.mps.synchronize()
    return (time.monotonic() - start) / 5


def mps_worker(batch, result):
    """Run MPS separately so a stalled benchmark can be stopped."""
    try:
        result.put(time_device(batch, "mps"))
    except (RuntimeError, NotImplementedError) as error:
        result.put(str(error).split("\n")[0])


def select_device(graphs):
    """Benchmark computation only; no test labels or model performance consulted."""
    batch = next(iter(DataLoader(graphs[:64], batch_size=64)))
    timings = {"cpu": time_device(batch, "cpu")}
    if torch.backends.mps.is_available():
        process_context = mp.get_context("spawn")
        result_queue = process_context.Queue()
        process = process_context.Process(target=mps_worker, args=(batch, result_queue))
        process.start()
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join()
            timings["mps"] = "Exceeded 20-second setup/benchmark limit; CPU selected."
        else:
            timings["mps"] = (
                result_queue.get(timeout=2)
                if process.exitcode == 0
                else f"Benchmark worker exit {process.exitcode}"
            )
    successful_timings = {
        device_name: duration
        for (device_name, duration) in timings.items()
        if isinstance(duration, float)
    }
    selected_device = min(successful_timings, key=successful_timings.get)
    print(
        f"Device benchmark seconds/forward+backward: {timings}; selected {selected_device}",
        flush=True,
    )
    return (selected_device, timings)


def training_figure(histories):
    """Show losses and validation ROC-AUC for all six GNN runs."""
    (fig, axes) = plt.subplots(2, 3, figsize=(12, 6))
    for row, strategy in enumerate(["random", "scaffold"]):
        for color, seed in zip(COLORS, SEEDS):
            history = histories[f"{strategy}_{seed}"]
            for column_index, metric in enumerate(
                ["train_loss", "validation_loss", "validation_roc_auc"]
            ):
                axes[row, column_index].plot(
                    history.epoch,
                    history[metric],
                    color=color,
                    label=f"Seed {seed}",
                    lw=1.4,
                )
                axes[row, column_index].set(
                    xlabel="Epoch",
                    ylabel=metric.replace("_", " "),
                    title=strategy.capitalize(),
                )
                axes[row, column_index].grid(alpha=0.15)
        axes[row, 0].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "training_curves")


def evaluation_figures(runs, predictions, labels, splits):
    """Plot final curves and confusion matrices from test probabilities."""
    (fig, axes) = plt.subplots(2, 2, figsize=(10, 8))
    (confusion_figure, confusion_axes) = plt.subplots(2, 4, figsize=(12, 5.5))
    for row, strategy in enumerate(["random", "scaffold"]):
        test_labels = labels[splits[strategy]["test"]]
        strategy_runs = [run for run in runs if run["split"] == strategy]
        for column_index, (run, color) in enumerate(zip(strategy_runs, COLORS)):
            run_id = run["run_id"]
            probabilities = predictions[run_id]
            title = (
                "Morgan LR"
                if run["model"] == "Morgan LR"
                else f"GINE seed {run['seed']}"
            )
            (false_positive_rate, sensitivity, _) = roc_curve(
                test_labels, probabilities
            )
            (precision, recall, _) = precision_recall_curve(test_labels, probabilities)
            axes[row, 0].plot(
                false_positive_rate,
                sensitivity,
                color=color,
                label=f"{title} ({run['roc_auc']:.3f})",
            )
            axes[row, 1].plot(
                recall, precision, color=color, label=f"{title} ({run['pr_auc']:.3f})"
            )
            confusion = np.array([[run["tn"], run["fp"]], [run["fn"], run["tp"]]])
            axis = confusion_axes[row, column_index]
            axis.imshow(confusion, cmap="Blues", vmin=0, vmax=len(test_labels))
            for (true_label, predicted_label), count in np.ndenumerate(confusion):
                axis.text(
                    predicted_label,
                    true_label,
                    str(count),
                    ha="center",
                    va="center",
                    color="white" if count > len(test_labels) / 2 else "black",
                    fontsize=10,
                )
            axis.set(
                xticks=[0, 1],
                yticks=[0, 1],
                xlabel="Predicted label",
                ylabel="True label",
                title=f"{strategy}: {title}\nt={run['threshold']:.3f}",
            )
        axes[row, 0].plot([0, 1], [0, 1], "k--", lw=0.7)
        axes[row, 1].axhline(test_labels.mean(), color="gray", ls="--", lw=0.7)
        axes[row, 0].set(
            title=f"{strategy.capitalize()} ROC",
            xlabel="False-positive rate",
            ylabel="Sensitivity",
        )
        axes[row, 1].set(
            title=f"{strategy.capitalize()} precision-recall",
            xlabel="Recall",
            ylabel="Precision",
        )
        for axis in axes[row]:
            axis.legend(fontsize=7)
            axis.grid(alpha=0.15)
    fig.tight_layout()
    save_figure(fig, "test_curves")
    confusion_figure.tight_layout()
    save_figure(confusion_figure, "confusion_matrices")


def error_analysis(
    frame,
    molecules,
    graphs,
    bitvectors,
    labels,
    splits,
    runs,
    predictions,
    models,
    device,
):
    run = next((r for r in runs if r["run_id"] == "scaffold_gnn_17"))
    test_indices = splits["scaffold"]["test"]
    train_indices = splits["scaffold"]["train"]
    probabilities = predictions[run["run_id"]]
    predicted_labels = probabilities >= run["threshold"]
    selected_errors = []
    for error_type, true_label in [("FP", 0), ("FN", 1)]:
        error_positions = [
            index
            for index in range(len(test_indices))
            if labels[test_indices[index]] == true_label
            and predicted_labels[index] != true_label
        ]
        error_positions.sort(key=lambda test_position: int(test_indices[test_position]))
        selected_errors.extend(
            ((error_type, test_position) for test_position in error_positions[:2])
        )
    training_probabilities = predict_gnn(
        models["scaffold_gnn_17"], graphs, train_indices, device
    )
    details = []
    (figure, axes) = plt.subplots(
        max(1, len(selected_errors)),
        2,
        figsize=(10, 2.5 * max(1, len(selected_errors))),
    )
    axes = np.asarray(axes).reshape(-1, 2)
    for row, (error_type, test_position) in enumerate(selected_errors):
        record_id = int(test_indices[test_position])
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                bitvectors[record_id], [bitvectors[index] for index in train_indices]
            )
        )
        nearest_position = int(np.argmax(similarities))
        neighbor_id = int(train_indices[nearest_position])
        details.append(
            dict(
                error=error_type,
                record_id=record_id,
                smiles=frame.smiles.iloc[record_id],
                canonical_smiles=frame.canonical_smiles.iloc[record_id],
                label=int(labels[record_id]),
                probability=float(probabilities[test_position]),
                threshold=run["threshold"],
                neighbor_id=neighbor_id,
                neighbor_smiles=frame.smiles.iloc[neighbor_id],
                neighbor_label=int(labels[neighbor_id]),
                neighbor_probability=float(training_probabilities[nearest_position]),
                tanimoto=float(similarities[nearest_position]),
                seed=17,
            )
        )
        for column_index, (molecule_id, probability, true_label) in enumerate(
            [
                (record_id, probabilities[test_position], labels[record_id]),
                (
                    neighbor_id,
                    training_probabilities[nearest_position],
                    labels[neighbor_id],
                ),
            ]
        ):
            axes[row, column_index].imshow(
                Draw.MolToImage(molecules[molecule_id], size=(620, 270))
            )
            axes[row, column_index].axis("off")
            prefix = (
                f"{error_type} test"
                if column_index == 0
                else "Nearest training molecule"
            )
            suffix = (
                ""
                if column_index == 0
                else f"; Tanimoto={similarities[nearest_position]:.3f}"
            )
            axes[row, column_index].set_title(
                f"{prefix} ID {molecule_id}; y={true_label}; p={probability:.3f}{suffix}",
                fontsize=9,
            )
    figure.tight_layout()
    save_figure(figure, "error_pairs")
    pd.DataFrame(details).to_csv(TABLES / "error_details.csv", index=False)
    return details


def run_pipeline(reuse_results=True):
    """Reuse only a hash-validated complete local run; otherwise fit from scratch."""
    for folder in [OUT, FIG, TABLES, MODELS]:
        folder.mkdir(exist_ok=True, parents=True)
    manifest_path = OUT / "manifest.json"
    if reuse_results and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        cache_matches = (
            manifest["sources"] == source_manifest() and manifest["config"] == CONFIG
        )
        cache_matches = cache_matches and all(
            (
                (ROOT / name).is_file() and file_hash(ROOT / name) == digest
                for (name, digest) in manifest["artifacts"].items()
            )
        )
        if cache_matches:
            print(
                "Reusing complete hash-validated experiment, without re-evaluating test sets.",
                flush=True,
            )
            return json.loads((OUT / "summary.json").read_text())
        print(
            "Cached artifacts incomplete or changed: starting a fresh experiment.",
            flush=True,
        )
    start = time.monotonic()
    milestones = {}
    seed_everything(SPLIT_SEED)
    torch.set_num_threads(4)
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    (graphs, molecules, frame, audit, duplicates) = load_clean_dataset(
        ROOT / "data/MoleculeNet"
    )
    frame.to_csv(TABLES / "molecules.csv", index=False)
    pd.DataFrame(duplicates).to_csv(TABLES / "canonical_duplicates.csv", index=False)
    print(f"Dataset audit: {audit}", flush=True)
    graph_inspection = inspect_figures(graphs, molecules, frame)
    batch_inspection = check_batches(graphs)
    (splits, split_summary) = make_splits(frame)
    split_summary.to_csv(TABLES / "split_summary.csv", index=False)
    split_manifest = frame[
        ["record_id", "canonical_smiles", "scaffold", "label"]
    ].copy()
    for name, subset in splits.items():
        for partition, indices in subset.items():
            split_manifest.loc[indices, f"{name}_subset"] = partition
    split_manifest.to_csv(TABLES / "split_manifest.csv", index=False)
    print(split_summary.to_string(index=False), flush=True)
    (fingerprint_matrix, bitvectors) = fingerprints(molecules)
    labels = frame.label.to_numpy()
    milestones["data_audit_and_splitting_seconds"] = time.monotonic() - start
    common_training = sorted(
        set(splits["random"]["train"]) & set(splits["scaffold"]["train"])
    )
    (device, device_benchmark) = select_device(
        [graphs[i] for i in common_training[:64]]
    )
    # Select all models before starting final test inference.
    models = {}
    selection = {}
    histories = {}
    training_start = time.monotonic()
    for strategy, subset in splits.items():
        print(
            f"Fitting {strategy} Morgan baseline (training/validation only)", flush=True
        )
        (model, metadata) = fit_baseline(fingerprint_matrix, labels, subset)
        run_id = f"{strategy}_lr"
        models[run_id] = model
        selection[run_id] = metadata
        for seed in SEEDS:
            run_id = f"{strategy}_gnn_{seed}"
            print(f"Fitting {run_id} (training/validation only)", flush=True)
            (model, history, metadata) = fit_gnn(
                graphs, labels, subset, seed, device, MODELS / f"{run_id}.pt"
            )
            models[run_id] = model
            selection[run_id] = metadata
            histories[f"{strategy}_{seed}"] = history
            history.to_csv(TABLES / f"{run_id}_history.csv", index=False)
    milestones["model_selection_seconds"] = time.monotonic() - training_start
    # Freeze checkpoint choices and thresholds before evaluating the test sets.
    lock = dict(
        config=CONFIG,
        selection=selection,
        sources=source_manifest(),
        split_manifest_sha256=file_hash(TABLES / "split_manifest.csv"),
        test_evaluation_started=False,
        error_analysis_model="scaffold_gnn_17",
    )
    save_json(OUT / "evaluation_lock.json", lock)
    print(
        "All eight final models and thresholds frozen. Beginning final test evaluation.",
        flush=True,
    )
    evaluation_start = time.monotonic()
    runs = []
    predictions = {}
    for strategy, subset in splits.items():
        for suffix in ["lr"] + [f"gnn_{seed}" for seed in SEEDS]:
            run_id = f"{strategy}_{suffix}"
            model = models[run_id]
            test_indices = subset["test"]
            probabilities = (
                model.predict_proba(fingerprint_matrix[test_indices])[:, 1]
                if suffix == "lr"
                else predict_gnn(model, graphs, test_indices, device)
            )
            predictions[run_id] = probabilities
            metadata = selection[run_id]
            metrics = classification_metrics(
                labels[test_indices], probabilities, metadata["threshold"]
            )
            row = dict(
                run_id=run_id,
                model="Morgan LR" if suffix == "lr" else "GINE",
                split=strategy,
                **metadata,
                **metrics,
            )
            runs.append(row)
            test_results = frame.iloc[test_indices][
                ["record_id", "canonical_smiles", "label"]
            ].copy()
            test_results["probability"] = probabilities
            test_results["threshold"] = metadata["threshold"]
            test_results["prediction"] = (
                probabilities >= metadata["threshold"]
            ).astype(int)
            test_results.to_csv(TABLES / f"{run_id}_predictions.csv", index=False)
            print(f"Final {run_id}: {metrics}", flush=True)
    pd.DataFrame(runs).drop(columns=["candidate_results"], errors="ignore").to_csv(
        TABLES / "comparison.csv", index=False
    )
    # Training-seed variation is measured separately within each fixed split.
    aggregate = []
    for strategy in splits:
        subset = [r for r in runs if r["split"] == strategy and r["model"] == "GINE"]
        row = dict(model="GINE", split=strategy, n_seeds=3)
        for metric in METRICS + ["threshold"]:
            values = [r[metric] for r in subset]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        aggregate.append(row)
    pd.DataFrame(aggregate).to_csv(TABLES / "seed_aggregate.csv", index=False)
    training_figure(histories)
    evaluation_figures(runs, predictions, labels, splits)
    errors = error_analysis(
        frame,
        molecules,
        graphs,
        bitvectors,
        labels,
        splits,
        runs,
        predictions,
        models,
        device,
    )
    milestones["test_evaluation_and_figures_seconds"] = (
        time.monotonic() - evaluation_start
    )
    milestones["total_seconds"] = time.monotonic() - start
    summary = dict(
        audit=audit,
        config=CONFIG,
        device=device,
        device_benchmark=device_benchmark,
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            torch_geometric=torch_geometric.__version__,
            rdkit=rdBase.rdkitVersion,
            platform=platform.platform(),
            cpu_threads=4,
        ),
        feature_schema=dict(
            atom=[
                dict(name=name, cardinality=len(categories))
                for (name, categories) in x_map.items()
            ],
            bond=[
                dict(name=name, cardinality=len(categories))
                for (name, categories) in e_map.items()
            ],
            encoding="Independent learned embeddings per categorical column; sum within atom or bond",
        ),
        graph_inspection=graph_inspection,
        batch_inspection=batch_inspection,
        split_summary=split_summary.to_dict("records"),
        split_method="Scaffold groups shuffled with seed2026, stable sorted largest first, allocated to largest remaining target capacity without labels. Empty scaffold is one group.",
        threshold_method="Maximum validation balanced accuracy on 181 points from0.05 to0.95; ties nearest0.5.",
        pr_auc_definition="Average precision (non-interpolated area under precision-recall curve; sklearn average_precision_score)",
        baseline_selection={
            run_id: value
            for (run_id, value) in selection.items()
            if run_id.endswith("_lr")
        },
        runs=runs,
        aggregate=aggregate,
        error_examples=errors,
        timings=milestones,
        assertions=dict(
            finite_labels=True,
            nonempty_graphs=True,
            batching_complete=True,
            splits_complete=True,
            indices_disjoint=True,
            canonical_smiles_disjoint=True,
            scaffold_groups_disjoint=True,
            test_selection_leakage=False,
        ),
    )
    save_json(OUT / "summary.json", summary)
    lock["test_evaluation_started"] = True
    lock["test_evaluation_completed"] = True
    save_json(OUT / "evaluation_lock.json", lock)
    artifacts = {
        str(path.relative_to(ROOT)): file_hash(path)
        for path in OUT.rglob("*")
        if path.is_file() and path.name != "manifest.json" and (path.suffix != ".html")
    }
    save_json(
        manifest_path,
        dict(sources=source_manifest(), config=CONFIG, artifacts=artifacts),
    )
    print(
        f"Complete: {milestones['total_seconds']:.1f}s; summary={OUT / 'summary.json'}",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run_pipeline()
