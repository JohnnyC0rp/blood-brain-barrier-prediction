"""Dataset cleaning, molecular splits, and reproducible BBBP classifiers."""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils.smiles import e_map, x_map


SPLIT_SEED = 2026
SEEDS = [17, 29, 43]
CONFIG = {
    'hidden': 64,
    'layers': 3,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'batch_size': 64,
    'max_epochs': 80,
    'patience': 15,
    'fingerprint_radius': 2,
    'fingerprint_bits': 2048,
    'baseline_C': [0.01, 0.1, 1.0, 10.0],
    'split_seed': SPLIT_SEED,
    'model_seeds': SEEDS,
    'threshold_metric': 'balanced_accuracy',
    'checkpoint_metric': 'validation_roc_auc',
}


def seed_everything(seed):
    """Fix the random generators before creating a split or model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CPU is deterministic; some MPS reductions can vary at floating precision.
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_json(path, value):
    """Save readable JSON and reject non-finite numbers."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_clean_dataset(root):
    """Audit the download, remove invalid records, and resolve duplicates."""
    dataset = MoleculeNet(root=str(root), name='BBBP')

    # PyG skips empty graphs itself. Inspect its raw CSV to count those records.
    raw = pd.read_csv(dataset.raw_paths[0])
    raw_molecules = [Chem.MolFromSmiles(str(smiles)) for smiles in raw.smiles]
    invalid_raw = []
    for index, molecule in enumerate(raw_molecules):
        if molecule is None or molecule.GetNumAtoms() == 0:
            invalid_raw.append(index)

    canonical_groups = {}
    invalid_loaded = []
    nonfinite_labels = []
    for loader_index, graph in enumerate(dataset):
        molecule = Chem.MolFromSmiles(graph.smiles)
        if molecule is None or molecule.GetNumAtoms() == 0 or graph.num_nodes == 0:
            invalid_loaded.append(loader_index)
            continue
        if graph.y.numel() != 1 or not torch.isfinite(graph.y).all():
            nonfinite_labels.append(loader_index)
            continue

        label = int(graph.y.item())
        assert label in (0, 1)
        canonical_smiles = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
        record = (loader_index, label, graph, molecule)
        canonical_groups.setdefault(canonical_smiles, []).append(record)

    graphs = []
    molecules = []
    records = []
    duplicate_records = []
    conflicting_records_removed = 0
    consistent_records_removed = 0

    for canonical_smiles, group in canonical_groups.items():
        group_labels = {record[1] for record in group}
        has_conflicting_labels = len(group_labels) > 1
        if len(group) > 1:
            original_smiles = {record[2].smiles for record in group}
            duplicate_records.append({
                'canonical_smiles': canonical_smiles,
                'records': len(group),
                'labels': ','.join(str(record[1]) for record in group),
                'original_smiles_variants': len(original_smiles),
                'conflicting': has_conflicting_labels,
            })

        # A conflicting group supplies no reliable single target: remove it all.
        if has_conflicting_labels:
            conflicting_records_removed += len(group)
            continue

        # For consistent duplicates, keep the first original record only.
        consistent_records_removed += len(group) - 1
        loader_index, label, graph, molecule = group[0]
        graph = graph.clone()
        graph.record_id = torch.tensor([len(graphs)], dtype=torch.long)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule, includeChirality=False
        )
        # Acyclic molecules all have the empty scaffold and must stay together.
        graphs.append(graph)
        molecules.append(molecule)
        records.append({
            'record_id': len(records),
            'loader_index': loader_index,
            'smiles': graph.smiles,
            'canonical_smiles': canonical_smiles,
            'label': label,
            'atoms': molecule.GetNumAtoms(),
            'bonds': molecule.GetNumBonds(),
            'scaffold': scaffold,
        })

    frame = pd.DataFrame(records)
    assert len(frame.canonical_smiles.unique()) == len(frame)
    assert all(
        graph.num_nodes > 0 and torch.isfinite(graph.y).all()
        for graph in graphs
    )

    raw_labels = pd.to_numeric(raw.p_np, errors='coerce')
    audit = {
        'raw_records': len(raw),
        'loader_records': len(dataset),
        'loader_excluded_records': len(raw) - len(dataset),
        'invalid_or_empty_raw': len(invalid_raw),
        'invalid_raw_indices': invalid_raw,
        'nonfinite_raw_labels': int((~np.isfinite(raw_labels)).sum()),
        'invalid_or_empty_loaded': len(invalid_loaded),
        'nonfinite_loaded_labels': len(nonfinite_labels),
        'duplicate_groups': len(duplicate_records),
        'duplicate_records': sum(group['records'] for group in duplicate_records),
        'different_smiles_duplicate_groups': sum(
            group['original_smiles_variants'] > 1 for group in duplicate_records
        ),
        'conflicting_groups': sum(group['conflicting'] for group in duplicate_records),
        'conflicting_records_removed': conflicting_records_removed,
        'consistent_duplicate_records_removed': consistent_records_removed,
        'retained_records': len(frame),
        'raw_class_counts': {
            str(label): int(count) for label, count in raw.p_np.value_counts().items()
        },
        'class_counts': {
            str(label): int(count) for label, count in frame.label.value_counts().items()
        },
        'class_proportions': {
            str(label): float(proportion)
            for label, proportion in frame.label.value_counts(normalize=True).items()
        },
        'acyclic_records': int((frame.scaffold == '').sum()),
        'distribution': frame[['atoms', 'bonds']].describe().to_dict(),
        'raw_source': str(dataset.url.format(dataset.names['bbbp'][1])),
    }
    return graphs, molecules, frame, audit, duplicate_records


def make_splits(frame):
    """Create random and scaffold splits, then check coverage and leakage."""
    all_indices = np.arange(len(frame))
    train_indices, remaining_indices = train_test_split(
        all_indices,
        test_size=0.30,
        stratify=frame.label,
        random_state=SPLIT_SEED,
    )
    validation_indices, test_indices = train_test_split(
        remaining_indices,
        test_size=0.50,
        stratify=frame.label.iloc[remaining_indices],
        random_state=SPLIT_SEED,
    )
    splits = {
        'random': {
            'train': train_indices,
            'validation': validation_indices,
            'test': test_indices,
        }
    }

    scaffold_groups = [
        group.index.to_numpy()
        for _, group in frame.groupby('scaffold', sort=True)
    ]
    generator = np.random.default_rng(SPLIT_SEED)
    generator.shuffle(scaffold_groups)
    # Stable sorting keeps the seeded order when groups have the same size.
    scaffold_groups.sort(key=len, reverse=True)

    subset_names = ['train', 'validation', 'test']
    target_sizes = np.array([0.70, 0.15, 0.15]) * len(frame)
    subset_indices = [[], [], []]
    current_sizes = np.zeros(3, dtype=int)
    for group in scaffold_groups:
        # Assign a whole group to the largest remaining capacity, without labels.
        chosen_subset = int(np.argmax(target_sizes - current_sizes))
        subset_indices[chosen_subset].extend(group.tolist())
        current_sizes[chosen_subset] += len(group)

    splits['scaffold'] = {
        name: np.array(indices, dtype=int)
        for name, indices in zip(subset_names, subset_indices)
    }

    summary_rows = []
    for strategy, subsets in splits.items():
        combined_indices = np.concatenate(list(subsets.values()))
        assert len(combined_indices) == len(frame)
        assert len(set(combined_indices)) == len(frame)
        assert set(combined_indices) == set(all_indices)

        for position, name in enumerate(subset_names):
            selected_records = frame.iloc[subsets[name]]
            assert selected_records.label.nunique() == 2, (
                'Both classes are needed for ROC-AUC.'
            )
            for other_name in subset_names[position + 1:]:
                other_records = frame.iloc[subsets[other_name]]
                assert set(subsets[name]).isdisjoint(subsets[other_name])
                assert set(selected_records.canonical_smiles).isdisjoint(
                    other_records.canonical_smiles
                )
                if strategy == 'scaffold':
                    assert set(selected_records.scaffold).isdisjoint(
                        other_records.scaffold
                    )

            summary_rows.append({
                'split': strategy,
                'subset': name,
                'molecules': len(selected_records),
                'positive': int(selected_records.label.sum()),
                'positive_fraction': float(selected_records.label.mean()),
                'unique_scaffolds': selected_records.scaffold.nunique(),
            })
    return splits, pd.DataFrame(summary_rows)


def check_batches(graphs, batch_size=64):
    """Check every batch, including the last partial batch, without losing graphs."""
    seen_record_ids = []
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    first_batch_shapes = None

    for batch in loader:
        assert batch.y.numel() == batch.num_graphs
        assert torch.isfinite(batch.y).all()
        assert torch.all(batch.ptr[1:] > batch.ptr[:-1])
        seen_record_ids.extend(batch.record_id.tolist())

        if first_batch_shapes is None:
            first_batch_shapes = {
                name: list(getattr(batch, name).shape)
                for name in ['x', 'edge_index', 'edge_attr', 'batch', 'y']
            }
            first_batch_shapes['num_graphs'] = batch.num_graphs

    assert sorted(seen_record_ids) == list(range(len(graphs)))
    first_batch_shapes['total_graphs_verified'] = len(seen_record_ids)
    return first_batch_shapes


def fingerprints(molecules):
    """Return Morgan fingerprints as both model inputs and similarity bitvectors."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=CONFIG['fingerprint_radius'],
        fpSize=CONFIG['fingerprint_bits'],
    )
    bitvectors = [generator.GetFingerprint(molecule) for molecule in molecules]
    matrix = np.stack([
        np.asarray(fingerprint, dtype=np.float32) for fingerprint in bitvectors
    ])
    return matrix, bitvectors


def choose_threshold(labels, probability):
    """Choose a validation threshold that balances the two class recalls."""
    threshold_grid = np.linspace(0.05, 0.95, 181)
    scores = [
        balanced_accuracy_score(labels, probability >= threshold)
        for threshold in threshold_grid
    ]
    best_score = max(scores)
    best_indices = [
        index for index, score in enumerate(scores)
        if np.isclose(score, best_score)
    ]
    # The grid and tie rule are fixed before fitting; test labels are never used.
    chosen_index = min(
        best_indices, key=lambda index: abs(threshold_grid[index] - 0.5)
    )
    return float(threshold_grid[chosen_index]), float(scores[chosen_index])


def fit_baseline(matrix, labels, subset):
    """Select logistic-regression strength and threshold using validation only."""
    train_indices = subset['train']
    validation_indices = subset['validation']
    fitted_models = []
    candidate_results = []

    for inverse_regularization_strength in CONFIG['baseline_C']:
        model = LogisticRegression(
            C=inverse_regularization_strength,
            class_weight='balanced',
            solver='liblinear',
            max_iter=2000,
            random_state=SPLIT_SEED,
        )
        model.fit(matrix[train_indices], labels[train_indices])
        validation_probability = model.predict_proba(matrix[validation_indices])[:, 1]
        candidate_results.append({
            'C': inverse_regularization_strength,
            'validation_roc_auc': roc_auc_score(
                labels[validation_indices], validation_probability
            ),
        })
        fitted_models.append((model, validation_probability))

    selected_index = int(np.argmax([
        result['validation_roc_auc'] for result in candidate_results
    ]))
    model, validation_probability = fitted_models[selected_index]
    threshold, balanced_accuracy = choose_threshold(
        labels[validation_indices], validation_probability
    )
    metadata = {
        'C': candidate_results[selected_index]['C'],
        'threshold': threshold,
        'validation_roc_auc': candidate_results[selected_index]['validation_roc_auc'],
        'validation_balanced_accuracy': balanced_accuracy,
        'class_weight': 'balanced (training subset only)',
        'solver': 'liblinear',
        'max_iter': 2000,
        'seed': SPLIT_SEED,
        'candidate_results': candidate_results,
    }
    return model, metadata


class CategoricalEncoder(nn.Module):
    """Embed each categorical column separately and add its learned vectors."""

    def __init__(self, cardinalities, hidden):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(category_count, hidden) for category_count in cardinalities
        ])

    def forward(self, codes):
        column_embeddings = (
            embedding(codes[:, column])
            for column, embedding in enumerate(self.embeddings)
        )
        return sum(column_embeddings)


class MolecularGNN(nn.Module):
    """Three bond-aware GINE layers followed by one prediction per molecule."""

    def __init__(self):
        super().__init__()
        hidden_size = CONFIG['hidden']
        atom_cardinalities = [len(categories) for categories in x_map.values()]
        bond_cardinalities = [len(categories) for categories in e_map.values()]
        self.atom_encoder = CategoricalEncoder(atom_cardinalities, hidden_size)
        self.bond_encoder = CategoricalEncoder(bond_cardinalities, hidden_size)

        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                ),
                train_eps=True,
            )
            for _ in range(CONFIG['layers'])
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in self.convs
        ])
        self.dropout = nn.Dropout(CONFIG['dropout'])
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, graph):
        atom_vectors = self.atom_encoder(graph.x)
        bond_vectors = self.bond_encoder(graph.edge_attr)

        for convolution, normalization in zip(self.convs, self.norms):
            # Every message uses the neighbor's vector and its connecting bond.
            messages = convolution(atom_vectors, graph.edge_index, bond_vectors)
            updated_vectors = normalization(atom_vectors + messages)
            atom_vectors = self.dropout(torch.relu(updated_vectors))

        # Pool only within each molecule, then output one unnormalized logit.
        molecule_vectors = global_mean_pool(atom_vectors, graph.batch)
        return self.head(molecule_vectors).view(-1)


@torch.no_grad()
def predict_gnn(model, graphs, indices, device):
    """Predict probabilities in the exact order of the requested graph indices."""
    model.eval()
    probabilities = []
    selected_graphs = [graphs[index] for index in indices]
    loader = DataLoader(
        selected_graphs, batch_size=CONFIG['batch_size'], shuffle=False
    )
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probabilities.extend(logits.sigmoid().cpu().numpy().tolist())
    return np.asarray(probabilities)


def fit_gnn(graphs, labels, subset, seed, device, checkpoint_path):
    """Train one seeded GNN and retain its best validation ROC-AUC checkpoint."""
    seed_everything(seed)
    model = MolecularGNN().to(device)
    train_indices = subset['train']
    validation_indices = subset['validation']

    # Training prevalence determines the loss weight, including validation loss.
    positive_weight = float(
        (labels[train_indices] == 0).sum() / (labels[train_indices] == 1).sum()
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        [graphs[index] for index in train_indices],
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        [graphs[index] for index in validation_indices],
        batch_size=CONFIG['batch_size'],
        shuffle=False,
    )

    best_score = -np.inf
    best_epoch = 0
    best_state = None
    history = []
    start = time.monotonic()

    for epoch in range(1, CONFIG['max_epochs'] + 1):
        model.train()
        training_loss_sum = 0.0
        training_graph_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
            training_loss_sum += loss.item() * batch.num_graphs
            training_graph_count += batch.num_graphs

        model.eval()
        validation_loss_sum = 0.0
        validation_probabilities = []
        with torch.no_grad():
            for batch in validation_loader:
                batch = batch.to(device)
                logits = model(batch)
                validation_loss_sum += (
                    criterion(logits, batch.y.view(-1)).item() * batch.num_graphs
                )
                validation_probabilities.extend(logits.sigmoid().cpu().numpy().tolist())

        validation_auc = float(roc_auc_score(
            labels[validation_indices], validation_probabilities
        ))
        mean_training_loss = training_loss_sum / training_graph_count
        mean_validation_loss = validation_loss_sum / len(validation_indices)
        history.append({
            'epoch': epoch,
            'train_loss': mean_training_loss,
            'validation_loss': mean_validation_loss,
            'validation_roc_auc': validation_auc,
        })

        if validation_auc > best_score + 1e-6:
            best_score = validation_auc
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 5 == 0:
            print(
                f'  seed={seed} epoch={epoch:02d} train={mean_training_loss:.4f} '
                f'val={mean_validation_loss:.4f} AUC={validation_auc:.4f} '
                f'best={best_epoch} elapsed={time.monotonic() - start:.1f}s',
                flush=True,
            )
        if epoch - best_epoch >= CONFIG['patience']:
            break

    model.load_state_dict(best_state)
    torch.save({
        'state_dict': best_state,
        'config': CONFIG,
        'seed': seed,
        'best_epoch': best_epoch,
    }, checkpoint_path)

    # The chosen checkpoint still uses validation data to choose its threshold.
    validation_probability = predict_gnn(
        model, graphs, validation_indices, device
    )
    threshold, balanced_accuracy = choose_threshold(
        labels[validation_indices], validation_probability
    )
    metadata = {
        'seed': seed,
        'threshold': threshold,
        'best_epoch': best_epoch,
        'epochs_run': epoch,
        'validation_roc_auc': best_score,
        'validation_balanced_accuracy': balanced_accuracy,
        'pos_weight': positive_weight,
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'training_seconds': time.monotonic() - start,
    }
    return model, pd.DataFrame(history), metadata


def classification_metrics(labels, probability, threshold):
    """Compute ranking and decision metrics using a previously frozen threshold."""
    predicted = probability >= threshold
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predicted, labels=[0, 1]
    ).ravel()
    return {
        'roc_auc': float(roc_auc_score(labels, probability)),
        'pr_auc': float(average_precision_score(labels, probability)),
        'accuracy': float(accuracy_score(labels, predicted)),
        'sensitivity': float(true_positive / (true_positive + false_negative)),
        'specificity': float(true_negative / (true_negative + false_positive)),
        'f1': float(f1_score(labels, predicted, zero_division=0)),
        'tn': int(true_negative),
        'fp': int(false_positive),
        'fn': int(false_negative),
        'tp': int(true_positive),
    }
