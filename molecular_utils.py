"""Molecular graphs, leakage checks, and small reproducible BBBP models."""

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils.smiles import e_map, x_map

SPLIT_SEED = 2026
SEEDS = [17, 29, 43]
CONFIG = dict(hidden=64, layers=3, dropout=0.2, learning_rate=0.001,
              weight_decay=0.0001, batch_size=64, max_epochs=80, patience=15,
              fingerprint_radius=2, fingerprint_bits=2048,
              baseline_C=[0.01, 0.1, 1.0, 10.0], split_seed=SPLIT_SEED,
              model_seeds=SEEDS, threshold_metric='balanced_accuracy',
              checkpoint_metric='validation_roc_auc')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CPU is deterministic; MPS scatter reductions can vary at floating precision.
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_clean_dataset(root):
    """Audit the original download and apply one canonical molecule per label."""
    dataset = MoleculeNet(root=str(root), name='BBBP')
    raw = pd.read_csv(dataset.raw_paths[0])
    raw_mols = [Chem.MolFromSmiles(str(s)) for s in raw.smiles]
    invalid_raw = [i for i, m in enumerate(raw_mols)
                   if m is None or m.GetNumAtoms() == 0]
    groups = {}
    invalid_loaded, nonfinite = [], []
    for idx, graph in enumerate(dataset):
        mol = Chem.MolFromSmiles(graph.smiles)
        if mol is None or mol.GetNumAtoms() == 0 or graph.num_nodes == 0:
            invalid_loaded.append(idx)
            continue
        if graph.y.numel() != 1 or not torch.isfinite(graph.y).all():
            nonfinite.append(idx)
            continue
        label = int(graph.y.item())
        assert label in (0, 1)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        groups.setdefault(canonical, []).append((idx, label, graph, mol))
    graphs, molecules, records, duplicate_records = [], [], [], []
    conflicts, discarded_consistent = 0, 0
    for canonical, items in groups.items():
        conflict = len({item[1] for item in items}) > 1
        if len(items) > 1:
            duplicate_records.append(dict(canonical_smiles=canonical,
                                          records=len(items),
                                          labels=','.join(str(x[1]) for x in items),
                                          original_smiles_variants=len({x[2].smiles for x in items}),
                                          conflicting=conflict))
        if conflict:
            conflicts += len(items)
            continue
        discarded_consistent += len(items) - 1
        idx, label, graph, mol = items[0]
        graph = graph.clone()
        graph.record_id = torch.tensor([len(graphs)], dtype=torch.long)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        # All acyclic molecules share the empty scaffold and stay together.
        graphs.append(graph)
        molecules.append(mol)
        records.append(dict(record_id=len(records), loader_index=idx,
                            smiles=graph.smiles, canonical_smiles=canonical,
                            label=label, atoms=mol.GetNumAtoms(),
                            bonds=mol.GetNumBonds(), scaffold=scaffold))
    frame = pd.DataFrame(records)
    assert len(frame.canonical_smiles.unique()) == len(frame)
    assert all(g.num_nodes > 0 and torch.isfinite(g.y).all() for g in graphs)
    audit = dict(raw_records=len(raw), loader_records=len(dataset),
                 loader_excluded_records=len(raw)-len(dataset),
                 invalid_or_empty_raw=len(invalid_raw), invalid_raw_indices=invalid_raw,
                 nonfinite_raw_labels=int((~np.isfinite(pd.to_numeric(raw.p_np, errors='coerce'))).sum()),
                 invalid_or_empty_loaded=len(invalid_loaded),
                 nonfinite_loaded_labels=len(nonfinite),
                 duplicate_groups=len(duplicate_records),
                 duplicate_records=sum(d['records'] for d in duplicate_records),
                 different_smiles_duplicate_groups=sum(d['original_smiles_variants'] > 1 for d in duplicate_records),
                 conflicting_groups=sum(d['conflicting'] for d in duplicate_records),
                 conflicting_records_removed=conflicts,
                 consistent_duplicate_records_removed=discarded_consistent,
                 retained_records=len(frame),
                 raw_class_counts={str(k): int(v) for k,v in raw.p_np.value_counts().items()},
                 class_counts={str(k): int(v) for k,v in frame.label.value_counts().items()},
                 class_proportions={str(k): float(v) for k,v in frame.label.value_counts(normalize=True).items()},
                 acyclic_records=int((frame.scaffold == '').sum()),
                 distribution=frame[['atoms', 'bonds']].describe().to_dict(),
                 raw_source=str(dataset.url.format(dataset.names['bbbp'][1])))
    return graphs, molecules, frame, audit, duplicate_records


def make_splits(frame):
    """Fixed stratified random and size-only scaffold partitioning."""
    indices = np.arange(len(frame))
    train, rest = train_test_split(indices, test_size=0.30, stratify=frame.label,
                                  random_state=SPLIT_SEED)
    val, test = train_test_split(rest, test_size=0.50,
                               stratify=frame.label.iloc[rest], random_state=SPLIT_SEED)
    splits = {'random': dict(train=train, validation=val, test=test)}
    groups = [g.index.to_numpy() for _, g in frame.groupby('scaffold', sort=True)]
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)  # Stable: seeded order breaks equal-size ties.
    names = ['train', 'validation', 'test']
    targets = np.array([0.70, 0.15, 0.15]) * len(frame)
    buckets = [[], [], []]
    sizes = np.zeros(3, dtype=int)
    # Greedy largest remaining capacity: no label or model information is used.
    for group in groups:
        chosen = int(np.argmax(targets - sizes))
        buckets[chosen].extend(group.tolist())
        sizes[chosen] += len(group)
    splits['scaffold'] = {name: np.array(ids, dtype=int)
                          for name, ids in zip(names, buckets)}
    rows = []
    for strategy, subsets in splits.items():
        combined = np.concatenate(list(subsets.values()))
        assert len(combined) == len(frame) and len(set(combined)) == len(frame)
        assert set(combined) == set(indices)
        for i, name in enumerate(names):
            selected = frame.iloc[subsets[name]]
            assert selected.label.nunique() == 2, 'Both classes are needed for ROC-AUC.'
            for other in names[i+1:]:
                assert set(subsets[name]).isdisjoint(subsets[other])
                assert set(selected.canonical_smiles).isdisjoint(frame.iloc[subsets[other]].canonical_smiles)
                if strategy == 'scaffold':
                    assert set(selected.scaffold).isdisjoint(frame.iloc[subsets[other]].scaffold)
            rows.append(dict(split=strategy, subset=name, molecules=len(selected),
                             positive=int(selected.label.sum()),
                             positive_fraction=float(selected.label.mean()),
                             unique_scaffolds=selected.scaffold.nunique()))
    return splits, pd.DataFrame(rows)


def check_batches(graphs, batch_size=64):
    seen = []
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    first = None
    for batch in loader:
        assert batch.y.numel() == batch.num_graphs
        assert torch.isfinite(batch.y).all()
        assert torch.all(batch.ptr[1:] > batch.ptr[:-1])
        seen.extend(batch.record_id.tolist())
        if first is None:
            first = {k: list(getattr(batch, k).shape)
                     for k in ['x', 'edge_index', 'edge_attr', 'batch', 'y']}
            first.update(num_graphs=batch.num_graphs)
    assert sorted(seen) == list(range(len(graphs)))
    first['total_graphs_verified'] = len(seen)
    return first


def fingerprints(molecules):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=CONFIG['fingerprint_radius'],
                                                         fpSize=CONFIG['fingerprint_bits'])
    bitvectors = [generator.GetFingerprint(mol) for mol in molecules]
    matrix = np.stack([np.asarray(fp, dtype=np.float32) for fp in bitvectors])
    return matrix, bitvectors


def choose_threshold(y, probability):
    # Fixed grid declared before fitting; ties choose the threshold nearest 0.5.
    grid = np.linspace(0.05, 0.95, 181)
    scores = [balanced_accuracy_score(y, probability >= t) for t in grid]
    best = max(scores)
    candidates = [i for i,s in enumerate(scores) if np.isclose(s,best)]
    idx = min(candidates, key=lambda i: abs(grid[i] - 0.5))
    return float(grid[idx]), float(scores[idx])


def fit_baseline(matrix, labels, subset):
    train, val = subset['train'], subset['validation']
    fitted, rows = [], []
    for c in CONFIG['baseline_C']:
        model = LogisticRegression(C=c, class_weight='balanced', solver='liblinear',
                                   max_iter=2000, random_state=SPLIT_SEED)
        model.fit(matrix[train], labels[train])
        p = model.predict_proba(matrix[val])[:, 1]
        rows.append(dict(C=c, validation_roc_auc=roc_auc_score(labels[val], p)))
        fitted.append((model,p))
    selected = int(np.argmax([row['validation_roc_auc'] for row in rows]))
    model,p = fitted[selected]
    threshold, score = choose_threshold(labels[val], p)
    metadata = dict(C=rows[selected]['C'], threshold=threshold,
                    validation_roc_auc=rows[selected]['validation_roc_auc'],
                    validation_balanced_accuracy=score,
                    class_weight='balanced (training subset only)', solver='liblinear',
                    max_iter=2000, seed=SPLIT_SEED, candidate_results=rows)
    return model, metadata


class CategoricalEncoder(nn.Module):
    def __init__(self, cardinalities, hidden):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n,hidden) for n in cardinalities])

    def forward(self, codes):
        return sum(embedding(codes[:,i]) for i,embedding in enumerate(self.embeddings))


class MolecularGNN(nn.Module):
    """Three bond-aware GINE layers with residuals and mean graph pooling."""
    def __init__(self):
        super().__init__()
        hidden = CONFIG['hidden']
        self.atom_encoder = CategoricalEncoder([len(v) for v in x_map.values()], hidden)
        self.bond_encoder = CategoricalEncoder([len(v) for v in e_map.values()], hidden)
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),
                                   nn.Linear(hidden,hidden)),train_eps=True)
            for _ in range(CONFIG['layers'])])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in self.convs])
        self.dropout = nn.Dropout(CONFIG['dropout'])
        self.head = nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),
                                  nn.Dropout(CONFIG['dropout']), nn.Linear(hidden,1))

    def forward(self, graph):
        h = self.atom_encoder(graph.x)
        e = self.bond_encoder(graph.edge_attr)
        for conv,norm in zip(self.convs,self.norms):
            h = self.dropout(torch.relu(norm(h + conv(h,graph.edge_index,e))))
        return self.head(global_mean_pool(h,graph.batch)).view(-1)


@torch.no_grad()
def predict_gnn(model, graphs, indices, device):
    model.eval()
    probability = []
    for batch in DataLoader([graphs[i] for i in indices], batch_size=CONFIG['batch_size'],shuffle=False):
        probability.extend(model(batch.to(device)).sigmoid().cpu().numpy().tolist())
    return np.asarray(probability)


def fit_gnn(graphs, labels, subset, seed, device, checkpoint_path):
    seed_everything(seed)
    model = MolecularGNN().to(device)
    train,val = subset['train'], subset['validation']
    pos_weight = float((labels[train] == 0).sum() / (labels[train] == 1).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=device))
    optimizer = torch.optim.Adam(model.parameters(),lr=CONFIG['learning_rate'],
                                 weight_decay=CONFIG['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader([graphs[i] for i in train],batch_size=CONFIG['batch_size'],
                        shuffle=True,generator=generator)
    val_loader = DataLoader([graphs[i] for i in val],batch_size=CONFIG['batch_size'],shuffle=False)
    best_score,best_epoch,best_state = -np.inf,0,None
    history = []
    start = time.monotonic()
    for epoch in range(1,CONFIG['max_epochs']+1):
        model.train()
        train_loss,total = 0.0,0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch),batch.y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            total += batch.num_graphs
        model.eval()
        validation_loss,probabilities = 0.0,[]
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                validation_loss += criterion(logits,batch.y.view(-1)).item() * batch.num_graphs
                probabilities.extend(logits.sigmoid().cpu().numpy().tolist())
        auc = float(roc_auc_score(labels[val],probabilities))
        history.append(dict(epoch=epoch,train_loss=train_loss/total,
                            validation_loss=validation_loss/len(val),validation_roc_auc=auc))
        if auc > best_score + 1e-6:
            best_score,best_epoch = auc,epoch
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(f'  seed={seed} epoch={epoch:02d} train={train_loss/total:.4f} '
                  f'val={validation_loss/len(val):.4f} AUC={auc:.4f} '
                  f'best={best_epoch} elapsed={time.monotonic()-start:.1f}s',flush=True)
        if epoch-best_epoch >= CONFIG['patience']:
            break
    model.load_state_dict(best_state)
    torch.save(dict(state_dict=best_state,config=CONFIG,seed=seed,best_epoch=best_epoch),checkpoint_path)
    p = predict_gnn(model,graphs,val,device)
    threshold, score = choose_threshold(labels[val],p)
    metadata = dict(seed=seed,threshold=threshold,best_epoch=best_epoch,
                    epochs_run=epoch,validation_roc_auc=best_score,
                    validation_balanced_accuracy=score,pos_weight=pos_weight,
                    parameters=sum(p.numel() for p in model.parameters()),
                    training_seconds=time.monotonic()-start)
    return model,pd.DataFrame(history),metadata


def classification_metrics(y,p,threshold):
    predicted = p >= threshold
    tn,fp,fn,tp = confusion_matrix(y,predicted,labels=[0,1]).ravel()
    return dict(roc_auc=float(roc_auc_score(y,p)),pr_auc=float(average_precision_score(y,p)),
                accuracy=float(accuracy_score(y,predicted)),sensitivity=float(tp/(tp+fn)),
                specificity=float(tn/(tn+fp)),f1=float(f1_score(y,predicted,zero_division=0)),
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
