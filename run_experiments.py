"""Run the full BBBP experiment; all settings are declared in molecular_utils."""

import hashlib
import json
import multiprocessing as mp
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
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

from molecular_utils import (CONFIG, SEEDS, SPLIT_SEED, MolecularGNN,
                             check_batches, classification_metrics, fingerprints,
                             fit_baseline, fit_gnn, load_clean_dataset, make_splits,
                             predict_gnn, save_json, seed_everything)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
FIG = OUT / 'figures'
TABLES = OUT / 'tables'
MODELS = OUT / 'models'
METRICS = ['roc_auc', 'pr_auc', 'accuracy', 'sensitivity', 'specificity', 'f1']
COLORS = ['#173D54', '#B26129', '#307A67', '#9464A5']


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest():
    return {name: file_hash(ROOT/name) for name in ['molecular_utils.py','run_experiments.py']}


def save_figure(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def inspect_figures(graphs,molecules,frame):
    examples = []
    for label in [0,1]:
        selected = frame[frame.label==label].assign(length=lambda d:d.smiles.str.len())
        examples.extend(selected.sort_values(['length','record_id']).head(3).record_id.tolist())
    frame.iloc[examples].to_csv(TABLES/'examples.csv',index=False)
    image = Draw.MolsToGridImage([molecules[i] for i in examples],molsPerRow=3,
                                 subImgSize=(330,220),
                                 legends=[f'ID {i}; label {frame.label.iloc[i]}' for i in examples])
    image.save(FIG/'molecule_examples.png')
    # A small cyclic molecule with at least one bond keeps atom correspondence clear.
    graph_id = next(i for i,m in enumerate(molecules)
                    if 8 <= m.GetNumAtoms() <= 14 and m.GetNumBonds()>0 and m.GetRingInfo().NumRings()>0)
    mol = Chem.Mol(molecules[graph_id])
    rdDepictor.Compute2DCoords(mol)
    for atom in mol.GetAtoms():
        atom.SetProp('atomNote',str(atom.GetIdx()))
    fig,axes = plt.subplots(1,2,figsize=(10,3.6))
    axes[0].imshow(Draw.MolToImage(mol,size=(650,420)))
    axes[0].axis('off'); axes[0].set_title('Chemical structure (atom IDs)')
    network = nx.Graph()
    network.add_nodes_from(range(mol.GetNumAtoms()))
    network.add_edges_from((b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in mol.GetBonds())
    conf = mol.GetConformer()
    positions = {i:(conf.GetAtomPosition(i).x,conf.GetAtomPosition(i).y) for i in network}
    nx.draw_networkx(network,pos=positions,ax=axes[1],node_color='#DCECF1',node_size=460,
                     labels={i:f'{i}:{mol.GetAtomWithIdx(i).GetSymbol()}' for i in network},font_size=8)
    axes[1].axis('off'); axes[1].set_title('Graph: matching atom IDs; bonds are edges')
    fig.suptitle(f'Record {graph_id}; label={frame.label.iloc[graph_id]}')
    save_figure(fig,'graph_views')
    fig,axes = plt.subplots(1,2,figsize=(9,2.8))
    for ax,column in zip(axes,['atoms','bonds']):
        ax.hist(frame[column],bins=35,color=COLORS[0],edgecolor='white')
        ax.set(xlabel=f'{column.capitalize()} per molecule',ylabel='Molecules')
    fig.tight_layout(); save_figure(fig,'size_distributions')
    graph = graphs[graph_id]
    return dict(record_id=graph_id,smiles=graph.smiles,
                x_shape=list(graph.x.shape),edge_index_shape=list(graph.edge_index.shape),
                edge_attr_shape=list(graph.edge_attr.shape),y=graph.y.tolist(),
                x_first_atom=graph.x[0].tolist(),edge_attr_first_bond=graph.edge_attr[0].tolist())


def time_device(batch, name):
    seed_everything(SPLIT_SEED)
    torch.set_num_threads(4)
    model = MolecularGNN().to(name)
    candidate = batch.clone().to(name)
    model.train()
    for _ in range(2):
        model.zero_grad(); model(candidate).sum().backward()
    if name=='mps': torch.mps.synchronize()
    start = time.monotonic()
    for _ in range(5):
        model.zero_grad(); model(candidate).sum().backward()
    if name=='mps': torch.mps.synchronize()
    return (time.monotonic()-start)/5


def mps_worker(batch, result):
    try:
        result.put(time_device(batch,'mps'))
    except (RuntimeError,NotImplementedError) as error:
        result.put(str(error).split('\n')[0])


def select_device(graphs):
    """Benchmark computation only; no test labels or model performance consulted."""
    batch = next(iter(DataLoader(graphs[:64],batch_size=64)))
    timings = {'cpu':time_device(batch,'cpu')}
    if torch.backends.mps.is_available():
        context=mp.get_context('spawn'); result=context.Queue()
        process=context.Process(target=mps_worker,args=(batch,result))
        process.start(); process.join(timeout=20)
        if process.is_alive():
            process.terminate(); process.join()
            timings['mps']='Exceeded 20-second setup/benchmark limit; CPU selected.'
        else:
            timings['mps']=result.get(timeout=2) if process.exitcode==0 else f'Benchmark worker exit {process.exitcode}'
    valid = {k:v for k,v in timings.items() if isinstance(v,float)}
    selected = min(valid,key=valid.get)
    print(f'Device benchmark seconds/forward+backward: {timings}; selected {selected}',flush=True)
    return selected,timings


def training_figure(histories):
    fig,axes = plt.subplots(2,3,figsize=(12,6))
    for row,strategy in enumerate(['random','scaffold']):
        for color,seed in zip(COLORS,SEEDS):
            h = histories[f'{strategy}_{seed}']
            for col,column in enumerate(['train_loss','validation_loss','validation_roc_auc']):
                axes[row,col].plot(h.epoch,h[column],color=color,label=f'Seed {seed}',lw=1.4)
                axes[row,col].set(xlabel='Epoch',ylabel=column.replace('_',' '),title=strategy.capitalize())
                axes[row,col].grid(alpha=0.15)
        axes[row,0].legend(fontsize=8)
    fig.tight_layout(); save_figure(fig,'training_curves')


def evaluation_figures(runs,predictions,labels,splits):
    fig,axes = plt.subplots(2,2,figsize=(10,8))
    cmfig,cmaxes = plt.subplots(2,4,figsize=(12,5.5))
    for row,strategy in enumerate(['random','scaffold']):
        y = labels[splits[strategy]['test']]
        selected = [run for run in runs if run['split']==strategy]
        for col,(run,color) in enumerate(zip(selected,COLORS)):
            key=run['run_id']; p=predictions[key]
            title='Morgan LR' if run['model']=='Morgan LR' else f"GINE seed {run['seed']}"
            fpr,tpr,_ = roc_curve(y,p)
            precision,recall,_ = precision_recall_curve(y,p)
            axes[row,0].plot(fpr,tpr,color=color,label=f"{title} ({run['roc_auc']:.3f})")
            axes[row,1].plot(recall,precision,color=color,label=f"{title} ({run['pr_auc']:.3f})")
            matrix=np.array([[run['tn'],run['fp']],[run['fn'],run['tp']]])
            ax=cmaxes[row,col]; ax.imshow(matrix,cmap='Blues',vmin=0,vmax=len(y))
            for (i,j),value in np.ndenumerate(matrix):
                ax.text(j,i,str(value),ha='center',va='center',
                        color='white' if value>len(y)/2 else 'black',fontsize=10)
            ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted label',ylabel='True label',
                   title=f"{strategy}: {title}\nt={run['threshold']:.3f}")
        axes[row,0].plot([0,1],[0,1],'k--',lw=0.7)
        axes[row,1].axhline(y.mean(),color='gray',ls='--',lw=0.7)
        axes[row,0].set(title=f'{strategy.capitalize()} ROC',xlabel='False-positive rate',ylabel='Sensitivity')
        axes[row,1].set(title=f'{strategy.capitalize()} precision-recall',xlabel='Recall',ylabel='Precision')
        for ax in axes[row]:
            ax.legend(fontsize=7); ax.grid(alpha=0.15)
    fig.tight_layout(); save_figure(fig,'test_curves')
    cmfig.tight_layout(); save_figure(cmfig,'confusion_matrices')


def error_analysis(frame,molecules,graphs,bitvectors,labels,splits,runs,predictions,models,device):
    # Fixed before any test evaluation: scaffold GNN seed 17, first two errors by ID.
    run=next(r for r in runs if r['run_id']=='scaffold_gnn_17')
    test=splits['scaffold']['test']; train=splits['scaffold']['train']
    p=predictions[run['run_id']]; predicted=p>=run['threshold']
    selected=[]
    for error,label in [('FP',0),('FN',1)]:
        locs=[i for i in range(len(test)) if labels[test[i]]==label and predicted[i]!=label]
        locs.sort(key=lambda j:int(test[j]))
        selected.extend((error,j) for j in locs[:2])
    train_probability=predict_gnn(models['scaffold_gnn_17'],graphs,train,device)
    details=[]
    figure,axes=plt.subplots(max(1,len(selected)),2,figsize=(10,2.5*max(1,len(selected))))
    axes=np.asarray(axes).reshape(-1,2)
    for row,(error,j) in enumerate(selected):
        idx=int(test[j])
        similarity=np.asarray(DataStructs.BulkTanimotoSimilarity(bitvectors[idx],[bitvectors[i] for i in train]))
        nearest=int(np.argmax(similarity)); neighbor=int(train[nearest])
        details.append(dict(error=error,record_id=idx,smiles=frame.smiles.iloc[idx],
                            canonical_smiles=frame.canonical_smiles.iloc[idx],label=int(labels[idx]),
                            probability=float(p[j]),threshold=run['threshold'],
                            neighbor_id=neighbor,neighbor_smiles=frame.smiles.iloc[neighbor],
                            neighbor_label=int(labels[neighbor]),neighbor_probability=float(train_probability[nearest]),
                            tanimoto=float(similarity[nearest]),seed=17))
        for col,(mol_id,prob,label) in enumerate([(idx,p[j],labels[idx]),
                                                 (neighbor,train_probability[nearest],labels[neighbor])]):
            axes[row,col].imshow(Draw.MolToImage(molecules[mol_id],size=(620,270)))
            axes[row,col].axis('off')
            prefix=f'{error} test' if col==0 else 'Nearest training molecule'
            suffix='' if col==0 else f'; Tanimoto={similarity[nearest]:.3f}'
            axes[row,col].set_title(f'{prefix} ID {mol_id}; y={label}; p={prob:.3f}{suffix}',fontsize=9)
    figure.tight_layout(); save_figure(figure,'error_pairs')
    pd.DataFrame(details).to_csv(TABLES/'error_details.csv',index=False)
    return details


def run_pipeline(reuse_results=True):
    """Reuse only a hash-validated complete local run; otherwise fit from scratch."""
    for folder in [OUT,FIG,TABLES,MODELS]: folder.mkdir(exist_ok=True,parents=True)
    manifest_path=OUT/'manifest.json'
    if reuse_results and manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        valid=(manifest['sources']==source_manifest() and manifest['config']==CONFIG)
        valid=valid and all((ROOT/name).is_file() and file_hash(ROOT/name)==digest
                            for name,digest in manifest['artifacts'].items())
        if valid:
            print('Reusing complete hash-validated experiment, without re-evaluating test sets.',flush=True)
            return json.loads((OUT/'summary.json').read_text())
        print('Cached artifacts incomplete or changed: starting a fresh experiment.',flush=True)
    start=time.monotonic(); milestones={}
    seed_everything(SPLIT_SEED); torch.set_num_threads(4)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    graphs,molecules,frame,audit,duplicates=load_clean_dataset(ROOT/'data/MoleculeNet')
    frame.to_csv(TABLES/'molecules.csv',index=False)
    pd.DataFrame(duplicates).to_csv(TABLES/'canonical_duplicates.csv',index=False)
    print(f'Dataset audit: {audit}',flush=True)
    graph_inspection=inspect_figures(graphs,molecules,frame)
    batch_inspection=check_batches(graphs)
    splits,split_summary=make_splits(frame)
    split_summary.to_csv(TABLES/'split_summary.csv',index=False)
    split_manifest=frame[['record_id','canonical_smiles','scaffold','label']].copy()
    for name,subset in splits.items():
        for partition,indices in subset.items(): split_manifest.loc[indices,f'{name}_subset']=partition
    split_manifest.to_csv(TABLES/'split_manifest.csv',index=False)
    print(split_summary.to_string(index=False),flush=True)
    matrix,bitvectors=fingerprints(molecules)
    labels=frame.label.to_numpy()
    milestones['data_audit_and_splitting_seconds']=time.monotonic()-start
    common_training = sorted(set(splits['random']['train']) & set(splits['scaffold']['train']))
    device,device_benchmark=select_device([graphs[i] for i in common_training[:64]])
    models,selection,histories={}, {}, {}
    train_start=time.monotonic()
    for strategy,subset in splits.items():
        print(f'Fitting {strategy} Morgan baseline (training/validation only)',flush=True)
        model,metadata=fit_baseline(matrix,labels,subset)
        key=f'{strategy}_lr'; models[key]=model; selection[key]=metadata
        for seed in SEEDS:
            key=f'{strategy}_gnn_{seed}'
            print(f'Fitting {key} (training/validation only)',flush=True)
            model,history,metadata=fit_gnn(graphs,labels,subset,seed,device,MODELS/f'{key}.pt')
            models[key]=model; selection[key]=metadata
            histories[f'{strategy}_{seed}']=history
            history.to_csv(TABLES/f'{key}_history.csv',index=False)
    milestones['model_selection_seconds']=time.monotonic()-train_start
    # Persist the lock before any model sees a test graph or any test metric.
    lock=dict(config=CONFIG,selection=selection,sources=source_manifest(),
              split_manifest_sha256=file_hash(TABLES/'split_manifest.csv'),
              test_evaluation_started=False,error_analysis_model='scaffold_gnn_17')
    save_json(OUT/'evaluation_lock.json',lock)
    print('All eight final models and thresholds frozen. Beginning final test evaluation.',flush=True)
    evaluation_start=time.monotonic(); runs=[]; predictions={}
    for strategy,subset in splits.items():
        for suffix in ['lr']+[f'gnn_{seed}' for seed in SEEDS]:
            key=f'{strategy}_{suffix}'; model=models[key]; test=subset['test']
            p=(model.predict_proba(matrix[test])[:,1] if suffix=='lr'
               else predict_gnn(model,graphs,test,device))
            predictions[key]=p; meta=selection[key]
            metrics=classification_metrics(labels[test],p,meta['threshold'])
            row=dict(run_id=key,model='Morgan LR' if suffix=='lr' else 'GINE',split=strategy,
                     **meta,**metrics)
            runs.append(row)
            result=frame.iloc[test][['record_id','canonical_smiles','label']].copy()
            result['probability']=p; result['threshold']=meta['threshold']
            result['prediction']=(p>=meta['threshold']).astype(int)
            result.to_csv(TABLES/f'{key}_predictions.csv',index=False)
            print(f'Final {key}: {metrics}',flush=True)
    pd.DataFrame(runs).drop(columns=['candidate_results'],errors='ignore').to_csv(TABLES/'comparison.csv',index=False)
    aggregate=[]
    for strategy in splits:
        subset=[r for r in runs if r['split']==strategy and r['model']=='GINE']
        row=dict(model='GINE',split=strategy,n_seeds=3)
        for metric in METRICS+['threshold']:
            values=[r[metric] for r in subset]
            row[f'{metric}_mean']=float(np.mean(values))
            row[f'{metric}_std']=float(np.std(values,ddof=1))
        aggregate.append(row)
    pd.DataFrame(aggregate).to_csv(TABLES/'seed_aggregate.csv',index=False)
    training_figure(histories)
    evaluation_figures(runs,predictions,labels,splits)
    errors=error_analysis(frame,molecules,graphs,bitvectors,labels,splits,runs,predictions,models,device)
    milestones['test_evaluation_and_figures_seconds']=time.monotonic()-evaluation_start
    milestones['total_seconds']=time.monotonic()-start
    summary=dict(audit=audit,config=CONFIG,device=device,device_benchmark=device_benchmark,
                 environment=dict(python=platform.python_version(),torch=torch.__version__,
                                  torch_geometric=torch_geometric.__version__,rdkit=rdBase.rdkitVersion,
                                  platform=platform.platform(),cpu_threads=4),
                 feature_schema=dict(atom=[dict(name=k,cardinality=len(v)) for k,v in x_map.items()],
                                     bond=[dict(name=k,cardinality=len(v)) for k,v in e_map.items()],
                                     encoding='Independent learned embeddings per categorical column; sum within atom or bond'),
                 graph_inspection=graph_inspection,batch_inspection=batch_inspection,
                 split_summary=split_summary.to_dict('records'),
                 split_method='Scaffold groups shuffled with seed2026, stable sorted largest first, allocated to largest remaining target capacity without labels. Empty scaffold is one group.',
                 threshold_method='Maximum validation balanced accuracy on 181 points from0.05 to0.95; ties nearest0.5.',
                 pr_auc_definition='Average precision (non-interpolated area under precision-recall curve; sklearn average_precision_score)',
                 baseline_selection={key:value for key,value in selection.items() if key.endswith('_lr')},
                 runs=runs,aggregate=aggregate,error_examples=errors,timings=milestones,
                 assertions=dict(finite_labels=True,nonempty_graphs=True,batching_complete=True,
                                 splits_complete=True,indices_disjoint=True,canonical_smiles_disjoint=True,
                                 scaffold_groups_disjoint=True,test_selection_leakage=False))
    save_json(OUT/'summary.json',summary)
    lock['test_evaluation_started']=True; lock['test_evaluation_completed']=True
    save_json(OUT/'evaluation_lock.json',lock)
    artifacts={str(p.relative_to(ROOT)):file_hash(p) for p in OUT.rglob('*')
               if p.is_file() and p.name!='manifest.json'}
    save_json(manifest_path,dict(sources=source_manifest(),config=CONFIG,artifacts=artifacts))
    print(f'Complete: {milestones["total_seconds"]:.1f}s; summary={OUT/"summary.json"}',flush=True)
    return summary


if __name__=='__main__':
    run_pipeline()
