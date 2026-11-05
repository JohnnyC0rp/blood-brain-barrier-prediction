"""Checks for molecular invariance, bond use, and graph batching integrity."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.utils import from_smiles

from molecular_utils import MolecularGNN, check_batches, seed_everything


class MolecularChecks(unittest.TestCase):
    def setUp(self):
        seed_everything(2026)
        torch.set_num_threads(2)
        self.model = MolecularGNN().eval()
        self.graphs = [from_smiles(s) for s in ('CCO', 'c1ccccc1', 'C')]
        for index, graph in enumerate(self.graphs):
            graph.y = torch.tensor([[float(index % 2)]])
            graph.record_id = torch.tensor([index])

    def test_prediction_is_per_graph_and_independent_of_batch_neighbors(self):
        with torch.no_grad():
            together = self.model(Batch.from_data_list(self.graphs))
            alone = torch.cat([self.model(Batch.from_data_list([g])) for g in self.graphs])
        self.assertEqual(tuple(together.shape), (3,))
        self.assertTrue(torch.isfinite(together).all())
        torch.testing.assert_close(together, alone, atol=1e-5, rtol=1e-5)

    def test_relabeling_atoms_preserves_graph_prediction(self):
        graph = self.graphs[0]
        order = torch.tensor([2, 0, 1])
        inverse = torch.argsort(order)
        reordered = graph.clone()
        reordered.x = graph.x[order]
        reordered.edge_index = inverse[graph.edge_index]
        with torch.no_grad():
            expected = self.model(Batch.from_data_list([graph]))
            actual = self.model(Batch.from_data_list([reordered]))
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_bond_categories_change_messages(self):
        graph = self.graphs[0]
        changed = graph.clone()
        changed.edge_attr[:, 0] = 2
        with torch.no_grad():
            expected = self.model(Batch.from_data_list([graph]))
            actual = self.model(Batch.from_data_list([changed]))
        self.assertGreater(float((actual - expected).abs().max()), 1e-6)

    def test_batch_audit_includes_last_partial_batch_and_detects_lost_graph(self):
        self.assertEqual(check_batches(self.graphs, batch_size=2)['total_graphs_verified'], 3)
        self.graphs[-1].record_id = torch.tensor([0])
        with self.assertRaises(AssertionError):
            check_batches(self.graphs, batch_size=2)

    def test_structure_figures_save_with_jupyter_renderer_installed(self):
        from rdkit.Chem.Draw import IPythonConsole
        import run_experiments
        IPythonConsole.InstallIPythonRenderer()
        smiles = ['CCc1ccccc1', 'CCCc1ccccc1', 'CCCCc1ccccc1',
                  'COc1ccccc1', 'CCOc1ccccc1', 'CCCOc1ccccc1']
        graphs = [from_smiles(s) for s in smiles]
        molecules = [Chem.MolFromSmiles(s) for s in smiles]
        for i, graph in enumerate(graphs):
            graph.y = torch.tensor([[float(i % 2)]])
        frame = pd.DataFrame([dict(record_id=i, smiles=s, label=i % 2,
                                   atoms=m.GetNumAtoms(), bonds=m.GetNumBonds())
                              for i, (s, m) in enumerate(zip(smiles, molecules))])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch.object(run_experiments, 'FIG', folder), patch.object(run_experiments, 'TABLES', folder):
                run_experiments.inspect_figures(graphs, molecules, frame)
            self.assertGreater((folder / 'molecule_examples.png').stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
