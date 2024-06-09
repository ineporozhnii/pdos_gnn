import os
import csv
import torch
import tarfile
import functools
import numpy as np
import pandas as pd
from torch_scatter import scatter
from torch.utils.data import Dataset


class MaterialData(Dataset):
    def __init__(self, root=None, id_file=None, data_file=None, n_orbitals=3, train_on_atomic_dos=False):
        self.id_file = id_file
        self.data_file = data_file
        self.n_orbitals = n_orbitals
        self.train_on_atomic_dos = train_on_atomic_dos

        assert os.path.exists(self.id_file),  f'{self.id_file} does not exist!'
        self.ids = pd.read_csv(self.id_file).iloc[:, 0].tolist()
 
        assert os.path.exists(self.data_file),  f'{self.data_file} does not exist!'
        print(" Opening tar ...")
        self.graphs_data = tarfile.open(self.data_file)
        print(" Done!")
        

    def __len__(self):
        length = len(self.ids)
        return length


    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_id = self.ids[idx]
        
        material_graph = torch.load(self.graphs_data.extractfile(os.path.basename(self.data_file).removesuffix('.tar')+f'/{cif_id}_crystal_graph_pdos.pt'))
        material_graph.edge_attr = torch.squeeze(material_graph.edge_attr, 1)

        target_pdos = material_graph.pdos
        n_atoms = int(target_pdos.shape[0]/self.n_orbitals)
        atoms = []
        e_diff = torch.mean(material_graph.e_diff)
        
        for i in range(n_atoms):
            atoms.extend([i]*self.n_orbitals)
        atoms_batch = torch.LongTensor(np.array(atoms))
        size = int(atoms_batch.max().item() + 1)
        atomic_dos = scatter(target_pdos, atoms_batch, dim=0, dim_size=size, reduce='add')
        atomic_dos_cdf = torch.cumsum(atomic_dos, dim=1)*e_diff
        material_graph.atomic_dos = atomic_dos
        material_graph.atomic_dos_cdf = atomic_dos_cdf
        material_graph.atoms_batch = atoms_batch
        material_graph.pdos_cdf = torch.cumsum(material_graph.pdos, dim=1)*e_diff
        material_graph.dos_cdf = torch.cumsum(material_graph.dos, dim=1)*e_diff

        return material_graph
