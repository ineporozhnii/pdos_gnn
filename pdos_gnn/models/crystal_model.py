

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import BatchNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_add_pool as gsp
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size

class CrystalGraphConv(MessagePassing):
    def __init__(self, nbr_fea_len, l1, l2, use_cdf, atom_fea_len=64,
                 aggr: str = 'add', bias: bool = True, **kwargs):
        super(CrystalGraphConv, self).__init__(aggr=aggr, flow='target_to_source', **kwargs)
        self.use_cdf = use_cdf
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        self.BatchNorm = BatchNorm(self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self.mlp = nn.Sequential(
                nn.Linear(2*self.atom_fea_len+self.nbr_fea_len, l1), #self.atom_fea_len+2*int(self.atom_fea_len/3)
                nn.Softplus(),
                nn.Linear(l1, l2), #self.atom_fea_len+2*int(self.atom_fea_len/3), self.atom_fea_len+int(self.atom_fea_len/3
                nn.Softplus(),
                nn.Linear(l2, self.atom_fea_len), #self.atom_fea_len+int(self.atom_fea_len/3), self.atom_fea_len
                nn.Softplus() )

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None):
        """"""
        x:PairTensor = (x, x)
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out, edge_attr = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        #if not self.use_cdf:
        out = self.softplus(self.BatchNorm(out) + x[1])
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor):
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(z)

    def update(self, x, edge_attr):
        return x, edge_attr

class ProDosNet(nn.Module):

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1, l1=256, l2=256, grid=256, n_orbitals=9, use_mlp=True, use_cdf=False,):
        super(ProDosNet, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.grid = grid
        self.n_orbitals = n_orbitals
        self.embedding_1 = nn.Linear(orig_atom_fea_len, orig_atom_fea_len)
        self.embedding_2 = nn.Linear(orig_atom_fea_len, orig_atom_fea_len)
        self.convs = nn.ModuleList([CrystalGraphConv(atom_fea_len=orig_atom_fea_len,
                                    nbr_fea_len=nbr_fea_len, l1=self.l1, l2=self.l2, use_cdf=use_cdf)
                                    for _ in range(n_conv)])

        self.embed_softplus = nn.Softplus()
        self.conv_to_fc_softplus = nn.Softplus()
        self.conv_to_fc_sigmoid = nn.Sigmoid()
        self.conv_to_fc = nn.Linear(orig_atom_fea_len, n_orbitals*grid)

        self.fc_out_1 = nn.Linear(orig_atom_fea_len, 256) 
        self.fc_out_2 = nn.Linear(256, 512)
        self.fc_out_3 = nn.Linear(512, n_orbitals*grid)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.2)


    def forward(self, node_fea, edge_index, edge_attr, batch, atoms_batch, use_cdf=False, train_on_pdos=False): 
        node_fea = self.embedding_1(node_fea)
        node_fea = self.embed_softplus(node_fea)
        node_fea = self.embedding_2(node_fea)
        node_fea = self.embed_softplus(node_fea)

        for i, conv_func in enumerate(self.convs):
            node_fea = conv_func(x=node_fea, edge_index=edge_index, edge_attr=edge_attr)

        if not use_cdf and train_on_pdos:
            pdos = self.conv_to_fc_sigmoid(self.conv_to_fc(node_fea))
        else:
            node_fea = self.conv_to_fc_softplus(self.fc_out_1(node_fea))
            node_fea = self.dropout_1(node_fea)
            node_fea = self.conv_to_fc_softplus(self.fc_out_2(node_fea))
            node_fea = self.dropout_2(node_fea)
            pdos = self.conv_to_fc_sigmoid(self.fc_out_3(node_fea))
        
        dos = gsp(pdos, batch)
        dos = torch.split(dos, self.grid, 1)
        dos = torch.stack(dos, 1)
        dos = torch.sum(dos, 1)
        pdos = pdos.view(pdos.shape[0]*self.n_orbitals, self.grid)
        atomic_dos = gsp(pdos, atoms_batch)
    
        return pdos, atomic_dos, dos
