import os
import json
import torch
import warnings
import numpy as np
from typing import Union, List
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.core import Orbital
from pymatgen.electronic_structure.dos import CompleteDos
import matplotlib.pyplot as plt

class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2/self.var**2)
    

class CrystalGraphPDOS():
    """
        Class to construct graph representation of materials with Projected Density of States as target property
    """
    def __init__(self, 
                 dos_dir: str, 
                 cif_dir: str, 
                 radius: int = 8, 
                 max_num_nbr: int = 12, 
                 sigma: float = 0.3, 
                 bound_low: float = -20.0,
                 bound_high: float = 10.0,
                 grid: int = 256, 
                 max_element: int = 83, 
                 n_orbitals: int = 9,
                 gauss_distance_min: float = 0.0,
                 gauss_distance_max: float = 8.0,
                 gauss_distance_step: float = 0.2,
                 norm_dos: bool = False,
                 norm_pdos: bool = False
                 ):
        """
            Holds initial parametes
            -----------------------
            - dos_dir:      Path to directory with Density of States json files
            - cif_dir:      Path to directory with material cif files
            - radius:       Maximum radius for neighbors search during graph construction
            - max_num_nbr:  Maximum number of neighbours in the graph
            - sigma:        Density of States broadening parameter
            - bound_low:    Lower boundary of DOS energy window (eV)
            - bound_high:   Higher boundary of DOS energy window (eV)
            - grid:         Number of density grid points
            - max_element:  Maximum element number in the crystal
            - n_orbitals:   Number of orbitals per atom (default=9; 1s, 3p, 5d orbitals)
            - norm_dos:     Normalize total DOS if True
            - norm_pdos:    Normalize orbital PDOS if True
        """

        self.dos_dir = dos_dir
        self.cif_dir = cif_dir
        self.radius = radius
        self.sigma = sigma
        self.bound_low = bound_low
        self.bound_high = bound_high
        self.grid = grid
        self.max_num_nbr = max_num_nbr
        self.max_element = max_element
        self.n_orbitals = n_orbitals
        self.gauss_distance_max = gauss_distance_max
        self.norm_dos = norm_dos
        self.norm_pdos = norm_pdos

        print(
        f"""\t------------------------------------------------
        |        Data Preprocessing Parameters         |
        ------------------------------------------------
            - dos_dir:      {self.dos_dir}
            - cif_dir:      {self.cif_dir}
            - radius:       {self.radius}
            - max_num_nbr:  {self.max_num_nbr}
            - sigma:        {self.sigma}
            - bound_low:    {self.bound_low}
            - bound_high:   {self.bound_high}
            - grid:         {self.grid}
            - max_element:  {self.max_element}
            - n_orbitals:   {self.n_orbitals}
            - norm_pdos:    {self.norm_pdos}
        ------------------------------------------------
        """)
        


        self.gdf = GaussianDistance(dmin=gauss_distance_min, dmax=self.gauss_distance_max, step=gauss_distance_step)
        self.orbital_name_list = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2"]
        self.spin=Spin(1)

        features_dir = os.path.join(os.path.dirname(__file__), '../features')
        with open(f"{features_dir}/atom_features_one_hot.json", "r") as element_one_hot_features_file:
                    self.element_one_hot_features = json.load(element_one_hot_features_file)
        with open(f"{features_dir}/orbit_radius_fea_scaled.json", "r") as orbit_radius_fea_file:
                    self.orbit_radius_fea = json.load(orbit_radius_fea_file)
        with open(f"{features_dir}/n_electrons.json", "r") as n_electrons_file:
                    self.n_electrons_dict = json.load(n_electrons_file)



    def _get_normalization_coefficient(self, energy, density, e_min, e_max, complete_dos=None, orbital=False):
        """
            normalize_total_dos function calculates normalization coefficient for Density of States
            ---------------------------------------------------------------------------------------
            Input:
                - complete_dos:     Density of States object
                - energy:           DOS energies
                - density:          DOS densities
                - e_min:            Lower boundary of integration
                - e_max:            Higher boundary of integration
            Returns:
                - norm_coefficient: Normalization coefficient for DOS
        """

        energy_norm = energy[(energy < e_max) & (energy > e_min)]
        density_norm = density[(energy < e_max) & (energy > e_min)]
        area = np.trapz(density_norm, x=energy_norm)
        if orbital: 
            if area == 0:
                return 1
            else:
                norm_coefficient = 2 / area
            return norm_coefficient
        else:
            n_electrons = 0
            for site in complete_dos.structure:
                n_electrons += int(self.n_electrons_dict[site.specie.symbol])
            norm_coefficient = n_electrons/area
            return norm_coefficient

    def _preproc_dos(self, energies, densities, extend_range=10, sigma=0.2, smear=True):
        """
            _preproc_dos function extends DOS energy window and applies broadening (smearing)
            --------------------------------------------------------------------------------
        """
        if sigma == 0.0:
            smear = False

        diff = [energies[i + 1] - energies[i]
                for i in range(len(energies) - 1)]
        avgdiff = sum(diff) / len(diff)

        min_energy = energies[0]
        max_energy = energies[-1]+avgdiff

        try:
            energies_low = np.linspace(min_energy-extend_range, min_energy, int(extend_range/avgdiff), endpoint=False)
        except Exception as e:
            print(energies)
            print(avgdiff)
            print(diff)
        energies_high = np.linspace(max_energy, max_energy+extend_range, int(extend_range/avgdiff), endpoint=True)
        densities_low = np.zeros(len(energies_low))
        densities_high = np.zeros(len(energies_high))
        energies_extend = np.concatenate((energies_low, energies, energies_high), axis=None)
        densities_extend = np.concatenate((densities_low, densities, densities_high), axis=None)

        if smear:
            smeared_dens = gaussian_filter1d(densities_extend, sigma/avgdiff)
            return energies_extend, smeared_dens
    
        return energies_extend, densities_extend


    def _get_graph(self, crystal: Structure, material_id: str) -> Union[LongTensor, Tensor, Tensor, List, List]:
        """
            Generates crystal graph from cif pymatgen structure
            Input:
                - crystal:      Pymatgen structure
                - material_id:  Materials Project id
            Output:
                - bond_index:                   Tensor of edge indices that encode conections in graph
                - bond_attr:                    Tensor of edge features that 
                - distances:                    Tensor of distances between atoms
                - crystal_orbital_name_list:    List of orbital names (s, px, py, etc.) 
                - crystal_orbital_index_list:   List of orbital indices corresponding to orbital types
        """
        all_nbrs = crystal.get_all_neighbors(self.gauss_distance_max, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        bond_index = [[],[]]
        bond_attr = []
        crystal_orbital_name_list = []
        crystal_orbital_index_list = []
  
        for i, nbrs in enumerate(all_nbrs):
                # Add orbital names and indices for particular site i to lists for crystal data
                crystal_orbital_name_list.append(self.orbital_name_list)
                crystal_orbital_index_list.append(list(range(0, 9)))

                if len(nbrs) < self.max_num_nbr:
                        warnings.warn('In {} found less neighbours than set maximum number of neighbors (n_max = {}). Material will be skipped.'.format(material_id, self.max_num_nbr), stacklevel=2)
                        return None
                else:
                        # Create crystal graph (atomic) bond indices and bond attributes (distances)
                        bond_index[0] += [i]*self.max_num_nbr
                        bond_index[1].extend(list(map(lambda x: x[2], nbrs[:self.max_num_nbr])))
                        # bond_attr.extend(np.asarray(list(map(lambda x: x[1], nbrs[:self.max_num_nbr]))))
                        # Create bond features as [x_distance, y_distance, z_distance, distance] vector
                        nbrs_coords = np.asarray(list(map(lambda x: x[0].coords, nbrs[:self.max_num_nbr])))
                        nbrs_coords_distances = nbrs_coords - crystal.sites[i].coords
                        nbrs_distances = np.asarray(list(map(lambda x: [x[1]], nbrs[:self.max_num_nbr])))
                        bonds = np.append(nbrs_coords_distances, nbrs_distances, 1)
                        bond_attr.extend(bonds)

        bond_index, bond_attr = np.array(bond_index), np.array(bond_attr)
        distances = torch.unsqueeze(Tensor(bond_attr), 1)
        #bond_attr = self.gdf.expand(bond_attr)
        bond_attr = torch.unsqueeze(Tensor(bond_attr), 1)
        bond_index = LongTensor(bond_index)
        return bond_index, bond_attr, distances, crystal_orbital_name_list, crystal_orbital_index_list


    def _get_node_features(self, material_file_name, crystal, crystal_orbital_name_list, crystal_orbital_index_list, efermi=None, complete_dos=None, total_dos_norm_coef=None, for_training=False):
        sites = []
        elements = []
        atom_fea = []
        target_pdos = []
        target_pdos_cdf = []
        orbital_types = []
        orbital_max_density_list = []

        # Iterate over orbital energies for each site
        for k, site_orbitals in enumerate(crystal_orbital_index_list):
            if complete_dos is not None:
                try:
                    site = complete_dos.structure[k]
                except Exception as e:
                     print(e)
                     print(crystal)
                     print("-----------------")
                     print(complete_dos.structure)
                     print(material_file_name)
                     print("-----------------")
            else:
                site = crystal.sites[k]
            site_one_hot_features = np.asarray(self.element_one_hot_features[str(crystal.sites[k].specie.number)])
            site_orbit_radius_features = np.asarray(self.orbit_radius_fea[str(crystal.sites[k].specie.number)])
            site_features = np.concatenate((site_one_hot_features, site_orbit_radius_features))
            atom_fea.append(site_features)
            for i, _ in enumerate(site_orbitals):
                orbital_types.append(crystal_orbital_name_list[k][i])
                elements.append(str(crystal.sites[k].specie))
                sites.append(str(k))

                if for_training:
                    orbital = Orbital(crystal_orbital_index_list[k][i])
                    orbital_dos = complete_dos.get_site_orbital_dos(site, orbital)
                    
                    if np.isnan(orbital_dos.energies).any():
                        warnings.warn("Structure {} contains NaN in orbital_dos energies and will be skipped".format(material_file_name), stacklevel=2)
                        return None
                         
                    energies_ext, densities_ext = self._preproc_dos(orbital_dos.energies, orbital_dos.get_densities(spin=self.spin), sigma=self.sigma, extend_range=30, smear=True)
                    if self.norm_pdos:
                        orbital_norm_coefficient = self._get_normalization_coefficient(energy=orbital_dos.energies, density=orbital_dos.get_densities(spin=self.spin), e_min=np.min(orbital_dos.energies), e_max=np.max(orbital_dos.energies), orbital=True)
                    else: 
                        orbital_norm_coefficient = 1.0

                    densities_ext = densities_ext * orbital_norm_coefficient
                    densities_ext_cdf = np.cumsum(densities_ext) * np.mean(np.diff(energies_ext))

                    interp_funtion = interp1d(energies_ext, densities_ext)
                    interp_funtion_cdf = interp1d(energies_ext, densities_ext_cdf)
                    shifted_energies = np.linspace(self.bound_low+efermi, self.bound_high+efermi, self.grid)
                    e_diff = np.mean(np.diff(shifted_energies)) 

                    target_orbital_density = interp_funtion(shifted_energies)
                    target_orbital_density_cdf = interp_funtion_cdf(shifted_energies)

                    target_pdos.append(target_orbital_density)
                    target_pdos_cdf.append(target_orbital_density_cdf)
                    orbital_max_density_list.append(np.max(target_orbital_density))

        atom_fea = np.array(atom_fea)
        atom_fea = Tensor(atom_fea)

        if for_training:
            target_pdos = np.array(target_pdos)
            target_pdos = Tensor(target_pdos)
            target_pdos_cdf = np.array(target_pdos_cdf)
            target_pdos_cdf = Tensor(target_pdos_cdf)
            target_dos = torch.sum(target_pdos, dim=0)
            target_dos = torch.unsqueeze(target_dos, 0)
            target_dos_cdf = torch.sum(target_pdos_cdf, dim=0)
            target_dos_cdf = torch.unsqueeze(target_dos_cdf, 0)
            e_diff = Tensor([e_diff])
            plt.title(material_file_name)
            plt.plot(np.linspace(-20, 10, 256), target_dos.detach().numpy()[0])
            plt.show()
            return
            return atom_fea, target_dos, target_pdos, target_dos_cdf, target_pdos_cdf, elements, sites, orbital_types, orbital_max_density_list, e_diff

        return atom_fea, elements, sites, orbital_types


    def get_crystal_pdos_graph(self, material_file: str) -> Data:
        """
            This method loads material cif and DOS files, asserts that material fits the dataset constraints, and returns a crystal graph
            Input: 
                - material_file: path to material's cif file
            Output:
                - crystal_graph: Pytorch Geometric Data structure containing material crystal graph with node features, 
                                 edge features, and PDOS as target for training 
            
        """
        material_file_name = os.path.split(material_file)[1].split(".")[0]
        crystal = Structure.from_file(material_file)

        # Check if material contains elements with higher atomic number than specified by parameter max_element
        atomic_numbers = []
        for site in crystal:
            atomic_numbers.append(site.specie.number)
        heaviest_element = np.max(atomic_numbers)
        if heaviest_element > self.max_element:
            warnings.warn("Structure {} contains element with atomic number = {} and will be skipped. Max atomic number = {}.".format(material_file_name, heaviest_element, self.max_element), stacklevel=2)
            return None

        with open(os.path.join(self.dos_dir, material_file_name+"_dos.json"), "r") as material_dos_file:
            material_dos_file = json.load(material_dos_file)

        # Check if material has spin polarized PDOS calculation
        if len(material_dos_file["densities"]) > 1: 
            warnings.warn("Structure {} contains spin-polarized PDOS calculation and will be skipped".format(material_file_name), stacklevel=2)
            return None

        # Check that material structure in cif file has same number of atoms as structure in complete_dos
        complete_dos = CompleteDos.from_dict(material_dos_file)
        if not crystal.matches(complete_dos.structure) or len(complete_dos.structure) != len(crystal):
            warnings.warn("Structure {} does not match structure in PDOS file and will be skipped".format(material_file_name), stacklevel=2)
            return None

        if self.norm_dos:
            # Check DOS normalization 
            norm_coefficient = self._get_normalization_coefficient(energy=complete_dos.energies, density=complete_dos.get_densities(spin=self.spin), e_min=np.min(complete_dos.energies), e_max=np.max(complete_dos.efermi), complete_dos=complete_dos)
        else:
            norm_coefficient = 1.0

        # Get crystal graph
        graph_parameters = self._get_graph(crystal, material_file_name)

        if graph_parameters is None:
            return None
        
        bond_index, bond_attr, distances, crystal_orbital_name_list, crystal_orbital_index_list = graph_parameters
            
        # Get features and targets
        node_features = self._get_node_features(material_file_name, crystal, crystal_orbital_name_list, crystal_orbital_index_list, efermi=material_dos_file["efermi"], complete_dos=complete_dos, total_dos_norm_coef=norm_coefficient, for_training=True)
        if node_features is None:
            return None
        
        atom_fea, target_dos, target_pdos, target_dos_cdf, target_pdos_cdf, elements, sites, orbital_types, orbital_max_density_list, e_diff = node_features

        crystal_graph = Data(x=atom_fea, edge_index=bond_index, edge_attr=bond_attr, dos=target_dos, pdos=target_pdos, dos_cdf=target_dos_cdf, pdos_cdf=target_pdos_cdf, material_id=material_file_name, elements=elements, sites=sites, orbital_types=orbital_types, distances=distances, orbital_max_density=np.max(orbital_max_density_list), e_diff=e_diff)
        return crystal_graph


    def get_crystal_pdos_graph_pred(self, material_file: str) -> Data:
        """
            Constructs and returns crystal graph without target PDOS values
            Input:
                - material_file: cif file of material
                - structure:     pymatgen structure (if provided, will be used instead of cif file) 
            Output:
                - crystal_graph: Pytorch Geometric Data structure containing material crystal graph with node features and  
                                 edge features for prediction when target PDOS values are unknown
        """

        material_file_name = os.path.split(material_file)[1].split(".")[0]
        crystal = Structure.from_file(material_file)
        graph_parameters = self._get_graph(crystal, material_file_name)
        if graph_parameters is None:
            return None 
        bond_index, bond_attr, distances, crystal_orbital_name_list, crystal_orbital_index_list = graph_parameters
        node_features = self._get_node_features(material_file_name, crystal, crystal_orbital_name_list, crystal_orbital_index_list, for_training=False)
        if node_features is not None:
            atom_fea, elements, sites, orbital_types = node_features
        
        n_atoms = int(atom_fea.shape[0])
        atoms = []
        for i in range(n_atoms):
            atoms.extend([i]*self.n_orbitals)
        atoms_batch = LongTensor(np.array(atoms))

        crystal_graph = Data(x=atom_fea, edge_index=bond_index, edge_attr=bond_attr, material_id=material_file_name, elements=elements, sites=sites, orbital_types=orbital_types, atoms_batch=atoms_batch)
        return crystal_graph