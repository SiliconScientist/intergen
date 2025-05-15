from ase.db import connect
from ase.atoms import Atoms
from ase.build import fcc111, hcp0001
import copy
import pandas as pd
from timeit import default_timer as timer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np

def id_unique_sites(
    atoms: Atoms,
    indices: list[int],
    masked_element: str,
    symprec: float = 0.05,
    angle_tolerance: float = 1.0,
    converter: AseAtomsAdaptor = AseAtomsAdaptor(),
) -> list[int]:
    structure = converter.get_structure(atoms)
    symmetry_object = SpacegroupAnalyzer(
        structure, symprec=symprec, angle_tolerance=angle_tolerance
    )
    masked_sites = np.where(np.array(atoms.get_chemical_symbols()) == masked_element)[0]
    equivalent_atoms = symmetry_object.get_symmetry_dataset()["equivalent_atoms"]
    unique_indices = np.unique([equivalent_atoms[i] for i in indices])
    unique_unmasked_indices = np.setdiff1d(unique_indices, masked_sites)
    return unique_unmasked_indices


def swap_element(atoms: Atoms, index: int, element: str) -> Atoms:
    new_atoms = copy.deepcopy(atoms)
    new_symbols = atoms.get_chemical_symbols()
    new_symbols[index] = element
    new_atoms.set_chemical_symbols(new_symbols)
    return new_atoms


def enumerate_unique_substitutions(
    atoms: Atoms,
    indices: list[int],
    element: str,
) -> list[Atoms]:
    """
    Generate a list of structures with the given element substituted at each
    symmetry-unique site from the provided indices.

    Parameters:
        atoms (Atoms): Original atomic structure.
        indices (list[int]): Candidate site indices for substitution.
        element (str): Element symbol to substitute in.

    Returns:
        list[Atoms]: List of new structures with one substitution each.
    """
    unique_site_indices = id_unique_sites(
        atoms=atoms,
        indices=indices,
        element=element,
    )
    atoms_list = []
    for index in unique_site_indices:
        new_atoms = swap_element(atoms=atoms, index=index, element=element)
        atoms_list.append(new_atoms)
    return atoms_list


def delete_duplicate_atoms(
    atoms_list,
    converter: AseAtomsAdaptor = AseAtomsAdaptor(),
    matcher: StructureMatcher = StructureMatcher(),
) -> list[Atoms]:
    """Removes duplicates ASE Atoms objects based on PyMatGen structure matching."""
    structures = [converter.get_structure(atoms) for atoms in atoms_list]
    unique_atoms = []
    for i, candidate in enumerate(structures):
        is_duplicate = any(
            matcher.fit(candidate, comparison) for comparison in structures[i + 1 :]
        )
        if not is_duplicate:
            unique_atoms.append(atoms_list[i])
    return unique_atoms

def main():
    lattice_constant_df = pd.read_csv(
        "./LatticeConstantsPure.csv",
        index_col=0,
        skiprows=1,
    )

    hcpHostList = [
        "Ti",
        "Ru",
        "Co",
        "Sc",
        "Hf",
        "Zr",
        "Re",
        "Os",
    ]  #'Ti', 'Ru', 'Co', 'Sc', 'Hf', 'Zr', 'Re', 'Os'
    fccHostList = [
        "Cu",
        "Ag",
        "Au",
        "Ni",
        "Pt",
        "Pd",
        "Rh",
        "Ir",
    ]  #'Cu', 'Ag', 'Au', 'Ni', 'Pt', 'Pd', 'Rh', 'Ir'

    hostSlabs = []
    for host in fccHostList:
        slab = fcc111(
            host,
            size=(3, 3, 4),
            vacuum=10.0,
            a=lattice_constant_df.loc[host, "FCC_LatticeConstant_PW91_1"],
        )[::-1]
        hostSlabs.append(slab)

    for host in hcpHostList:
        slab = hcp0001(
            host,
            size=(3, 3, 4),
            vacuum=10.0,
            a=lattice_constant_df.loc[host, "HCP_LatticeConstant_PW91_1"],
            c=lattice_constant_df.loc[host, "HCP_LatticeConstant_PW91_1"]
            * lattice_constant_df.loc[host, "HCP_c_over_a_PW91_1"],
        )[::-1]
        hostSlabs.append(slab)

    dopants = [
        "Cu",
        "Ag",
        "Au",
        "Ni",
        "Pt",
        "Pd",
        "Co",
        "Rh",
        "Ir",
        "Fe",
        "Ru",
        "Os",
        "Mn",
        "Re",
        "Cr",
        "Mo",
        "W",
        "V",
        "Ta",
        "Ti",
        "Zr",
        "Hf",
        "Sc",
    ]

    dopingLocations = list(range(0, 18))  # Make 18

    start = timer()

    configDB = connect(
        "configs_TwoLayers_UpToThreeDopants.db"
    )  # configs_ForOandOH_Pure, configs_ForOandOH, configs_ForOandOH_2Dopant, configs_ForOandOH_1Dopant
    writeToDB = True
    totNumDope = 3  # =len(surfaceAtoms)

    aseConverter = AseAtomsAdaptor()
    allSlabs = []
    surfaceAtomsList = []
    previousList = []  # May be unnecessary
    for hostSlab in hostSlabs:
        if writeToDB:
            configDB.write(hostSlab)
        allSlabs.append(hostSlab)
        for dopant in dopants:
            # Single atom:
            previousList = enumerate_unique_substitutions(
                atoms=hostSlab, indices=dopingLocations, element=dopant
            )
            for atoms in previousList:
                if writeToDB:
                    configDB.write(atoms)
                allSlabs.append(atoms)
            # More atoms:
            for i in range(1, totNumDope):
                tempList = []
                for atoms in previousList:
                    newList = enumerate_unique_substitutions(
                        atoms=atoms, indices=dopingLocations, element=dopant
                    )
                    #                 newList = duplicateFunction(newList) #This seems to be unnecessary--makes things take a bit longer.
                    for atoms in newList:
                        tempList.append(atoms)
                previousList = delete_duplicate_atoms(atoms_list=tempList)
                for atoms in previousList:
                    if writeToDB:
                        configDB.write(atoms)
                    allSlabs.append(atoms)
                    # TODO: delete duplicates here?

                # previousList = uniqueList
    end = timer()

    print("Time(s): ", end - start)
    print("Number of surfaces: ", len(allSlabs))


if __name__ == "__main__":
    main()
