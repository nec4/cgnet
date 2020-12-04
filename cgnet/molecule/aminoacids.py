# Author: Brooke Husic


import numpy as np
import warnings

# These radii and masses were obtained from the following repository:
# https://github.com/ZiZ1/model_builder/blob/master/models/mappings/atom_types.py

# The radii were calculated by assuming a sphere and solving for the radius
# using the molar volumes at 25 Celcius reported in Table 6, column 1, of:
# Haeckel, M., Hinz, H,-J., Hedwig, G. (1999). Partial molar volumes of
# proteins: amino acid side-chain contributions derived from the partial
# molar volumes of some tripeptides over the temperature range 10-90 C.
# Biophysical Chemistry. https://doi.org/10.1016/S0301-4622(99)00104-0

# radii are reported in NANOMETERS
RESIDUE_RADII = {
                'ALA': 0.1845, 'ARG': 0.3134,
                'ASN': 0.2478, 'ASP': 0.2335,
                'CYS': 0.2276, 'GLN': 0.2733,
                'GLU': 0.2639, 'GLY': 0.1,
                'HIS': 0.2836, 'ILE': 0.2890,
                'LEU': 0.2887, 'LYS': 0.2938,
                'MET': 0.2916, 'PHE': 0.3140,
                'PRO': 0.2419, 'SER': 0.1936,
                'THR': 0.2376, 'TRP': 0.3422,
                'TYR': 0.3169, 'VAL': 0.2620,
                'CA' : 0.17, 'C' : 0.17,
                "N" : 0.1625,
                'H' : 0.1,  "O" : 0.15
                }

# masses are reported in AMUS
RESIDUE_MASSES = {
                'ALA':   89.0935, 'ARG':  174.2017,
                'ASN':  132.1184, 'ASP':  133.1032,
                'CYS':  121.1590, 'GLN':  146.1451,
                'GLU':  147.1299, 'GLY':   75.0669,
                'HIS':  155.1552, 'ILE':  131.1736,
                'LEU':  131.1736, 'LYS':  146.1882,
                'MET':  149.2124, 'PHE':  165.1900,
                'PRO':  115.1310, 'SER':  105.0930,
                'THR':  119.1197, 'TRP':  204.2262,
                'TYR':  181.1894, 'VAL':  117.1469,
                'CA' : 12.00, "H" : 1.00,
                'N' : 14.00, 'O' : 16.00
                }


def calculate_hard_sphere_minima(bead_pairs, radii_dictionary,
                                 units='Angstroms', prefactor=0.7,
                                 prefactor_ignore=None):
    """This function uses amino acid radii to calculate a minimum contact
    distance between beads in a CGMolecule given a dictionary of input radii.

    Parameters
    ----------
    bead_pairs : list of two-element tuples
        Each tuple contains the two atom labels in the coarse-grained
        representation for which a mininum distance should be calculated.
    radii_dictionary : dictionary
        Dictionary that encodes bead type radii. The keys are integers that
        are found in bead_pairs, while the corresponding values are radii of
        those bead types
    prefactor : float (default=0.7)
        Factor by which each atomic radii should be multiplied.
        The default of 0.7 is inspired by reference [1].
    prefactor_ignore: list (default=None)
        List of specific bead types for which no prefactor will be applied. One
        might use this, for example, to preserve the radii of non CG backbone
        beads.

    Returns
    -------
    hard_sphere_minima : list of floats
        Each element contains the minimum hard sphere distance corresponding
        to the same index in the input list of bead_pairs

    References
    ----------
    [1] Cheung, M. S., Finke, J. M., Callahan, B., Onuchic, J. N. (2003).
        Exploring the interplay between topology and secondary structure
        formation in the protein folding problem. J. Phys. Chem. B.
        https://doi.org/10.1021/jp034441r

    """
    if prefactor_ignore is None:
        prefactor_ignore = []
    prefactors = []
    for bead_pair in bead_pairs:
        b1, b2 = bead_pair
        if b1 in prefactor_ignore:
            p1 = 1.0
        else:
            p1 = prefactor

        if b2 in prefactor_ignore:
            p2 = 1.0
        else:
            p2 = prefactor
        prefactors.append((p1, p2))

    hard_sphere_minima = np.array(
                    [(p1*radii_dictionary[b1] +
                    p2*radii_dictionary[b2])
                    for (b1, b2), (p1, p2) in zip(bead_pairs, prefactors)]
                    )

    hard_sphere_minima = [np.round(dist, 4) for dist in hard_sphere_minima]

    return hard_sphere_minima
