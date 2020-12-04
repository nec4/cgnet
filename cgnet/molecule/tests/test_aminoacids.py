# Author: Brooke Husic

import numpy as np
import itertools

from cgnet.molecule import (CGMolecule, RESIDUE_RADII,
                            calculate_hard_sphere_minima)

# Here, we make a simple dictionary to denote the bead labels
# for different bead types for a variable length CA toy model

num_res = np.random.randint(low=5, high=10)

# Beads can be one of of 50 "types"
bead_labels = np.random.randint(low=1, high=50, size=(num_res,))

# Here, we generate all possible bead pairs
bead_pairs = list(itertools.combinations(np.arange(num_res), 2))
label_pairs = bead_labels[np.array(bead_pairs)]

# Here, we give a random positive radii to each bead type:
radii_dictionary = {}
for label in bead_labels:
    radii_dictionary[label] = np.random.uniform(low=0, high=10)

def test_minima_calculation_values():
    # This is a manual test of the minima calculations

    # Designate a random prefactor (i.e., scaling factor for each radius
    # in the calculation
    prefactor = np.random.uniform(0.5, 1.3)

    # Perform the manual calculation
    manual_distances = []
    for pair in label_pairs:
        rad1 = radii_dictionary[pair[0]]
        rad2 = radii_dictionary[pair[1]]
        manual_distances.append(prefactor*rad1 + prefactor*rad2)

    # Perform the automatic calculation
    distances = calculate_hard_sphere_minima(label_pairs,radii_dictionary,
                                             prefactor=prefactor)

    # The high tolerance is due to the significant figures in the
    # master list
    np.testing.assert_allclose(manual_distances, distances, rtol=1e-4)


def tests_prefactor_ignore_list():
    # This tests the 'prefactor_ignore' kwarg in 'calculate_hard_sphere_minima'
    # to ensure that selected radii are not scaled by the prefactor if they
    # are so chosen

    # Designate a random prefactor (i.e., scaling factor for each radius
    # in the calculation
    prefactor = np.random.uniform(0.5, 1.3)

    # Here, we pick the bead types that we want exempt from prefactor scaling
    num_unique_types = len(np.unique(bead_labels))
    num_exempt = np.random.randint(low=1, high=num_unique_types)
    ignore_list = np.random.choice(np.unique(bead_labels), num_exempt, replace=False)

    # Perform the manual calculation
    manual_distances = []
    for pair in label_pairs:
        if pair[0] in ignore_list:
            rad1 = radii_dictionary[pair[0]]
        else:
            rad1 = prefactor * radii_dictionary[pair[0]]
        if pair[1] in ignore_list:
            rad2 = radii_dictionary[pair[1]]
        else:
            rad2 = prefactor * radii_dictionary[pair[1]]

        manual_distances.append(rad1 + rad2)

    # Perform the automatic calculation
    distances = calculate_hard_sphere_minima(label_pairs,radii_dictionary,
                                             prefactor=prefactor,
                                             prefactor_ignore=ignore_list)

    # The high tolerance is due to the significant figures in the
    # master list
    np.testing.assert_allclose(manual_distances, distances, rtol=1e-4)

