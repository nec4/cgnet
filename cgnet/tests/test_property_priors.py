# Author: Nick Charron

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from cgnet.network import (CGnet, ForceLoss, RepulsionLayer,
                           HarmonicLayer, ZscoreLayer, Simulation)
from cgnet.feature import (GeometryStatistics, GeometryFeature,
                           LinearLayer, FeatureCombiner, SchnetFeature,
                           CGBeadEmbedding, GaussianRBF, ResidueStatistics)


# The following tests are used to validate priors that require 
# an embedding, bead, or residue property to be passed to them
# for on-the-fly interaction calculations

# We start be creating a random length molecule
# with random residue types (up to 3)
# we also define a fix number of beads per residue (from 2 to 5)

num_res = np.random.randint(low=5, high=15)
beads_per_res = np.random.randint(low=2, high=5)
residue_types = np.random.randint(low=1, high=4, size=num_res)

# Next, we create some random coordinates for a random number
# of frames

n_frames = np.random.randint(low=10, high=25)
coords = np.random.randn(n_frames, num_res * beads_per_res, 3)

# Here, we define a random residue encodings for each
# example

residue_encodings = []
for _ in range(n_frames):
    example_residue_encodings = np.repeat(residue_types, beads_per_res)
    residue_encodings.append(example_residue_encodings)
residue_encodings = np.array(residue_encodings)

# Lastly, we create a GeometryFeature layer to process distances 
# and angles for use in testing the prior layers below

# for now, we artifically set the backbone indices to easily get the
# sequential angles
backbone_inds = np.arange(num_res * beads_per_res)

stats = GeometryStatistics(coords, backbone_inds=backbone_inds,
                           get_all_distances=True,
                           get_backbone_angles=True,
                           adjacent_backbone_bonds=True)

geom_layer = GeometryFeature(feature_tuples=stats.feature_tuples,
                             n_beads=num_res * beads_per_res)

# Here, we use ResidueStatistics to get some statistics for the
# bonds and angles that can be used for the prior layers below

res_stats = ResidueStatistics(encoded_residues=residue_encodings,
                              beads_per_residue=beads_per_res,
                              num_res=num_res)

def test_residue_statistics():
    # Tests to make sure that ResidueStatistics is able to accurately
    # produce bead encodings for beads_per_residue=int

    single_residue = np.arange(beads_per_res)
    manual_beads_per_residue = np.tile(single_residue, num_res)
    np.testing.assert_equal(manual_beads_per_residue,
                            res_stats.beads_per_residue)


def test_residue_stats_get_features():
    # Tests to make sure that the get_residue_features method of
    # ResidueStatistics returns the proper features involved in
    # a specified encoding

    # First, we choose a random set of encodings, that are of the form
    # (*local_beads, *residue_types)

    # for now, we just use distances 
    num_encodings = np.random.randint(low=5, high=10)
    out_encodings = []
    bead_idx = np.random.choice(np.arange(len(stats.descriptions['Distances'])),
                             num_encodings, replace=False)
    beads = [ bead_tuple for num, bead_tuple in
              enumerate(stats.descriptions['Distances']) if num in bead_idx]
    manual_residue_types = np.random.choice(residue_types,
                                     size=(num_encodings, 2))

    for bead_pair, res_types in zip(beads, manual_residue_types):
        out_encodings.append((*bead_pair, *res_types))

    single_residue = np.arange(beads_per_res)
    manual_beads_per_residue = np.tile(single_residue, num_res)

    # Next, we perform a manual extraction of the features from the
    # full distance data

    selected_idx = stats.return_indices(beads)
    distance_subset = stats.distances[:, selected_idx]
    manual_residue_beads = residue_encodings[:, np.array(beads)]
    manual_local_beads = np.repeat(manual_beads_per_residue[None, np.array(beads)],
                            len(manual_residue_beads), 0)
    manual_final_labels = np.concatenate((manual_local_beads,
                                         manual_residue_beads), axis=2)

    manual_residue_features = {}
    for encoding in out_encodings:
        manual_residue_features[encoding] = []

    for encoding in out_encodings:
        for example, labels in zip(distance_subset, manual_final_labels):
            for num, label in enumerate(labels):
                if tuple(label) == encoding:
                    manual_residue_features[encoding].append(example[num])

    for encoding in out_encodings:
        manual_residue_features[encoding] = np.array(manual_residue_features[encoding])

    # Next, we use the residue statistics method to compare to the manual_residue
    # features dictionary above

    residue_features = res_stats.get_residue_features(distance_subset, beads,
                                                      out_encodings)

    assert set(manual_residue_features.keys()) == set(residue_features.keys())
    for key in manual_residue_features.keys():
        print(manual_residue_features[key])
        print(residue_features[key])
        np.testing.assert_equal(manual_residue_features[key],
                                residue_features[key])

"""
def test_bonds_residue_harmonic_layer():
    # Tests to make sure that the residue harmonic layer
    # correctly calculates the harmonic contributions for
    # bonds from examples with varying residue encodings

"""

