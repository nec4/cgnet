# Author: Nick Charron

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from cgnet.network import (CGnet, ForceLoss, RepulsionLayer,
                           HarmonicLayer, ZscoreLayer, Simulation,
                           ResidueHarmonicLayer, BeadRepulsionLayer,
                           PriorForceComputer)
from cgnet.feature import (GeometryStatistics, GeometryFeature,
                           LinearLayer, FeatureCombiner, SchnetFeature,
                           CGBeadEmbedding, GaussianRBF, ResidueStatistics)
from test_feature_combiner import _get_random_schnet_feature
from itertools import combinations


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

coords = np.random.randn(n_frames, num_res * beads_per_res, 3).astype('float32')
tensor_coords = torch.tensor(coords, requires_grad=True)

# Here, we define a random residue encodings for each
# example

residue_encodings = []
for _ in range(n_frames):
    example_residue_encodings = np.repeat(residue_types, beads_per_res)
    residue_encodings.append(example_residue_encodings)
residue_encodings = np.array(residue_encodings)
tensor_residue_encodings = torch.tensor(residue_encodings)

# We also want to make embedding properties that can be used for 
# testing bead prior layers. These integer labels are NOT necessarily 
# realted to the residue labels 

embeddings = np.random.randint(low=1, high=3,
                               size=(n_frames, num_res * beads_per_res))
tensor_embeddings = torch.tensor(embeddings)

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

### Random Normal Harmonic Prior ###


num_distances = np.random.randint(low=5, high=10)
normal_bead_idx = np.random.choice(np.arange(len(stats.descriptions['Distances'])),
                         num_distances, replace=False)
normal_beads = [ bead_tuple for num, bead_tuple in
          enumerate(stats.descriptions['Distances']) if num in normal_bead_idx]

normal_selected_idx = stats.return_indices(normal_beads)
normal_distance_subset = stats.distances[:, normal_selected_idx]

# parameters for normal harmoniclayer
normal_interaction_params = []
for i in range(len(normal_beads)):
    normal_interaction_params.append({'mean' : torch.tensor(np.random.uniform(low=1, high=5)),
                                      'k' : torch.tensor(np.random.uniform(low=1, high=5))})
normal_hlayer = HarmonicLayer(normal_selected_idx, normal_interaction_params)

### Random Residue Harmonic Prior ###

residue_num_encodings = np.random.randint(low=5, high=10)
residue_bead_idx = np.random.choice(np.arange(len(stats.descriptions['Distances'])),
                         residue_num_encodings, replace=False)
residue_beads = [ bead_tuple for num, bead_tuple in
          enumerate(stats.descriptions['Distances']) if num in residue_bead_idx]
residue_bead_array = np.array(residue_beads)
manual_residue_types = np.random.choice(residue_types,
                                 size=(residue_num_encodings, 2))

residue_selected_idx = stats.return_indices(residue_beads)
residue_distance_subset = stats.distances[:, residue_selected_idx]

# parameters for residue harmoniclayer
residue_interaction_params = res_stats.get_stats(residue_distance_subset,
                                                 residue_beads)
for key, param_dict in residue_interaction_params.items():
    float_conv = {}
    float_conv['mean'] = torch.tensor(param_dict['mean']).float()
    float_conv['k'] = torch.tensor(param_dict['k']).float()
    residue_interaction_params[key] = float_conv

residue_hlayer = ResidueHarmonicLayer(residue_selected_idx,
                                      residue_interaction_params,
                                      bead_array=residue_bead_array,
                                      beads_per_residue=res_stats.beads_per_residue)

### Random Bead Prior Layer 

bead_num_encodings = np.random.randint(low=5, high=10)
bead_bead_idx = np.random.choice(np.arange(len(stats.descriptions['Distances'])),
                         bead_num_encodings, replace=False)
bead_beads = [ bead_tuple for num, bead_tuple in
          enumerate(stats.descriptions['Distances']) if num in bead_bead_idx]
bead_bead_array = np.array(bead_beads)

bead_labels = list(combinations(np.unique(embeddings), 2))
for label in np.unique(embeddings):
    bead_labels.append((label,label))

bead_selected_idx = stats.return_indices(bead_beads)
bead_distance_subset = stats.distances[:, bead_selected_idx]

# Here, we just choose some random exvol and exp parameters 
# for each bead combination type:

bead_interaction_params = {}
for pair in bead_labels:
    bead_interaction_params[pair] = {'ex_vol' : np.random.uniform(low=1, high=5),
                                     'exp' : np.random.uniform(low=1, high=5)}

bead_replayer = BeadRepulsionLayer(bead_selected_idx,
                                      bead_interaction_params,
                                      bead_array=bead_bead_array)


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


def test_bonds_residue_harmonic_layer():
    # Tests to make sure that the residue harmonic layer
    print(n_frames)
    # correctly calculates the harmonic contributions for
    # bonds from examples with varying residue encodings
    # First, we produce the manual calculation of the 
    # residue-dependent harmonic contributions 
    # First, we make the lookup table

    harmonic_parameters = torch.tensor([])
    column_lookup = {}
    for num, (label, interaction) in enumerate(residue_interaction_params.items()):
        column_lookup[label] = num
        harmonic_parameters = torch.cat((harmonic_parameters,
                                        torch.tensor([[interaction['k']],
                                                      [interaction['mean']]])),
                                                     dim=1)
    lookup_idx = []
    residue_beads = residue_encodings[:, residue_bead_array]
    local_beads = np.repeat(res_stats.beads_per_residue[None, residue_bead_array],
                            len(residue_beads), 0)
    final_labels = np.concatenate((local_beads, residue_beads), axis=2)
    for example in final_labels:
        lookup_idx.append([column_lookup[tuple(label)] for label in example])
    manual_energy = torch.sum(harmonic_parameters[0, lookup_idx] *
                              (torch.tensor(residue_distance_subset)
                               - harmonic_parameters[1, lookup_idx]) ** 2,
                              1).reshape(n_frames, 1) / 2

    # Finally, we test by asserting this manual calculation is reproduced with the 
    # the ResidueharmonicLayer

    energy = residue_hlayer(torch.tensor(residue_distance_subset),
                            tensor_residue_encodings)

    np.testing.assert_equal(manual_energy.detach().numpy(),
                            energy.numpy())


def test_bead_repulsion_layer():
    # Tests to make sure that the bead repulsion layer
    # correctly calculates the repulsion contributions for
    # bonds from examples with varying bead encodings
    # First, we produce the manual calculation of the 
    # bead-dependent repulsion contributions 
    # First, we make the lookup table

    repulsion_parameters = torch.tensor([])
    column_lookup = {}
    for num, (label, interaction) in enumerate(bead_interaction_params.items()):
        column_lookup[label] = num
        repulsion_parameters = torch.cat((repulsion_parameters,
                                        torch.tensor([[interaction['ex_vol']],
                                                      [interaction['exp']]])),
                                                     dim=1)
    lookup_idx = []
    final_labels = embeddings[:, bead_bead_array]
    for example in final_labels:
        lookup_idx.append([column_lookup[tuple(np.sort(label))] for label in example])

    manual_energy = torch.sum((repulsion_parameters[0, lookup_idx]/torch.tensor(bead_distance_subset))
                       ** repulsion_parameters[1, lookup_idx],
                       1).reshape(n_frames, 1) / 2

    # Finally, we test by asserting this manual calculation is reproduced with the 
    # the ResidueharmonicLayer

    energy = bead_replayer(torch.tensor(bead_distance_subset),
                            tensor_embeddings)

    np.testing.assert_equal(manual_energy.detach().numpy(),
                            energy.numpy())


def test_prior_cgnet_forwarding_logic():
    # Here, we test the ability of prior flow control in CGnet.forward()

    # We make a simply random model with a GeometryFeature, a SchNetFeature
    # and a random terminal network architecture

    n_layers = np.random.randint(low=5, high=10)
    width = np.random.randint(low=10, high=50)
    schnet_feature, _, feature_width = _get_random_schnet_feature(
                                           num_res * beads_per_res)
    layer_list = [geom_layer, schnet_feature]
    distance_idx = stats.return_indices(stats.descriptions['Distances'])
    feature = FeatureCombiner([geom_layer, schnet_feature],
                              distance_indices=distance_idx)
    arch = LinearLayer(feature_width, width, activation=nn.Tanh())
    criterion = ForceLoss()

    for _ in range(n_layers - 2):
        arch += LinearLayer(width, width, activation=nn.Tanh())
    arch += LinearLayer(width, 1, nn.Tanh())

    prior_sets = [ [normal_hlayer],
                   [normal_hlayer, residue_hlayer],
                   [normal_hlayer, bead_replayer],
                   [residue_hlayer],
                   [bead_replayer],
                   [normal_hlayer, residue_hlayer, bead_replayer] ]
    # First, we condisder the case where we have normal priors, but
    # no _ResiduePriorLayers or _BeadPriorLayers
    # For now, we just check to see that the coordinates and properties
    # pass through ok, but  we should check raised warnings in the future
    for prior_set in prior_sets:
        model = CGnet(arch=arch, feature=feature, priors=prior_set,
                      criterion=criterion)
        energy, forces = model(tensor_coords, tensor_embeddings,
                               tensor_residue_encodings)

def test_prior_computer():
    # Tests to make sure that the correct prior forces are calculated 
    # Using the prior force computer

    # Produce the tensor of invariant features from the coordinates
    geom_feat_out = geom_layer(tensor_coords)

    normal_distance_subset = geom_feat_out[:, normal_selected_idx]
    residue_distance_subset = geom_feat_out[:, residue_selected_idx]
    bead_distance_subset = geom_feat_out[:, bead_selected_idx]

    # Next, we produce the energies and forces manually for each prior
    # First, we start with the normal prior

    harmonic_parameters = torch.tensor([])
    for param_dict in normal_interaction_params:
        harmonic_parameters = torch.cat((harmonic_parameters,
                                         torch.tensor([[param_dict['k']],
                                                       [param_dict['mean']]])),
                                                       dim=1)

    manual_normal_energy = torch.sum(harmonic_parameters[0, :] *
                              (normal_distance_subset
                               - harmonic_parameters[1, :]) ** 2,
                              1).reshape(n_frames, 1) / 2

    # Next, we calculate the energy and forces from the residue prior

    harmonic_parameters = torch.tensor([])
    column_lookup = {}
    for num, (label, interaction) in enumerate(residue_interaction_params.items()):
        column_lookup[label] = num
        harmonic_parameters = torch.cat((harmonic_parameters,
                                        torch.tensor([[interaction['k']],
                                                      [interaction['mean']]])),
                                                     dim=1)
    lookup_idx = []
    residue_beads = residue_encodings[:, residue_bead_array]
    local_beads = np.repeat(res_stats.beads_per_residue[None, residue_bead_array],
                            len(residue_beads), 0)
    final_labels = np.concatenate((local_beads, residue_beads), axis=2)
    for example in final_labels:
        lookup_idx.append([column_lookup[tuple(label)] for label in example])
    manual_residue_energy = torch.sum(harmonic_parameters[0, lookup_idx] *
                              (residue_distance_subset
                               - harmonic_parameters[1, lookup_idx]) ** 2,
                              1).reshape(n_frames, 1) / 2

    # Lastly, we calculate the energy and forces from the bead prior

    repulsion_parameters = torch.tensor([])
    column_lookup = {}
    for num, (label, interaction) in enumerate(bead_interaction_params.items()):
        column_lookup[label] = num
        repulsion_parameters = torch.cat((repulsion_parameters,
                                        torch.tensor([[interaction['ex_vol']],
                                                      [interaction['exp']]])),
                                                     dim=1)
    lookup_idx = []
    final_labels = embeddings[:, bead_bead_array]
    for example in final_labels:
        lookup_idx.append([column_lookup[tuple(np.sort(label))] for label in example])

    manual_bead_energy = torch.sum((repulsion_parameters[0, lookup_idx]/bead_distance_subset)
                       ** repulsion_parameters[1, lookup_idx],
                       1).reshape(n_frames, 1) / 2

    # Here, we sum up each manual energy/force contribution
    total_manual_energy = torch.sum(manual_bead_energy + manual_residue_energy + manual_normal_energy)
    total_manual_forces = torch.autograd.grad(-total_manual_energy, tensor_coords,
        create_graph=True, retain_graph=True)[0]

    # Next, we create a PriorForceComputer module
    prior_computer = PriorForceComputer([bead_replayer, residue_hlayer, normal_hlayer],
                                        geom_layer)


    prior_energies, prior_forces = prior_computer(tensor_coords,
                                                  tensor_embeddings,
                                                  tensor_residue_encodings)

    # Lastly, we assert that these separate calculations result in the same
    # total forces and energies
    np.testing.assert_equal(total_manual_energy.detach().numpy(),
                            prior_energies.detach().numpy())
    np.testing.assert_equal(total_manual_forces.detach().numpy(),
                            prior_forces.detach().numpy())
