# Author: Nick Charron 

import numpy as np
import scipy.spatial
import torch

from cgnet.feature import GeometryStatistics, EmbeddingStatistics
from cgnet.network import EmbeddingHarmonicLayer

# The following sets up our pseud-simulation data

# Number of frames
frames = np.random.randint(5, 10) * 2 #double this to avoid nonzero variances

# Number of coarse-grained beads. We need at least 8 so we can do
# dihedrals in the backbone tests (where every other atom is designated
# as a backbone atom)
beads = np.random.randint(8, 20)

# Number of dimensions; for now geometry only handles 3
dims = 3

# Create a pseudo simulation dataset, but double each frame to get nonzero variances
data = np.random.randn(frames, beads, dims)


# Create embeddings that vary from frame to frame to simulate multiple
# different molecules within the same dataset

embeddings = np.random.randint(low=1, high=10, size=(int(frames/2),beads)) #again,double this so
# we dont get nonzero variances.

embeddings = np.tile(embeddings, (2,1))

# random temperature
temperature = np.random.uniform(low=250,high=350)
KBOLTZMANN = 1.38064852e-23
AVOGADRO = 6.022140857e23
JPERKCAL = 4184
beta = JPERKCAL/KBOLTZMANN/AVOGADRO/temperature

stats = EmbeddingStatistics(data, embeddings, backbone_inds='all',
                           get_all_distances=True,
                           get_backbone_angles=True,
                           get_backbone_dihedrals=True,
                           get_redundant_distance_mapping=True,
                           temperature=temperature)
beta = stats.beta


def test_embedding_harmonic_layer():
    # Tests to make sure that interaction parameters are assembled 
    # properly according to the input embeddings, resulting in the
    # correct energy predictions

    # First, we construct a random set of distances and angles
    # with which we can constrain with an embedding dependent prior

    num_bonds = np.random.randint(1, high=len(stats._distance_pairs))
    num_angles = np.random.randint(1, high=len(stats._angle_trips))

    bond_idx = np.random.choice(np.arange(0, len(stats._distance_pairs)),
                                num_bonds, replace=False)
    angle_idx = np.random.choice(np.arange(0, len(stats._angle_trips)),
                                num_angles, replace=False)

    bond_features = [stats._distance_pairs[i] for i in bond_idx]
    angle_features = [stats._angle_trips[i] for i in angle_idx]
    #print(bond_features)
    bond_dict = stats.get_prior_statistics(bond_features)
    angle_dict = stats.get_prior_statistics(angle_features)
    #print(bond_dict)
    # Here, the callback indices are just placeholders, though
    # we still calculate them in the intended fashion using the 
    # stats.return_indices method

    bond_callback_idx = stats.return_indices(bond_features)
    angle_callback_idx = stats.return_indices(angle_features)

    # Next, we construct embedding harmonic layers for both feature sets

    bond_embedding_hlayer = EmbeddingHarmonicLayer(bond_callback_idx,
                                bond_dict["Distances"], bond_features)
    angle_embedding_hlayer = EmbeddingHarmonicLayer(angle_callback_idx,
                                angle_dict["Angles"], angle_features)

    #print(bond_embedding_hlayer.parameter_dict)

    # we produce a manual calculation of the energies for each
    # feature set

    distances = torch.tensor(stats.distances[:, bond_idx])
    angles = torch.tensor(stats.angles[:, angle_idx])
    embeddings = torch.tensor(stats.embeddings)

    # first, we handle the bonds
    bond_tuples = torch.tensor(bond_features)
    embedding_tuples = embeddings[:, bond_tuples]
    num_examples = distances.size()[0]
    bond_tensor_lookups = torch.cat((bond_tuples[None, :].repeat(num_examples,1,1),
                              embedding_tuples), dim=-1)

    bond_means = torch.tensor([[bond_dict["Distances"][tuple(lookup)]['mean']
                          for lookup in bond_tensor_lookups[i].numpy()]
                          for i in range(num_examples)])
    bond_constants = torch.tensor([[bond_dict["Distances"][tuple(lookup)]['k']
                          for lookup in bond_tensor_lookups[i].numpy()]
                          for i in range(num_examples)])

    manual_bond_energy = torch.sum(bond_constants * ((distances - bond_means)**2)) / 2.0

    # next, we handle the angles
    angle_tuples = torch.tensor(angle_features)
    embedding_tuples = embeddings[:, angle_tuples]
    num_examples = angles.size()[0]
    angle_tensor_lookups = torch.cat((angle_tuples[None, :].repeat(num_examples,1,1),
                              embedding_tuples), dim=-1)

    angle_means = torch.tensor([[angle_dict["Angles"][tuple(lookup)]['mean']
                          for lookup in angle_tensor_lookups[i].numpy()]
                          for i in range(num_examples)])
    angle_constants = torch.tensor([[angle_dict["Angles"][tuple(lookup)]['k']
                          for lookup in angle_tensor_lookups[i].numpy()]
                          for i in range(num_examples)])

    manual_angle_energy = torch.sum(angle_constants * ((angles - angle_means)**2)) / 2.0

    # Lastly, we compute the same energies using the EmbeddingHarmonicLayers
    # and compare their output with the manual calculations above

    bond_energy = bond_embedding_hlayer(distances, embeddings)
    angle_energy = angle_embedding_hlayer(angles, embeddings)

    np.testing.assert_equal(manual_bond_energy.numpy(),
                            bond_energy.numpy())
    np.testing.assert_equal(manual_angle_energy.numpy(),
                            angle_energy.numpy())


def test_embedding_harmonic_layer_mixed_features():
    # This repeats the same test as above but instead combines the angle and 
    # bond features into one feature list and combines the two 
    # EmbeddingHarmonicLayers into one EmbeddingHarmonicLayer

    num_bonds = np.random.randint(1, high=len(stats._distance_pairs))
    num_angles = np.random.randint(1, high=len(stats._angle_trips))

    bond_idx = np.random.choice(np.arange(0, len(stats._distance_pairs)),
                                num_bonds, replace=False)
    angle_idx = np.random.choice(np.arange(0, len(stats._angle_trips)),
                                num_angles, replace=False)

    bond_features = [stats._distance_pairs[i] for i in bond_idx]
    angle_features = [stats._angle_trips[i] for i in angle_idx]
    features = bond_features + angle_features

    # In this case, we must reform the dictionary to remove the outer
    # feature type keys "Distances" and "Angles"

    feature_dict = stats.get_prior_statistics(features)
    processed_dict = {}
    for key, value in feature_dict["Distances"].items():
        processed_dict[key] = value
    for key, value in feature_dict["Angles"].items():
        processed_dict[key] = value

    # Here, the callback indices are just placeholders, though
    # we still calculate them in the intended fashion using the 
    # stats.return_indices method

    callback_idx = stats.return_indices(features)

    # Next, we construct embedding harmonic layers for both feature sets

    combined_embedding_hlayer = EmbeddingHarmonicLayer(callback_idx,
                                processed_dict, features)

    # we produce a manual calculation of the energies for each
    # feature set

    distances = torch.tensor(stats.distances[:, bond_idx])
    angles = torch.tensor(stats.angles[:, angle_idx])
    input_features = torch.cat(distances, angles, dim=1)
    embeddings = torch.tensor(stats.embeddings)

    # first, we handle the bonds
    feature_tuples = torch.tensor(features)
    embedding_tuples = embeddings[:, feature_tuples]
    num_examples = input_features.size()[0]
    tensor_lookups = torch.cat((bond_tuples[None, :].repeat(num_examples,1,1),
                              embedding_tuples), dim=-1)

    means = torch.tensor([[processed_dict[tuple(lookup)]['mean']
                          for lookup in tensor_lookups[i].numpy()]
                          for i in range(num_examples)])
    constants = torch.tensor([[processed_dict[tuple(lookup)]['k']
                          for lookup in tensor_lookups[i].numpy()]
                          for i in range(num_examples)])

    manual_energy = torch.sum(constants * ((input_features - means)**2)) / 2.0

    # Lastly, we compute the same energies using the EmbeddingHarmonicLayers
    # and compare their output with the manual calculations above

    energy = combined_embedding_hlayer(input_feautures, embeddings)

    np.testing.assert_equal(manual_energy.numpy(),
                            energy.numpy())

