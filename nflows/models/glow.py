# Define flows
import torch
import normflows as nf


def get_glow_model():

    L = 3
    K = 16

    input_shape = (1, 320, 320)
    channels = 1
    hidden_channels = 256
    split_mode = "channel"
    scale = True
    num_classes = 2

    # Set up flows, distributions and merge operations
    q0 = []
    merges = []
    flows = []
    for i in range(L):
        flows_ = []
        for j in range(K):
            flows_ += [
                nf.flows.GlowBlock(
                    channels * 2 ** (L + 1 - i),
                    hidden_channels,
                    split_mode=split_mode,
                    scale=scale,
                )
            ]
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]
        latent_shape = (
            input_shape[0] * 2 ** (L - i),
            input_shape[1] // 2 ** (L - i),
            input_shape[2] // 2 ** (L - i),
        )
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (
                input_shape[0] * 2 ** (L - i),
                input_shape[1] // 2 ** (L - i),
                input_shape[2] // 2 ** (L - i),
            )
        else:
            latent_shape = (
                input_shape[0] * 2 ** (L + 1),
                input_shape[1] // 2**L,
                input_shape[2] // 2**L,
            )
        q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]

    # Construct flow model
    return nf.MultiscaleFlow(q0, flows, merges, class_cond=True)
