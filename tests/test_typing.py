from disRNN_MP.typing import validate_instantiatable

# The dictionary you want to validate
params = {
    '_target_': 'disRNN_MP.rnn.network_defs.make_transformed_disrnn',
    'latent_size': 5,
    'update_mlp_shape': [3, 3],
    'choice_mlp_shape': [2,],
    'obs_size': 2,
    'target_size': 2,
    'eval_mode': False
}

validate_instantiatable(params)