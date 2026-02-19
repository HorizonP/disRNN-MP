# %% 
from disRNN_MP.agent.MP_julia import EnvironmentBanditsMP_julia, multiEnvironmentBanditsMP_julia
import numpy as np

mpt = EnvironmentBanditsMP_julia()
mmpt = multiEnvironmentBanditsMP_julia()
# %%
for i in range(1000):
    mpt.step(np.random.choice(2))

# %%
mpt.history
mpt.biasCount
mpt.reward_probs_all

# %%
mmpt.n_sess = 5
mmpt.new_session()

for i in range(2000):
    mmpt.step(np.random.choice(2, 5))

# %%

sim_hist = mmpt.history

mmpt.eval_on_dataset(mmpt.history[['tri_id', 'sess_id', 'choice', 'reward']])

eval_hist = mmpt.history

assert np.all(sim_hist == eval_hist)