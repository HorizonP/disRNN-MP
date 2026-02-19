# %%
#!%load_ext autoreload
#!%autoreload 3

# %% imports
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
import jax

from disRNN_MP.rnn.train_db import Base, trainingSession, ModelTrainee, trainingRecord, trainingFork
from disRNN_MP.rnn.utils import make_param_metric_expLL
from disRNN_MP.rnn.network_defs import make_transformed_disrnn
from disRNN_MP.rnn.plots import disrnn_dashboard_figure

engine = create_engine('sqlite:///tst_ModelTrainer.db', echo=False)
Base.metadata.create_all(engine)

# %% compose to test bottleneck converge criteria
    
mt = ModelTrainee(
    dry_model = {
        '_target_': 'disRNN_MP.rnn.utils.make_RNNtransformed',
        '_hk_module': 'disRNN_MP.rnn.disrnn2.hkDisRNN',
        'latent_size': 8, 
        'update_mlp_shape': [3,3], 
        'choice_mlp_shape': [4,2,], 
        'target_size': 2, 
        'eval_mode': False
    },
    dry_eval_model = {
        '_target_': 'disRNN_MP.rnn.utils.make_RNNtransformed',
        '_hk_module': 'disRNN_MP.rnn.disrnn2.hkDisRNN',
        'latent_size': 8, 
        'update_mlp_shape': [3,3], 
        'choice_mlp_shape': [4,2,], 
        'target_size': 2, 
        'eval_mode': True
    },
    dry_default_datasets = {
        '_target_': 'disRNN_MP.dataset.train_test_datasets',
        'dat_or_path': Path("../data/mp_beh_m18_500completed.npy").resolve().as_posix(),
        'n_sess_sample': 42,
        'seed': 78
    },
    dry_default_optimizer = {
        '_target_': 'optax.adam', 
        'learning_rate': 1e-3
    },
    dry_default_make_param_metric = {
            '_target_': 'disRNN_MP.rnn.disrnn2.make_param_metric_expLL',
            '_partial_': True,
        },

)

se1 = trainingSession(
    dry_make_train_step = {
        '_target_': 'disRNN_MP.rnn.utils.make_train_step_with_loss_fun', 
        '_partial_': True,
        'make_loss_fun': {
            '_target_': 'disRNN_MP.rnn.disrnn2.make_loss_fun',
            '_partial_': True,
        },
        'penalty_scale': 0,
        'beta_scale': 1,
    },
    n_step = 200,
    steps_per_block = 100,
)

se2 = trainingSession(
    dry_make_train_step = {
        '_target_': 'disRNN_MP.rnn.utils.make_train_step_with_loss_fun', 
        '_partial_': True,
        'make_loss_fun': {
            '_target_': 'disRNN_MP.rnn.disrnn2.make_loss_fun',
            '_partial_': True,
        },
        'penalty_scale': 5e-4,
        'beta_scale': 1,
    },
    n_step = 2000,
    steps_per_block = 100,
    dry_stop_criteria = {
        '_target_': 'disRNN_MP.rnn.stop_crit.disrnnConvTest_meanDiff',
        'thres': 5e-4,
    },
)

mt.sessions = [se1, se2]

with Session(engine) as sess:
    sess.add(mt)
    sess.commit()
# %% training the most recent one
    
sess = Session(engine, expire_on_commit=False)
# dbtq = sess.execute(select(ModelTrainee).where(ModelTrainee.id == 5)).scalar_one()
dbtq = sess.query(ModelTrainee).order_by(ModelTrainee.id.desc()).first() 

dbtq.train(sess)
# %% 

disrnn_dashboard_figure(dbtq)