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

engine = create_engine('sqlite:///tst_ModelTrainer.db', echo=False)
Base.metadata.create_all(engine)

# %% version 1 compose

se1 = trainingSession(
    dry_optimizer = {
        '_target_': 'optax.adam', 
        'learning_rate': 1e-3
    },
    dry_make_train_step = {
        '_target_': 'disRNN_MP.rnn.utils.make_train_step', 
        '_partial_': True,
        'loss_type': 'penalized_categorical',
        'penalty_scale': 0
    },
    dry_make_param_metric = {
        # '_target_': 'disRNN_MP.rnn.utils.make_param_metric_expLL',
        '_target_': lambda: jax.tree_util.Partial(make_param_metric_expLL, n_action = 2),
        # '_partial_': True,
        # 'n_action': 2
    },
    dry_datasets = {
        '_target_': 'disRNN_MP.dataset.train_test_datasets',
        'dat_or_path': Path("../data/mp_beh_m18_500completed.npy").resolve().as_posix(),
        'n_sess_sample': 42,
        'seed': 5
    },
    n_step = 20,
    steps_per_block = 10
)

se2 = trainingSession(
    dry_optimizer = {
        '_target_': 'optax.adam', 
        'learning_rate': 1e-3
    },
    dry_make_train_step = {
        '_target_': 'disRNN_MP.rnn.utils.make_train_step', 
        '_partial_': True,
        'loss_type': 'penalized_categorical',
        'penalty_scale': 1e-3
    },
    dry_make_param_metric = {
        '_target_': 'disRNN_MP.rnn.utils.make_param_metric_expLL',
        '_partial_': True,
        'n_action': 2
    },
    dry_datasets = {
        '_target_': 'disRNN_MP.dataset.train_test_datasets',
        'dat_or_path': Path("../data/mp_beh_m18_500completed.npy").resolve().as_posix(),
        'n_sess_sample': 42,
        'seed': 5
    },
    n_step = 20,
    steps_per_block = 10
)



mt = ModelTrainee(
    dry_model = {
        '_target_': 'disRNN_MP.rnn.network_defs.make_transformed_disrnn',
        'latent_size': 5, 
        'update_mlp_shape': [3,3], 
        'choice_mlp_shape': [2,], 
        'obs_size': 2, 
        'target_size': 2, 
        'eval_mode': False
    },
    eval_model = make_transformed_disrnn(5, [3,3], [2,], 2, 2, True)
)

mt.sessions = [se1, se2]

with Session(engine) as sess:
    sess.add(mt)
    sess.commit()
# %% version 2 compose
    
mt = ModelTrainee(
    dry_model = {
        '_target_': 'disRNN_MP.rnn.network_defs.make_transformed_disrnn',
        'latent_size': 5, 
        'update_mlp_shape': [3,3], 
        'choice_mlp_shape': [2,], 
        'obs_size': 2, 
        'target_size': 2, 
        'eval_mode': False
    },
    eval_model = make_transformed_disrnn(5, [3,3], [2,], 2, 2, True),
    dry_default_datasets = {
        '_target_': 'disRNN_MP.dataset.train_test_datasets',
        'dat_or_path': Path("../data/mp_beh_m18_500completed.npy").resolve().as_posix(),
        'n_sess_sample': 42,
        'seed': 5
    },
    dry_default_optimizer = {
        '_target_': 'optax.adam', 
        'learning_rate': 1e-3
    },
)

se1 = trainingSession(
    dry_make_train_step = {
        '_target_': 'disRNN_MP.rnn.utils.make_train_step', 
        '_partial_': True,
        'loss_type': 'penalized_categorical',
        'penalty_scale': 0
    },
    make_param_metric = jax.tree_util.Partial(make_param_metric_expLL, n_action = 2),
    n_step = 20,
    steps_per_block = 10
)

se2 = trainingSession(
    dry_make_train_step = {
        '_target_': 'disRNN_MP.rnn.utils.make_train_step', 
        '_partial_': True,
        'loss_type': 'penalized_categorical',
        'penalty_scale': 1e-3
    },
    dry_make_param_metric = {
        '_target_': 'disRNN_MP.rnn.utils.make_param_metric_expLL',
        '_partial_': True,
        'n_action': 2
    },
    n_step = 20,
    steps_per_block = 10
)

mt.sessions = [se1, se2]

with Session(engine) as sess:
    sess.add(mt)
    sess.commit()
# %%
# %% version 2 compose (disrnn2)
    
mt = ModelTrainee(
    dry_model = {
        '_target_': 'disRNN_MP.rnn.utils.make_RNNtransformed',
        '_hk_module': 'disRNN_MP.rnn.disrnn2.hkDisRNN',
        'latent_size': 5, 
        'update_mlp_shape': [3,3], 
        'choice_mlp_shape': [2,], 
        'target_size': 2, 
        'eval_mode': False
    },
    dry_eval_model = {
        '_target_': 'disRNN_MP.rnn.utils.make_RNNtransformed',
        '_hk_module': 'disRNN_MP.rnn.disrnn2.hkDisRNN',
        'latent_size': 5, 
        'update_mlp_shape': [3,3], 
        'choice_mlp_shape': [2,], 
        'target_size': 2, 
        'eval_mode': True
    },
    dry_default_datasets = {
        '_target_': 'disRNN_MP.dataset.train_test_datasets',
        'dat_or_path': Path("../data/mp_beh_m18_500completed.npy").resolve().as_posix(),
        'n_sess_sample': 42,
        'seed': 5
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
    n_step = 20,
    steps_per_block = 10
)

se2 = trainingSession(
    dry_make_train_step = {
        '_target_': 'disRNN_MP.rnn.utils.make_train_step_with_loss_fun', 
        '_partial_': True,
        'make_loss_fun': {
            '_target_': 'disRNN_MP.rnn.disrnn2.make_loss_fun',
            '_partial_': True,
        },
        'penalty_scale': 1e-3,
        'beta_scale': 1,
    },
    n_step = 20,
    steps_per_block = 10
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
# %% fork and train
sess = Session(engine, expire_on_commit=False)
rec = sess.query(trainingRecord).where(trainingRecord.step == 21).order_by(trainingRecord.id.asc()).first()
omt = rec.parent_training

# ModelTrainee.delete(sess, 4)
# sess.query(trainingFork).where(trainingFork.id == 4).delete()

mt = ModelTrainee.fork_from(rec)
mt.sessions.append(trainingSession(
    dry_make_train_step = {
        '_target_': 'disRNN_MP.rnn.utils.make_train_step', 
        '_partial_': True,
        'loss_type': 'penalized_categorical',
        'penalty_scale': 1e-3
    },
    dry_make_param_metric = {
        '_target_': 'disRNN_MP.rnn.utils.make_param_metric_expLL',
        '_partial_': True,
        'n_action': 2
    },
    n_step = 20,
    steps_per_block = 10
))

sess.add(mt)
mt.train(sess)
sess.commit()

rec2 = [rec for rec in mt.records if rec.step == 11][0]
rec1 = [rec for rec in omt.records if rec.step == 31][0]

rec2.test_metric == rec1.test_metric

# %% test cli.train_ModelTrainee
import subprocess

# Define the command and arguments
command = [
    "python",            # Python interpreter
    "-m",                # Run library module as a script
    "disRNN_MP.cli.train_ModelTrainee",  # Module to run
    "sql_engine.url=sqlite:///tst_ModelTrainer.db"  # Argument to pass
]

# Run the command
result = subprocess.run(command)
# %%
#!%load_ext snakeviz
#!%snakeviz_config -h localhost -p 8901