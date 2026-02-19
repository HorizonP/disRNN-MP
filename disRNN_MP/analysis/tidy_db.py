""" functions for tidying up model meta information from database entries of `ModelTrainee`
"""
import re
from typing import Tuple

import pandas as pd
from flax.serialization import msgpack_restore
from sqlalchemy import and_
from disRNN_MP.rnn.train_db import ModelTrainee, trainingRecord

def re_search_1st(pat, string):
    match = re.search(pat, str(string))
    if match:
        return match.group(1)
    else:
        return None
    
class db_scrapper:
    """utility functions to scrap information from columns in the ModelTrainee database
    """
    @classmethod
    def model_type(cls, dry_model: dict):
        x = dry_model
        return "disRNN" if 'choice_mlp_shape' in x or 'update_mlp_shape' in x or 'eval_mode' in x else re_search_1st(r"make_(\w+)", x['hkRNN']['_target_'])
    
    @classmethod
    def latent_size(cls, dry_model: dict):
        x = dry_model
        return x['hkRNN']['hidden_size'] if 'hkRNN' in x else x['latent_size']
    
    @classmethod
    def monkey(cls, descrp: str):
        return re_search_1st(r"(m\d+)", descrp)
    
    @classmethod
    def optimizer(cls, dry_optim: dict):
        return dry_optim['_target_']
    
    @classmethod
    def lr(cls, dry_optim: dict):
        return dry_optim['learning_rate']
    
    @classmethod
    def batch_sz(cls, dry_ds: dict):
        return dry_ds['batch_size'] if dry_ds is not None and 'batch_size' in dry_ds else None
    
    @classmethod
    def params(cls, params:bytes):
        return msgpack_restore(params)
    
    @classmethod
    def disrnn_penalty_scale(cls, make_train_step: dict):
        return make_train_step['penalty_scale'] if 'penalty_scale' in make_train_step else None
    
    @classmethod
    def disrnn_beta_scale(cls, make_train_step: dict):
        return make_train_step['beta_scale'] if 'beta_scale' in make_train_step else 1
    
    @classmethod
    def disrnn_theta_scale(cls, make_train_step: dict):
        return make_train_step['theta_scale'] if 'theta_scale' in make_train_step else 1
    
    @classmethod
    def disrnn_choiceMLP(cls, model: dict):
        return str(model['choice_mlp_shape']) if 'choice_mlp_shape' in model else None
    
    @classmethod
    def disrnn_updateMLP(cls, model: dict):
        return str(model['update_mlp_shape']) if 'update_mlp_shape' in model else None

def _metric_info(mt):
    step_max, max_v = max(
        [(rec.step, rec.test_metric) for rec in mt.records], 
        key = lambda x: x[1]
    )
    return dict(
        max_test_metric = max_v,
        step_max_test_metric = step_max,
    )


def rnn_model_info(mt:ModelTrainee):
    ds = mt.dry_default_datasets if mt.dry_default_datasets is not None else mt.sessions[0].dry_datasets
    optim = mt.dry_default_optimizer if mt.dry_default_optimizer is not None else mt.sessions[0].dry_optimizer

    return dict(
        model_arch = re_search_1st(r"make_(\w+)", mt.dry_model['hkRNN']['_target_']),
        latent_size = mt.dry_model['hkRNN']['hidden_size'],
        init_seed = mt.init_rand_seed,
        monkey = re_search_1st(r"(m\d+)", mt.description),
        optimizer = optim['_target_'],
        learning_rate = optim['learning_rate'],
        batch_size = ds['batch_size'] if ds is not None and 'batch_size' in ds else None,
        total_step = sum([se.n_step for se in mt.sessions]),
        **_metric_info(mt),
    )


def tidy_best_params(df: pd.DataFrame):
    df = df.assign(
        latent_size = [db_scrapper.latent_size(x) for x in df['model']],
        model_arch = [db_scrapper.model_type(x) for x in df['model']],
        monkey = [db_scrapper.monkey(x) for x in df['description']],
        optimizer = [db_scrapper.optimizer(x) for x in df['default_optimizer']],
        learning_rate = [db_scrapper.lr(x) for x in df['default_optimizer']],
        batch_size = [db_scrapper.batch_sz(x) for x in df['default_datasets']]
    )
    return df


def get_mt_model_by_step(db_sess, mt_id, step, alt_cwd = None) -> Tuple[ModelTrainee, trainingRecord]:
    """Query the database session for a ModelTrainee with the given training id and step number

    Args:
        db_sess (_type_): open sqlalchemy session
        mt_id (_type_): training id
        step (_type_): step

    Raises:
        ValueError: cannot find the given step in history record

    Returns:
        Tuple[ModelTrainee, trainingRecord]
    """
    # 
    mt:ModelTrainee = db_sess.query(ModelTrainee).where(ModelTrainee.id == mt_id).one()
    # Load the data from the database into the ModelTrainee object
    mt.materialize(alt_cwd = alt_cwd)

    rec = db_sess.query(trainingRecord).where(and_(trainingRecord.training_id == mt_id, trainingRecord.step == step)).one()
    
    # Return a tuple containing the ModelTrainee and the corresponding training record
    return (mt, rec)


