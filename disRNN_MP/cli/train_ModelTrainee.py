"""start training a ModelTrainee specified in a database

example usage: 

this will start training on the earliest ModelTrainee in the database that no other worker is working on
>>> python -m disRNN_MP.cli.train_ModelTrainee sql_engine.url='sqlite:///scripts/tst.db' 

specify `+trainee_id` will try to train that ModelTrainee:
>>> python -m disRNN_MP.cli.train_ModelTrainee sql_engine.url='sqlite:///scripts/tst.db' +trainee_id=1

engine url path is relative to the working directory where this command is called
"""

import os
import sys
from pathlib import Path
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".32"

import logging
import hydra
import jax

from omegaconf import DictConfig


from sqlalchemy import create_engine, or_
from sqlalchemy.orm import Session

from disRNN_MP.rnn.train_db import Base, ModelTrainee


@hydra.main(version_base=None, config_path=".", config_name="train_ModelTrainee.yaml")
def main(cfg : DictConfig) -> None:
    logging.info(f"Working directory: {os.getcwd()}")
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get() # type: ignore
    out_dir = Path(hydra_cfg['runtime']['output_dir'])  # type: ignore
    logging.info(f"output directory: {out_dir}")

    engine = create_engine(**cfg.sql_engine)
    Base.metadata.create_all(engine)

    # if cfg.platform == 'cpu':
    jax.config.update('jax_platform_name', cfg.jax_platform)

    if 'max_na_try' in cfg:
        max_na_try = cfg.max_na_try
    else:
        max_na_try = 10

    def _proceed_with_res(res, sess, **kwargs):
        """process the result from database query

        Returns:
            int: 
                -1 - no suitable instance to run
                0 - finished all trainingSession of the current ModelTrainee
                1 - abort due to another worker working on the instance
                2 - abort due to unresolvable NaN in parameters
                3 - abort due to external signal (such as SIGINT, SIGTERM)
        """
        if res is None:
            logging.info('cannot found any suitable ModelTrainee for training')
            return -1
        
        logging.info(f'started working on trainee (id = {res.id})')
        if 'SLURM_JOB_ID' in os.environ:
            worker = os.environ['SLURM_JOB_ID']
            logging.info(f'slurm job id: {worker}')
        else:
            worker = None
        
        code = res.train(sess, worker, **kwargs)
        return code
    
    def _search_and_train(**kwargs):
        with Session(engine) as sess:
                res = (sess.query(ModelTrainee)
                    .filter(
                        ModelTrainee.worker == None, 
                        or_(
                            ModelTrainee.chk_step < ModelTrainee.total_steps,
                            ModelTrainee.chk_step == None))
                    .order_by(ModelTrainee.id)
                    .first()
                )
                code = _proceed_with_res(res, sess, **kwargs)
        return code

    # find and do job(s)
    if 'trainee_id' in cfg and cfg.trainee_id is not None: # train a specific instance 
        with Session(engine, expire_on_commit=False) as sess:
            res = (sess.query(ModelTrainee)
                .filter(
                    ModelTrainee.worker == None, 
                    or_(
                        ModelTrainee.chk_step < ModelTrainee.total_steps,
                        ModelTrainee.chk_step == None),
                    ModelTrainee.id == cfg.trainee_id)
                .order_by(ModelTrainee.id)
                .one_or_none()
            )     
            code = _proceed_with_res(res, sess, max_na_retry = max_na_try)
    elif 'scavenge' in cfg and cfg.scavenge is True: # scavenge mode: keep training until no trainable instance left
        code = None
        while code in [None, 0, 1, 2]: # possibly search for other unfinished job and take them
            code = _search_and_train(max_na_retry = max_na_try, pause_unresolvable_NaN = True)
    else: # train the firstly found instance only
        code = None
        while code in [None, 1]: # retry-able
            code = _search_and_train(max_na_retry = max_na_try, pause_unresolvable_NaN = True)
                

    if code == 3:
        sys.exit(1)

    

            

        

            

if __name__ == "__main__":
    main()