import os
from pathlib import Path
import logging
from typing import Optional
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, ListConfig

from .train import RNNtraining


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="MP_disRNN.yaml")
def main(cfg : DictConfig) -> None:
    log.info(f"Working directory: {os.getcwd()}")
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get() # type: ignore
    out_dir = Path(hydra_cfg['runtime']['output_dir'])  # type: ignore
    log.info(f"output directory: {out_dir}")

    # initialize the RNNtraining instance
    train_obj = instantiate(cfg.training, _convert_ = 'partial')
    assert isinstance(train_obj, RNNtraining)

    def run_post_proc(key:str):
        """run functions listed in config's `key` item
        functions are expected as f(train_data: RNNtraining, outdir: Path, ...)
        Args:
            key (str): the name of the item contains functions to run in the config
        """
        if _check_conf_item(cfg, key, ListConfig):
            for pp in cfg[key]:
                try:
                    OmegaConf.set_struct(pp, False) # to allow me add new fields
                    f = instantiate(pp, outdir = out_dir, _partial_ = True, _convert_ = 'partial')
                    f(train_data = train_obj) # dataclass cannot be passed as it is
                except Exception as e:
                    logging.error(e)

    # initialize and do train_sessions
    for sess in cfg.sessions:
        if len(sess) > 1:
            raise ValueError("specification of each session should be a dictionary with 0 or 1 item")
        
        sess_overrides = next(iter(sess.values()))
        sess_name = next(iter(sess.keys()))

        sess_conf = OmegaConf.merge(cfg.default_session, sess_overrides, {'name': sess_name})
        # logging.info(f"start trainning session: {sess_conf['name']}\n{OmegaConf.to_yaml(sess_conf)}")
        sess_conf = OmegaConf.to_container(sess_conf)

        # train_sess = instantiate(sess_conf, _convert_ = 'partial', name = sess_name)

        # add functions to run after each block to train_sess.
        if _check_conf_item(cfg, 'post_block', ListConfig):
            for pp in cfg['post_block']:
                OmegaConf.set_struct(pp, False) # to allow me add new fields
                f = instantiate(pp, outdir = out_dir, _partial_ = True, _convert_ = 'partial')
                logging.info(f"add post_block function: {f}")
                sess_conf['block_update_funcs'].append(f)
                
        logging.info(f"start trainning session: {sess_conf}")
        train_obj.train(**sess_conf)

        run_post_proc("post_sess")
    
    run_post_proc("post_processing")
  


def _check_conf_item(cfg: DictConfig, key:str, typ: Optional[type] = None) -> bool:
    if key in cfg:
        if typ is not None:
            tf = isinstance(cfg[key], typ)
            if typ is ListConfig:
                tf &= len(cfg[key]) > 0
            return tf
        else:
            return True
    return False



if __name__ == "__main__":
    main()