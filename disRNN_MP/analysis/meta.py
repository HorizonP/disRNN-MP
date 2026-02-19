import itertools
from typing import Callable, List,Iterable, Literal, Type, Union
from warnings import warn

from disRNN_MP.analysis.disrnn_static_analy import disRNN_params_analyzer
from disRNN_MP.rnn.pkl_instantiate import _pkl_instantiate
from disRNN_MP.rnn.utils import transform_hkRNN
from diskcache import Cache
import jax

import pandas as pd
from sqlalchemy import select, func, and_
from sqlalchemy.orm import Query, Session

from disRNN_MP.analysis.tidy_db import db_scrapper
from disRNN_MP.analysis.model_graph import group_isomorphic
from disRNN_MP.rnn.train_db import ModelTrainee, trainingSession, trainingRecord
from disRNN_MP.analysis.disrnn_analyzer import disRNN_model_analyzer
from disRNN_MP.analysis.utils import N_edge_full_conn

from tqdm import tqdm
tqdm.pandas()

def query_converged_models(sess: Session, train_ids: Iterable[int]):
    """get all the last step model of each session (except 1st session) of the training series
    
    Args:
        sess (_type_): _description_
        train_ids (Iterable): _description_

    Returns:
        Query: a sqlalchemy query that will return a list of trainingRecord
    """

    # a sub-query to calculate the last step in each session
    subq = (
        sess.query(
            trainingRecord.session_id,
            func.max(trainingRecord.step).label("sess_last_step")
        )
        .group_by(trainingRecord.session_id)
        .subquery()
    )

    q = (sess.query(trainingRecord)
        .join(trainingSession, trainingSession.id == trainingRecord.session_id)
        .join(subq, and_(
            subq.c.session_id == trainingRecord.session_id,
            subq.c.sess_last_step == trainingRecord.step))
        .where(
            trainingRecord.training_id.in_(train_ids),
            trainingSession.index >= 1,
            )
    )

    return q

def rename_columns(cols, mapper):
    return [
    c.label(mapper[c.name]) 
    if c.name in mapper
    else c
    for c in cols]

def query_converged_models_df(conn, train_ids: Iterable):
    subq = (
        select(
            trainingRecord.session_id,
            func.max(trainingRecord.step).label("sess_last_step")
        )
        .group_by(trainingRecord.session_id)
        .subquery()
    )

        
    trec_cols = [c for c in trainingRecord.__table__.columns if c.name not in [
        "time_added", 'index', 'opt_state', 'rand_key', 'worker']]

    # trec_cols = map(lambda c: c.label("r_" + c.name), trec_cols)

    tses_col_rename = {
        'index': 'sess_index',
        'name': 'sess_name',
    }

    mt_col_rename = {
        'worker': 'mt_worker',
    }

    tses_cols = [c for c in trainingSession.__table__.columns if c.name not in [
        'id', 'training_id', ]]

    tses_cols = rename_columns(tses_cols, tses_col_rename)

    # tses_cols = map(lambda c: c.label("s_" + c.name), tses_cols)

    mt_cols = [c for c in ModelTrainee.__table__.columns if c.name not in [
        'id', 'index', 'training_id', 'chk_params', 'chk_opt_state', 'chk_rand_key']]

    mt_cols = rename_columns(mt_cols, mt_col_rename)

    # mt_cols = map(lambda c: c.label("m_" + c.name), mt_cols)

    q = (select(*trec_cols, *tses_cols, *mt_cols, ModelTrainee.total_steps)
        .join(trainingSession, trainingSession.id == trainingRecord.session_id)
        .join(ModelTrainee, ModelTrainee.id == trainingRecord.training_id)
        .join(subq, and_(
            subq.c.session_id == trainingRecord.session_id,
            subq.c.sess_last_step == trainingRecord.step))
        .where(
            trainingRecord.training_id.in_(train_ids),
            trainingSession.index >= 1,
            )
    )

    res = pd.read_sql_query(q, conn)

    return res

def _with_default(value, default):
    """return the value if it is not None, otherwise return default"""
    if value is None:
        return default
    else:
        return value

def _det_train_status(mt: ModelTrainee):
    """determine training status from the database columns
    if worker column has anything, will use it
    if worker column is empty, determine if it is finished by comparing total_steps and current step
    """
    _st = mt.worker
    if _st is None:
        if mt.total_steps == mt.chk_step:
            _st = "finished"
        else:
            _st = "unfinished"
    return _st

def _det_train_status_df(mt_worker, total_steps, chk_step):
    """determine training status from the database columns
    if worker column has anything, will use it
    if worker column is empty, determine if it is finished by comparing total_steps and current step
    """
    _st = mt_worker
    if _st is None:
        if total_steps == chk_step:
            _st = "finished"
        else:
            _st = "unfinished"
    return _st

def _disrnn_static_attrs(mda: disRNN_params_analyzer):
    """get the hyperparameter subset of disRNN static attributes"""
    keys = mda._static_attrs.keys() - {'_auto_repr', '_creation_frame_id', '_submodules', 'get_update_sigma', 'get_latent_sigma', 'get_choice_sigma', 'apply_latent_bn'}
    return {k:mda._static_attrs[k] for k in keys}
class disRNN_model_meta_analysis:
    """ this class pull the analyses of many trained disRNN models together

    the core data is a dataframe contains the descriptions of these disRNN models

    the init function will instantiate from a database query
    """

    df: pd.DataFrame
    """a dataframe containing meta information about each model and training hyper-parameters"""

    objs: pd.DataFrame
    """a dataframe containing (not easily serializable) objects for each model"""

    def __init__(self, df, objs) -> None:

        self.df = df
        self.objs = objs

        self.e_model_architecture()
        self._set_forward_funcs()

    @classmethod
    def from_TRec_query(cls, qTRec: List[trainingRecord] | Query, model_analyzer: Union[Type[disRNN_model_analyzer], Callable[..., disRNN_model_analyzer]] = disRNN_model_analyzer):
        
        if isinstance(qTRec, Query):
            qTRec = qTRec.all()
            
        print('gather basic information from database...')
        # basic = pd.DataFrame({
        #     'rec_id': [r.id for r in qTRec],
        #     'training_id': [r.parent_training.id for r in qTRec],
        #     'step': [r.step for r in qTRec],
        #     'train_metric': [r.train_metric for r in qTRec],
        #     'test_metric': [r.test_metric for r in qTRec],
        #     'init_rand_seed': [r.parent_training.init_rand_seed for r in qTRec],
        #     'description': [r.parent_training.description for r in qTRec],
        #     'sess_name': [r.parent_session.name for r in qTRec],
        #     'sess_index': [r.parent_session.index for r in qTRec],
        #     'training_status': [_det_train_status(r.parent_training) for r in qTRec],
        # })

        # print('gather python objects from database...')
        # objs = pd.DataFrame({
        #     'rec_id': [r.id for r in qTRec],
        #     'md_analyzer': [model_analyzer.from_trainingRecord(r, fix_name=True, jit=False) for r in qTRec],
        #     'dry_model': [r.parent_training.dry_model for r in qTRec],
        #     'dry_training_step': [r.parent_session.dry_make_train_step for r in qTRec],
        #     'dry_optimizer': [_with_default(
        #         r.parent_session.dry_optimizer, r.parent_training.dry_default_optimizer) for r in qTRec],
        #     'dry_dataset': [_with_default(
        #         r.parent_session.dry_datasets, r.parent_training.dry_default_datasets) for r in qTRec],
        #     'dry_metric_func': [_with_default(
        #         r.parent_session.dry_make_param_metric, r.parent_training.dry_default_make_param_metric) for r in qTRec],
        # })

        basic = []
        objs = []
        for r in tqdm(qTRec):
            basic.append({
                'rec_id': r.id,
                'training_id': r.parent_training.id,
                'step': r.step,
                'train_metric': r.train_metric,
                'test_metric': r.test_metric,
                'init_rand_seed': r.parent_training.init_rand_seed,
                'description': r.parent_training.description,
                'sess_name': r.parent_session.name,
                'sess_index': r.parent_session.index,
                'training_status': _det_train_status(r.parent_training),
            })

            objs.append({
                'rec_id': r.id,
                'md_analyzer': model_analyzer.from_trainingRecord(r, fix_name=True),
                'dry_model': r.parent_training.dry_model,
                'dry_training_step': r.parent_session.dry_make_train_step,
                'dry_optimizer': _with_default(
                    r.parent_session.dry_optimizer, r.parent_training.dry_default_optimizer),
                'dry_dataset': _with_default(
                    r.parent_session.dry_datasets, r.parent_training.dry_default_datasets),
                'dry_metric_func': _with_default(
                    r.parent_session.dry_make_param_metric, r.parent_training.dry_default_make_param_metric),
            })


        # these are the properties for sure to have from a ModelTrainee database

        df = pd.DataFrame(basic).astype({'description': 'string', 'sess_name': 'string', 'training_status': 'string'})
        df.set_index('rec_id', inplace=True)
        objs = pd.DataFrame(objs)
        objs.set_index('rec_id', inplace=True)

        return cls(df, objs)

    @classmethod
    def from_query_df(cls, qdf: pd.DataFrame, model_analyzer: Union[Type[disRNN_model_analyzer], Callable[..., disRNN_model_analyzer]] = disRNN_model_analyzer):
        """Instantiation from a query dataframe
        
        Intake data via pandas reading SQL query can be much faster than iterating over list of SQLalchemy ORM instances
        
        :param qdf: query dataframe
        :type qdf: pd.DataFrame
        :param model_analyzer: the model analyzer class used to initialize analyzer for each model
        :type model_analyzer: Union[Type[disRNN_model_analyzer], Callable[..., disRNN_model_analyzer]]
        """
        df = (
            qdf
            .rename(columns={
                'recordID': 'rec_id',})
            .set_index('rec_id', drop=True)
        )

        status = df.apply(lambda r: _det_train_status_df(r['mt_worker'], r['total_steps'], r['chk_step']), axis=1)

                
        optims = df.apply(lambda r: _with_default(r['optimizer'], r['default_optimizer']), axis=1).rename('dry_optimizer')
        ds = df.apply(lambda r: _with_default(r['datasets'], r['default_datasets']), axis=1).rename('dry_dataset')
        metric = df.apply(lambda r: _with_default(r['make_param_metric'], r['default_make_param_metric']), axis=1).rename('dry_metric_func')

                
        md_analyzer = df.apply(lambda r: model_analyzer(
                make_network = _pkl_instantiate(
                    r['eval_model'] if r['eval_model'] is not None else r['model']).model_haiku, 
                params = r['parameter'],
            ), axis = 1) # type: ignore
        
        basic = df[['training_id', 'step', 'train_metric', 'test_metric', 'init_rand_seed', 'description', 'sess_name','sess_index']].assign(training_status=status).astype({'description': 'string', 'sess_name': 'string', 'training_status': 'string'})

        objs = df[['model']].assign(
            dry_optimizer = optims,
            dry_dataset = ds,
            dry_metric_func = metric,
            md_analyzer = md_analyzer,
            dry_training_step = df['make_train_step'],
        )

        return cls(basic, objs)

    def __repr__(self):
        strs_ = [
            super().__repr__(),
            f"N of models: {self.df.shape[0]}",
            f"N of training series: {len(self.df['training_id'].unique())}",
        ]
        return "\n".join(strs_)


    # for these extraction methods, I need to care about:
    # 1. whether these methods can be re-run to replace old calculation
    # 2. whether they assume an aligned `df` and `objs` dataframe

    def e_model_architecture(self):

        self.df['n_latents'] = self.objs['md_analyzer'].apply(lambda x: x.n_latents)
        self.df['update_mlp_shape'] = self.objs['md_analyzer'].apply(lambda x: x.update_mlp_shape)
        self.df['choice_mlp_shape'] = self.objs['md_analyzer'].apply(lambda x: x.choice_mlp_shape)
    
        return self

    def e_model_topology(self, bn_sigma_thre=0.4, cal_isomorphic: bool = True, ):
        for m in self.objs['md_analyzer']:
            m.set_bn_sigma_thre(bn_sigma_thre)

        self._set_forward_funcs()
        
        if cal_isomorphic:
            print("calculate model graphs...")
            self.objs['graph'] = [m.model_graph() for m in self.objs['md_analyzer']]
            self.df['graph_grp'] = pd.Series(group_isomorphic(self.objs['graph']), index = self.objs.index)

        self.df['N_open_lat_bn'] = self.objs['md_analyzer'].apply(lambda x: len(x.np.open_latent()))
        self.df['N_open_ch_bn'] = self.objs['md_analyzer'].apply(lambda x: len(x.np.open_choice_latent()))
        self.df['N_open_upd_bn'] = self.objs['md_analyzer'].apply(lambda x: jax.device_get(x.np.N_open_update_bn)) # make sure it's on cpu
        self.df['N_max_lat2lat_update_bn'] = self.objs['md_analyzer'].apply(lambda x: jax.device_get(x.np.N_max_lat2lat_update_bn))
        

        # ========= calculate graph fullness
        obs_dims = self.objs['md_analyzer'].apply(lambda x: x.np.obs_dim)
        graph_nodes = pd.concat([
            obs_dims + self.df['N_open_lat_bn'], # layer 1 and 2: update edges
            self.df['N_open_lat_bn'], 
            pd.Series(1, index=self.df.index, name = "N_chMLP") # layer 2 and 3: choice edges
            ], axis=1).reindex(self.df.index).values

        # how many edges for a fully connected graph given the number of nodes in each layer
        N_full_edges = jax.device_get(jax.vmap(N_edge_full_conn)(graph_nodes)) 

        # actual number of edges
        edges = (self.df['N_open_upd_bn'] + self.df['N_open_ch_bn']).values

        self.df['graph_fullness'] = edges/N_full_edges # all follows self.df index order
        return self
        
    def e_training_parameters(self):

        self.df['optimizer'] = self.objs['dry_optimizer'].apply(db_scrapper.optimizer)
        self.df['optimizer_lr'] = self.objs['dry_optimizer'].apply(db_scrapper.lr)
        self.df['disrnn_penalty_scale'] = self.objs['dry_training_step'].apply(db_scrapper.disrnn_penalty_scale)
        self.df['disrnn_beta_scale'] = self.objs['dry_training_step'].apply(db_scrapper.disrnn_beta_scale)
        self.df['disrnn_theta_scale'] = self.objs['dry_training_step'].apply(db_scrapper.disrnn_theta_scale)

        self.df['bn_penalty_lat'] = self.df['disrnn_penalty_scale']
        self.df['bn_penalty_upd'] = self.df['disrnn_penalty_scale'] * self.df['disrnn_beta_scale']
        self.df['bn_penalty_ch'] = self.df['disrnn_penalty_scale'] * self.df['disrnn_theta_scale']

        return self
    
    def _set_forward_funcs(self):
        """consolidate forward functions for identical models
        when the disrnn analyzers was initialized for each parameter associated models, all the analyzers have different jitted forward function. Call this function will first group all the models by their static attribute (assuming models with same attributes are identical) and set same jitted forward function to each group. This will dramatically speed up forward-related calculations by reusing already jitted forward function from other models in the same group.
        """

        for key, group in itertools.groupby(self.objs['md_analyzer'], _disrnn_static_attrs):
            item = next(group)

            tfd = transform_hkRNN(item.make_network)
            jf = jax.jit(tfd.apply)

            tfd2 = transform_hkRNN(item.trimmed.make_network)
            jf2 = jax.jit(tfd2.apply)

            item._forward_func = jf
            item.trimmed._forward_func = jf2
            for item in group: # working on remaining items in the group
                item._forward_func = jf
                item.trimmed._forward_func = jf2

    def c_likelihood(self, 
            trimmed_md_ll: bool = True, 
            store_in_column: str|None = None, 
            datasets: pd.Series | None = None,
            ll_exponentiate: bool = False,
            ll_normalize: bool = False,
            bn_sigma_thre: None|float = None, 
            cache: None|Cache = None, 
            cache_descr: None|str = None
        ):
        """calculate likelihood and store it in self.df

        Args:
            trimmed_md_ll (bool, optional): _description_. Defaults to True.
            k_type: how to calculate number of parameters
            store_in_column (str, optional): _description_. Defaults to "BIC".
            datasets (pd.Series | None, optional): a pandas series of datasets keyed by rec_id. Defaults to None which means to look it up from self.objs['dataset'].
            bn_sigma_thre (None | float, optional): Optionally reset `bn_sigma_thre` for all models before calculating BIC. Defaults to None.
            cache (None | Cache, optional): _description_. Defaults to None.
            cache_descr (None | str, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            AttributeError: _description_

        Returns:
            _type_: _description_
        """        

        # =========== verifying inputs
        if datasets is not None:
            datasets = datasets.reset_index(drop=True).reindex(self.objs.index)
        elif 'dataset' in self.objs.columns:
            datasets = self.objs['dataset']
        else:
            raise ValueError("cannot find `dataset` column in self.objs nor function argument. Please make sure it exists")
        if datasets.hasnans:
            warn("the datasets series has NA, please make sure it doesn't")
        
        if not hasattr(self.objs['md_analyzer'].iloc[0], 'likelihood'):
            raise AttributeError("please make sure `md_analyzer` instances have method `likelihood`")

        if bn_sigma_thre is not None:
            self.objs['md_analyzer'].apply(lambda x: x.set_bn_sigma_thre(bn_sigma_thre))
        
        if trimmed_md_ll:
            mdas = self.objs['md_analyzer'].apply(lambda x: x.trimmed)
        else:
            mdas = self.objs['md_analyzer']

        self._set_forward_funcs()
        
        if cache is not None:
            print("calculating likelihood (with cache)...")

            if trimmed_md_ll:
                @cache.memoize(ignore=(0, 'disrnn_analyzer', 1, 'ds'), tag=cache_descr)
                def cachable_ll1(disrnn_analyzer, ds, rec_id, descr, bn_sigma_thre, trimmed_md_ll, normalize, exponentiate):
                    """arguments not used is for cache key"""
                    return float(disrnn_analyzer.likelihood(ds, normalize=normalize, exponentiate=exponentiate))
                
                lls = [
                    cachable_ll1(
                        mdas.iloc[i], 
                        datasets.iloc[i], 

                        rec_id=self.objs.index[i], 
                        descr=cache_descr, 
                        bn_sigma_thre=self.objs['md_analyzer'].iloc[i].bn_sigma_thre,
                        trimmed_md_ll = trimmed_md_ll,
                        normalize = ll_normalize,
                        exponentiate = ll_exponentiate,) 
                    for i in tqdm(range(self.objs.shape[0]))]
            else:
                @cache.memoize(ignore=(0, 'disrnn_analyzer', 1, 'ds'), tag=cache_descr)
                def cachable_ll2(disrnn_analyzer, ds, rec_id, descr, normalize, exponentiate):
                    """arguments not used is for cache key"""
                    return float(disrnn_analyzer.likelihood(ds, normalize=normalize, exponentiate=exponentiate))
                
                lls = [
                    cachable_ll2(
                        mdas.iloc[i], 
                        datasets.iloc[i], 

                        rec_id=self.objs.index[i], 
                        descr=cache_descr, 
                        normalize = ll_normalize,
                        exponentiate = ll_exponentiate,) 
                    for i in tqdm(range(self.objs.shape[0]))]
            
        else:
            print("calculating likelihood...")
            
            lls = [
                float(mdas.iloc[i].likelihood(datasets.iloc[i], exponentiate = ll_exponentiate, normalize = ll_normalize))
                for i in tqdm(range(self.objs.shape[0]))]

        res = pd.Series(lls, index = self.objs.index) 

        if store_in_column is not None:
            self.df[store_in_column] = res # make sure it is aligned

        return res

    def c_BIC(self, 
            trimmed_md_ll: bool = True, 
            k_type: Literal["IB", "params", "capacity"] = "params",
            store_in_column: str|None = "BIC", 
            datasets: pd.Series | None = None,
            bn_sigma_thre: None|float = None, 
            cache: None|Cache = None, 
            cache_descr: None|str = None
        ):
        """calculate BIC and store it in self.df

        Args:
            trimmed_md_ll (bool, optional): _description_. Defaults to True.
            k_type: how to calculate number of parameters
            col_name (str, optional): _description_. Defaults to "BIC".
            datasets (pd.Series | None, optional): a pandas series of datasets keyed by rec_id. Defaults to None which means to look it up from self.objs['dataset'].
            bn_sigma_thre (None | float, optional): Optionally reset `bn_sigma_thre` for all models before calculating BIC. Defaults to None.
            cache (None | Cache, optional): _description_. Defaults to None.
            cache_descr (None | str, optional): _description_. Defaults to None.

        Raises:
            ExecError: _description_
            AttributeError: _description_

        Returns:
            _type_: _description_
        """        

        # =========== verifying inputs
        if datasets is not None:
            datasets = datasets.reindex(self.objs.index)
        elif 'dataset' in self.objs.columns:
            datasets = self.objs['dataset']
        else:
            raise ValueError("cannot find `dataset` column in self.objs nor function argument. Please make sure it exists")
        if datasets.hasnans:
            warn("the datasets series has NA, please make sure it doesn't")
        
        if not hasattr(self.objs['md_analyzer'].iloc[0], 'likelihood'):
            raise AttributeError("please make sure `md_analyzer` instances have method `likelihood`")

        if bn_sigma_thre is not None:
            self.objs['md_analyzer'].apply(lambda x: x.set_bn_sigma_thre(bn_sigma_thre))
        
        if trimmed_md_ll:
            mdas = self.objs['md_analyzer'].apply(lambda x: x.trimmed)
        else:
            mdas = self.objs['md_analyzer']

        self._set_forward_funcs()

        lls = self.c_likelihood(
            trimmed_md_ll=trimmed_md_ll,
            datasets=datasets,
            ll_exponentiate = False,
            ll_normalize = False,
            bn_sigma_thre=bn_sigma_thre,
            cache=cache,
            cache_descr=cache_descr
        )

        tmpdf = mdas.to_frame('md_analyzer').join(datasets.rename('dataset')).join(lls.rename('ll'))

        bics = tmpdf.apply(lambda r: r['md_analyzer'].BIC(r['dataset'], likelihood = r['ll'], k_type = k_type), axis = 1)
        
        # if cache is not None:
        #     print("calculating likelihood (with cache)...")

        #     @cache.memoize(ignore=(0, 'disrnn_analyzer', 1, 'ds'))
        #     def cachable_BIC(disrnn_analyzer, ds, rec_id, descr, bn_sigma_thre, trimmed_md_ll, k_type):
        #         """arguments not used is for cache key"""
        #         return float(disrnn_analyzer.BIC(ds, k_type = k_type))
            
        #     bics = [
        #         cachable_BIC(
        #             mdas.iloc[i], 
        #             datasets.iloc[i], 

        #             rec_id=self.objs.index[i], 
        #             descr=cache_descr, 
        #             bn_sigma_thre=self.objs['md_analyzer'].iloc[i].bn_sigma_thre,
        #             trimmed_md_ll = trimmed_md_ll,
        #             k_type=k_type) 
        #         for i in tqdm(range(self.objs.shape[0]))]
            
        # else:
        #     print("calculating likelihood...")
            
        #     tmpdf = self.objs[['dataset']].join(mdas)
        #     ll = tmpdf.progress_apply(lambda r: r['md_analyzer'].likelihood(r['dataset']), axis=1).apply(jax.device_get).values # type: ignore
        #     N_data = tmpdf['dataset'].apply(lambda x: x.n_observations).values

        #     print("calculating effective number of params...")
        #     N_param = self.objs['md_analyzer'].progress_apply(lambda x: x.N_effective_params).apply(jax.device_get).values

        #     bics = jax.device_get(jax.vmap(bic)(N_param, N_data, ll))

        res = pd.Series(bics, index = self.objs.index) 

        if store_in_column is not None:
            # make sure it is aligned
            self.df[store_in_column] = res

        return res





from typing import Sequence
from sqlalchemy import tuple_, case
def get_trec_from_id_step(sess: Session, ids: Sequence[int], steps: Sequence[int]):
    '''get list of trainingRecords by list of ids and steps (and make sure order is kept)
    
    However, the ordering will be slow
    '''

    assert len(ids) == len(steps), f"length of `ids` ({len(ids)}) and `steps` ({len(steps)}) does not match"

    pairs = list(zip(ids, steps))

    order_case = case(
        *[
            (and_(trainingRecord.training_id == tid,
                    trainingRecord.step == st), i)
            for i, (tid, st) in enumerate(pairs)
        ],
        else_=len(pairs)
    )

    recs = (
        sess.query(trainingRecord)
        .filter(tuple_(trainingRecord.training_id, trainingRecord.step).in_(pairs))
        .order_by(order_case)
        .all()
    )
    return recs

def sharing_jitted_forward_func(mdas: Sequence[disRNN_model_analyzer]):
    for key, group in itertools.groupby(mdas, _disrnn_static_attrs):
        item = next(group)

        tfd = transform_hkRNN(item.make_network)
        jf = jax.jit(tfd.apply)

        tfd2 = transform_hkRNN(item.trimmed.make_network)
        jf2 = jax.jit(tfd2.apply)

        item._forward_func = jf
        item.trimmed._forward_func = jf2
        for item in group: # working on remaining items in the group
            item._forward_func = jf
            item.trimmed._forward_func = jf2
    
    return None



import sqlalchemy.orm as sao
import sqlalchemy as sa

def get_dry_model_df(conn: sa.Connection | sa.Engine, ids, steps):
    """return dataframe with columns:
    - training_id
    - step
    - parameter
    - dry_eval_model
    """
    q = (
        sa.select(
            trainingRecord.training_id, trainingRecord.step, 
            trainingRecord.parameter, 
            sa.func.coalesce(ModelTrainee.dry_eval_model, ModelTrainee.dry_model).label('dry_eval_model')
        )
        .join(trainingRecord.parent_training)
        .where(
            sa.tuple_(
                trainingRecord.training_id, 
                trainingRecord.step
            )
            .in_(zip(ids, steps))
        )
    )
    return pd.read_sql_query(q, conn)


def inst_model_df_from_db(
        conn: sa.Connection | sao.Session, 
        query_df: pd.DataFrame, 
        model_analyzer: Union[Type[disRNN_model_analyzer], Callable[..., disRNN_model_analyzer]] = disRNN_model_analyzer, 
        mt_id_col = 'training_id', step_col = 'step',
    ):
    """Pull instantiated disRNN models (analyzer) from database into a 'model' column
    
    Identifying models by combination of model training id and step

    Return the query_df with additional `model` column

    :param conn: sqlalchemy Connection
    :type conn: sa.Connection
    :param query_df: dataframe about the models to be pulled
    :type query_df: pd.DataFrame
    :param model_analyzer: model analyzer class or initiating function
    :type model_analyzer: Union[Type[disRNN_model_analyzer], Callable[..., disRNN_model_analyzer]]
    :param mt_id_col: column name for training id
    :param step_col: column name for step
    """

    _conn = conn.get_bind() if isinstance(conn, sao.Session) else conn

    md_df = get_dry_model_df(_conn, query_df[mt_id_col].to_list(), query_df[step_col].to_list())
    md_df['model'] = md_df.progress_apply(
        lambda row: model_analyzer(_pkl_instantiate(row['dry_eval_model']).model_haiku, row['parameter']), 
        axis=1
    ) # type: ignore
    
    sharing_jitted_forward_func(md_df['model'])

    return query_df.merge(md_df[[mt_id_col, step_col, 'model']], how='inner', on = [mt_id_col, step_col])