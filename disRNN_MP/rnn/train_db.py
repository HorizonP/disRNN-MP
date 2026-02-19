import os
from pathlib import Path
from typing import Callable, List, Optional, Union, Tuple
import datetime
import dataclasses
import signal
import logging
import time
import uuid
import base64
import json
from warnings import warn
from copy import deepcopy
from abc import ABC, abstractmethod

import cloudpickle
from disRNN_MP.rnn.pkl_instantiate import _pkl_instantiate
from disRNN_MP.utils import isequal_pytree
import numpy as np
import optax
import jax
import flax
from hydra.utils import instantiate
# from jax._src.lib import xla_client

from sqlalchemy import ForeignKey, create_engine, func, select, Index
from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass, Mapped, mapped_column, relationship, composite, Session
from sqlalchemy.types import TypeDecorator, LargeBinary, PickleType, JSON
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.ext.associationproxy import association_proxy, AssociationProxy

from disRNN_MP.dataset import trainingDataset
from disRNN_MP.rnn.utils import RandomKey, Params, Inputs, Outputs, TrainStepFun, Loss, OptState, nan_in_dict
from chex import ArrayTree # needed for sqlAlchemy to determine trainingState type

class GracefulKiller:
    """A utility class to gracefully handle termination signals (SIGINT, SIGTERM) in Python applications.
    Ref: https://stackoverflow.com/a/31464349
    ### Attributes:
        kill_now (bool): A flag indicating whether a termination signal has been received.

    ### Methods:
        __init__(): Initializes the signal handler for SIGINT and SIGTERM.
        exit_gracefully(sigNum, frame): Sets the kill_now flag to True upon receiving a signal and logs the event.

    Example:
        # Example usage of GracefulKiller
        import time

        killer = GracefulKiller()
        while not killer.kill_now:
            time.sleep(1)
            print("Performing task...")

        print("Termination signal received. Exiting gracefully.")

    Note:
        This class requires the 'signal' and 'logging' modules from the Python Standard Library.
    """
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, sigNum, frame):
        """
        Sets the kill_now flag to True and logs the signal reception with timestamp.

        Parameters:
            sigNum (int): The signal number received.
            frame: The current stack frame (ignored in this implementation).

        """
        _msg = f"[{datetime.datetime.now().isoformat(timespec='milliseconds')}]: received signal {sigNum}" 
        logging.info(_msg)
        print(_msg)
        self.kill_now = True

# ======== custom types
    
class PyTreeType(TypeDecorator):
    """
    Custom SQLAlchemy type to serialize/deserialize PyTrees using Flax's msgpack_serialize. Leaves will be restored to numpy
    """
    impl = LargeBinary

    def process_bind_param(self, value, dialect):
        if value is not None:
            # Serialize the PyTree using Flax's msgpack_serialize
            return flax.serialization.msgpack_serialize(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            # Deserialize the PyTree
            return flax.serialization.msgpack_restore(value)
        return value
    
    def compare_values(self, x, y):
        return isequal_pytree(x, y)
    
class JaxPyTreeType(TypeDecorator):
    """
    Custom SQLAlchemy type to serialize/deserialize PyTrees using Flax's msgpack_serialize. Leaves will be restored to jax
    """
    impl = LargeBinary

    def process_bind_param(self, value, dialect):
        if value is not None:
            # Serialize the PyTree using Flax's msgpack_serialize
            return flax.serialization.msgpack_serialize(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            # Deserialize the PyTree
            return jax.device_put(flax.serialization.msgpack_restore(value))
        return value
    
    def compare_values(self, x, y):
        return isequal_pytree(x, y)


def encode_base64_cloudpickle(obj):
    # This function is called for objects that aren't serializable by default
    return 'base64::' + base64.b64encode(cloudpickle.dumps(obj)).decode()

def decode_base64_cloudpickle(val:str):
    return cloudpickle.loads(base64.b64decode(val[8:]))

def base64_cloudpickle_hook(obj:dict):
    for key, val in obj.items(): 
        if isinstance(val, str) and val.startswith('base64::'):
            obj[key] = decode_base64_cloudpickle(val)
    return obj

JSON_ELE_TYPES = Union[str, int, float, bool, None]

class JSONWithCloudPickle(TypeDecorator):
    """
    A custom JSON serializer using cloudpickle for complex Python objects.

    This class extends TypeDecorator to provide custom serialization and deserialization
    for storing complex Python objects as JSON, using cloudpickle for objects that aren't
    natively serializable by JSON.

    ### Usage:
    - Use as a column type in SQLAlchemy models to store complex Python objects in a database.
    
    ### Notes:
    - JSON_ELE_TYPES define native JSON serializable types.
    - `jax.tree_util.tree_map` is used to efficiently traverse and process the objects.
    """

    impl = JSON

    def process_bind_param(self, value, dialect):
        # return json.dumps(value, default=encode_base64_cloudpickle)
        return jax.tree_util.tree_map(
            f = lambda v: encode_base64_cloudpickle(v) if not isinstance(v, JSON_ELE_TYPES) else v, 
            tree = value,
            is_leaf = lambda node: not isinstance(node, Union[dict, list]))

    def process_result_value(self, value, dialect):
        # return json.loads(value, object_hook=base64_cloudpickle_hook) if value else value
        return jax.tree_util.tree_map(lambda v: decode_base64_cloudpickle(v) if isinstance(v, str) and v.startswith('base64::') else v, value)


# ======== mapped classes for ModelTrainee and related

_JSON_type = JSONWithCloudPickle # the JSON type used for ModelTrainee and related classes

@dataclasses.dataclass(frozen=True)
class trainingState:
    """
    executed step count from 1.
    step 0 is the init params
    """
    step: int
    params: Params
    opt_state: optax.OptState
    rand_key: RandomKey


class Base(DeclarativeBase):
    pass

def _wrap_obj_in_target(obj) -> dict:
    return {
        '_target_': lambda: obj
    }

def _optionally_assign_wrapped(cls: object, attr: str, val):
    if val is not None:
        if getattr(cls, attr) is not None:
            warn(f'provided init argument {val} (in a wrapper) will override attr {attr} of object {cls}')
        
        if isinstance(val, dict) and '_target_' in val:
            # if provided val is already a hydra-instantiable dictionary, directly assign the value
            setattr(cls, attr, val)
        else:
            # otherwise, wrap it in a hydra-instantiable dictionary and assign the value
            setattr(cls, attr, _wrap_obj_in_target(val))

class stopCritTest(ABC):
    """an abstract class to represent flexible training stop criteria for ModelTrainee
    """
    def bind(self, mt:'ModelTrainee') -> None:
        self.trainee = mt

    @abstractmethod
    def test(self) -> bool:
        """test if the training should stop now
        True - stop now
        """
        pass

class ModelTrainee(Base):
    """the parent class for the training class, store bundle id and state
    
    Attributes:
        - train_options: an optional dictionary containing additional parameters for the `train` method
        - `model` and `eval_model`: anything that have an apply function
    """
    __tablename__ = 'training'
    id: Mapped[int] = mapped_column(primary_key=True) # will auto-increment

    sessions: Mapped[List["trainingSession"]] = relationship(
        back_populates='parent_training',
        primaryjoin="ModelTrainee.id==trainingSession.training_id",
        order_by='trainingSession.index', # ensure list is ordered
        collection_class=ordering_list('index'), # ensure list is ordered
        cascade="all, delete, delete-orphan",
    )
    records: Mapped[List["trainingRecord"]] = relationship(
        back_populates='parent_training',
        order_by='trainingRecord.index',
        collection_class=ordering_list('index'),
        cascade="all, delete, delete-orphan",
    )
    loss_trace: Mapped[List["trainingLoss"]] = relationship(
        back_populates='parent_training',
        order_by='trainingLoss.step',
        cascade="all, delete, delete-orphan",
    )
    parent_fork: Mapped['trainingFork'] = relationship(
        back_populates='child_training',
        cascade="all, delete, delete-orphan",
    )

    dry_model: Mapped[dict] = mapped_column('model', _JSON_type)
    dry_eval_model: Mapped[Optional[dict]] = mapped_column('eval_model', _JSON_type)
    init_rand_seed: Mapped[int] = mapped_column(default=0) # the random seed to generate init params
    name: Mapped[str] = mapped_column(default = lambda: f"trainer created at {datetime.datetime.now().isoformat(timespec='milliseconds')}")
    description: Mapped[Optional[str]]

    dry_default_datasets: Mapped[Optional[dict]] = mapped_column('default_datasets', _JSON_type)
    dry_default_optimizer: Mapped[Optional[dict]] = mapped_column('default_optimizer', _JSON_type)
    dry_default_make_param_metric: Mapped[Optional[dict]] = mapped_column('default_make_param_metric', _JSON_type)


    worker: Mapped[Optional[str]] = mapped_column(default=None)
    fork_from_record: AssociationProxy['trainingRecord'] = association_proxy(
        'parent_fork', 'forkRecord',
    )    
    # states
    chk_step: Mapped[Optional[int]] = mapped_column(nullable=True, )

    state: Mapped[trainingState] = composite(
        'chk_step',
        mapped_column('chk_params', JaxPyTreeType, nullable=True),
        mapped_column('chk_opt_state', PickleType, nullable=True),
        mapped_column('chk_rand_key', JaxPyTreeType, nullable=True),
    )

    def __init__(self, 
            model = None, 
            eval_model = None, 
            default_datasets = None,
            default_optimizer = None,
            default_make_param_metric = None,
            **kwargs):
        
        # Call the default __init__ provided by SQLAlchemy
        super().__init__(**kwargs)

        _optionally_assign_wrapped(self, 'dry_model', model)
        _optionally_assign_wrapped(self, 'dry_eval_model', eval_model)

        _optionally_assign_wrapped(self, 'dry_default_optimizer', default_optimizer)
        _optionally_assign_wrapped(self, 'dry_default_make_param_metric', default_make_param_metric)
        _optionally_assign_wrapped(self, 'dry_default_datasets', default_datasets)

    @hybrid_property
    def total_steps(self): # type: ignore
        # Instance-level implementation (Python)
        # This will be used when you access ModelTrainee.total_steps on an instance of ModelTrainee
        return sum(session.n_step for session in self.sessions)

    @total_steps.expression
    def total_steps(cls):
        # Class-level implementation (SQL)
        # This will be used when you query ModelTrainee.total_steps in a SQLAlchemy query
        subquery = (
            select(func.sum(trainingSession.n_step))
            .select_from(trainingSession)
            .where(trainingSession.training_id == cls.id)
            .correlate(cls)
        ).scalar_subquery()

        return subquery.label('total_steps')
    
    @classmethod
    def clone(cls, ori: 'ModelTrainee') -> 'ModelTrainee':
        """clone the definition of a ModelTrainee and associated trainingSessions to a new instance

        Args:
            ori (ModelTrainee): original ModelTrainee instance

        Returns:
            ModelTrainee: cloned
        """
        
        trainee_kwargs = ori.asdict()

        # modify description
        msg = f'[cloned from training.id={ori.id}] '
        trainee_kwargs['description'] = (
            msg if 'description' not in trainee_kwargs or trainee_kwargs['description'] is None else msg + trainee_kwargs['description']
        )

        # copy ModelTrainee definitions
        new = cls(**trainee_kwargs)

        # copy trainingSession definitions
        for se in ori.sessions:
            new.sessions.append(trainingSession(**se.asdict()))
        
        return new

    @classmethod
    def fork_from(cls, ori:'trainingRecord') -> "ModelTrainee":
        """create a new ModelTrainee instance forked from a record

        this function will create the new ModelTrainee instance by copying all configurations returned by `ModelTrainee.asdict` method on `ori.parent_training`.
        the checkpoint state of the isntance will be set as step 1 and copying parameters, opt_state, and random keys if exists

        Args:
            ori (trainingRecord): the record forked from

        Returns:
            ModelTrainee: the forked new instance of ModelTrainee
        """
        trainee_kwargs = ori.parent_training.asdict()

        msg = f'[forked from training.id={ori.parent_training.id}, history.RecordID={ori.id}] '

        trainee_kwargs['description'] = (
            msg if 'description' not in trainee_kwargs or trainee_kwargs['description'] is None else msg + trainee_kwargs['description']
        )

        new = cls(state = dataclasses.replace(ori.state, step = 1), **trainee_kwargs)
        new.parent_fork = trainingFork(forkRecord = ori)

        return new

    @classmethod
    def delete(cls, sql_sess:Session, trainee_id:int|List[int]|Tuple[int]):
        """delete a ModelTrainee and its related instances
        if trainee_id is list or tuple, consider each element as a trainee_id
        """
        if isinstance(trainee_id, (Tuple, List)):
            sql_sess.query(trainingLoss).where(trainingLoss.training_id.in_(trainee_id)).delete()
            sql_sess.query(trainingRecord).where(trainingRecord.training_id.in_(trainee_id)).delete()
            sql_sess.query(trainingSession).where(trainingSession.training_id.in_(trainee_id)).delete()
            sql_sess.query(ModelTrainee).where(ModelTrainee.id.in_(trainee_id)).delete()
            sql_sess.query(trainingFork).where(trainingFork.id.in_(trainee_id)).delete()
        else:
            sql_sess.query(trainingLoss).where(trainingLoss.training_id == trainee_id).delete()
            sql_sess.query(trainingRecord).where(trainingRecord.training_id == trainee_id).delete()
            sql_sess.query(trainingSession).where(trainingSession.training_id == trainee_id).delete()
            sql_sess.query(ModelTrainee).where(ModelTrainee.id == trainee_id).delete()
            sql_sess.query(trainingFork).where(trainingFork.id == trainee_id).delete()

        sql_sess.commit()

    def asdict(self):
        """retrieve re-initializable items of the instance as a dictionary"""
        keys = ['dry_model', 'dry_eval_model', 'init_rand_seed', 'description', 'dry_default_datasets', 'dry_default_optimizer', 'dry_default_make_param_metric']

        return deepcopy({k: getattr(self, k) for k in keys})

    def materialize(self, alt_cwd: str | Path | None = None):
        """materialize the ModelTrainee instance

        the ModelTrainee class stores its definitions and serialized data in a database (much like a dried flower). `materialize` will convert these definitions and serialized data back into python objects that lived in the memory

        Args:
            alt_cwd (str | Path | None, optional): if this is provided, the instance will be materialized at a different working directory. This will mainly affect loading data from disk, since most of them are stored as a relative path to a root folder (which is the working directory). Defaults to None.

        Raises:
            ValueError: when no session definition has been attached to the instance
        """        
        if alt_cwd:
            cwd = os.getcwd()
            os.chdir(alt_cwd)
        # instantiate models: self.model, self.eval_model
        self._instantiates()

        # instantiate training session functions
        if len(self.sessions) == 0:
            raise ValueError('there is no session for initialization')

        for se in self.sessions:
            se._instantiates()

        # initialize state
        if self.state is None or self.state.step is None:
            if self.fork_from_record is None:
                self._init_states()
            else:
                self._init_states(init_params=self.fork_from_record.parameter)

        if alt_cwd:
            os.chdir(cwd)
        

    def _instantiates(self):
        self.model = _pkl_instantiate(self.dry_model)
        if self.dry_eval_model is None:
            self.eval_model = self.model
        else:
            self.eval_model = _pkl_instantiate(self.dry_eval_model)
    

    def _init_states(self, init_params: Optional[Params] = None, init_opt_state: Optional[optax.OptState] = None):
        """generate initial states with `init_rand_seed`
        
        - make the very first `self.state`
        - has to be called after sessions has been added to the ModelTrainee
        - if provided any of init_params and init_opt_state, will use them as initial state
        - this function use `jax`
        """

        init_rand_key = jax.random.PRNGKey(self.init_rand_seed)
        se = self.sessions[0]

        if init_params is None:
            init_params = jax.jit(self.model.init)(init_rand_key, se.train_dataset.xs)

        if init_opt_state is None:
            init_opt_state = se.optimizer.init(init_params)

        self.state = trainingState(0, init_params, init_opt_state, init_rand_key) # type: ignore

    def append_record(self, rec: 'trainingRecord'):
        '''make sure reference among ModelTrainee, trainingSession, and trainingRecord are correct'''
        se_ind = self.curr_sess_ind
        if se_ind is None:
            raise ValueError('record cannot be added since training has not begin (no current session)')
        else:
            rec.parent_session = self.sessions[se_ind]
            rec.worker = self.worker
            self.records.append(rec)

    @property
    def curr_sess_ind(self) -> Optional[int]:
        """determine current trainingSession index by `self.state.step`

        Returns:
            int: index for self.sessions
        """
        n_steps = [se.n_step for se in self.sessions]
        if len(n_steps) == 0:
            raise ValueError('there has not been any sessions')
        else:
            ind:int = np.searchsorted(np.cumsum([0, *n_steps]), self.state.step) - 1 # type: ignore
            if ind < 0: # -1
                return None
            else:
                return ind


    def next_session(self, sess: Optional['trainingSession'] = None) -> Optional['trainingSession']:
        """get next session of a given session

        Args:
            sess (trainingSession): [optional] The current session whose next session is looked for. 
            If not given, deduce current session by current step
        Returns:
            None: when there's no next session
            trainingSession: next session
        """

        if sess is None:
            se_ind = self.curr_sess_ind
        else:
            if sess.parent_training is not self:
                raise ValueError('the provided trainingSession is not part of this ModelTrainee instance')

            se_ind = sess.index
        
        if se_ind is None:
            # when training is not begined yet, next session is first session
            return self.sessions[0]
        elif se_ind >= len(self.sessions) - 1:
            # current session is the end, no next session
            return None
        else:
            return self.sessions[se_ind + 1]


    def train(self, 
            sql_sess: Optional[Session] = None, 
            worker: Optional[str] = None, 
            max_na_retry: int = 10,
            pause_unresolvable_NaN: bool = True,
            materialize: bool = True,
        ) -> int:
        """proceed the training as planned in training sessions
        
        determine the start point by comparing state to trainingSession

        Args:
            sql_sess: if provided a SQLAlchemy session object, the function will periodically update the database. it will also close the connection at the end
            worker: the name of the worker (for updating the column in database).
            max_na_retry: how many attempts of retry to make when NaN is encountered. value <= 1 means no retry at all
            pause_unresolvable_NaN: if true, this function will set the worker for self as "unresolvable NaNs" to prevent other workers from retrying it 

        return an integer:
            0 - finished all trainingSession of the current ModelTrainee
            1 - abort due to another worker working on the instance
            2 - abort due to unresolvable NaN in parameters
            3 - abort due to external signal (such as SIGINT, SIGTERM)
        """

        def _checkpoint_to_db(terminate = False, worker = None):
            if sql_sess is not None:
                if terminate:
                    # release this item in the database, so that other worker may continue working on it
                    self.worker = worker
                    sql_sess.add(self)
                    sql_sess.commit()
                    sql_sess.close()
                else:
                    sql_sess.add(self)
                    sql_sess.commit()


        if self.worker is not None and worker != self.worker:
            logging.warning('there is another worker working on training the same model, so aborted')
            return 1
        
        if sql_sess is not None:
            # make sure commit does not expire instance, otherwise the functions will gone
            sql_sess.expire_on_commit = False 

            # only write down the worker name when a SQL session is provided. Otherwise, the function cannot punctually update the worker, which defeat the purpose of this `worker` field
            if worker is None:
                # generate a unique random identifier
                worker = str(uuid.uuid1())

            self.worker = worker
        _checkpoint_to_db() # checkpoint to claim the job
        logging.info(f"current worker is {self.worker}")

        if materialize:
            try:
                self.materialize() # make sure i
            except Exception as e:
                _checkpoint_to_db(terminate=True, worker="error when materializing the instance")
                raise e
        
        # default value for sess_stop_criteria
        # if sess_stop_criteria is None:
        #     sess_stop_criteria = [None] * len(self.sessions)
        
        # sess_stop_criteria = _pkl_instantiate(sess_stop_criteria)

        # initialize gracefulKiller to handle termination of training
        sig_term = GracefulKiller()

        t_start = time.time()

        if self.curr_sess_ind is None:
            # the training started freshly
            curr_se = self.next_session()
        else:
            curr_se = self.sessions[self.curr_sess_ind]
            # if the current session is just ended, it will be handled in the next while loop

        # run each session
        while curr_se is not None and not sig_term.kill_now: # session loop
            logging.debug(f"current session id: {curr_se.id}")
            # initialize current session
            train_dataset, test_dataset = curr_se.train_dataset, curr_se.test_dataset
            # report_steps = np.arange(se_start[curr_se.index], se_end[curr_se.index], curr_se.steps_per_block) 

            criteria = curr_se.stop_criteria
            if isinstance(criteria, stopCritTest):
                criteria.bind(self)

            end_step = curr_se.end_step
            step = self.state.step + 1
            # next report step is last report step + steps per block
            report_step = curr_se.records[-1].step + curr_se.steps_per_block if len(curr_se.records) > 0 else step
            while step < end_step + 1: # step loop
            # for step in range(self.state.step + 1, se_end[curr_se.index] + 1):
                # step is next step
                # when current session is just ended, this for-loop will not run since range is empty

                # proceed training, and retry when NaN exists
                n_attempt = 0
                _nxt_rand_key = self.state.rand_key
                 # potentially retrying only when the program is not going to be killed
                while not sig_term.kill_now: # retry-step loop
                    _nxt_rand_key, _curr_rand_key = jax.random.split(_nxt_rand_key, 2)

                    # potentially random sampling the batch
                    train_dataset.set_seed(int(jax.random.choice(_curr_rand_key, 1e7)))
                    xs, ys = next(train_dataset)

                    loss, params, opt_state = curr_se.train_step(
                        self.state.params, _curr_rand_key, self.state.opt_state, xs, ys)
                    n_attempt += 1

                    if nan_in_dict({'loss': loss, 'par':params, 'optstate':opt_state}):
                        # tried debug the NaNs before, it turns out due to the optimizer sometimes generate nan gradient
                        if n_attempt < max_na_retry:
                            logging.warning(f'found NaN in params at step {step} attempt {n_attempt}')
                            continue
                        else:
                            logging.error(f'unresolvable NaN in params at step {step} after {n_attempt} attempts of repeat')
                            if pause_unresolvable_NaN:
                                status = "unresolvable NaNs"
                            else:
                                status = None
                            _checkpoint_to_db(terminate=True, worker=status)
                            return 2
                    
                    # after this line, there will be no NaN in params
                    # the following part has to be put inside the current while loop, because when the above retry is breaked by sig_term, the following database updating procedure should not be proceeded

                    self.loss_trace.append(trainingLoss(step = step, value = float(loss), parent_session = curr_se)) # type: ignore
                
                    # evolve to next state
                    self.state = trainingState(step, params, opt_state, _nxt_rand_key)

                    # log history if at certain steps
                    if step > report_step:
                        print(f"step: {step}, report_step: {report_step}")
                    elif step == report_step:
                        # I may offload these to another worker in different process
                        rec = trainingRecord(
                            # step = step, 
                            # parameter = params, 
                            state = trainingState(step, params, opt_state, _nxt_rand_key),
                            train_metric = float(curr_se.param_metric(params, train_dataset.xs, train_dataset.ys)), 
                            test_metric = float(curr_se.param_metric(params, test_dataset.xs, test_dataset.ys))
                        )

                        self.append_record(rec)
                        print(f"step {step} is done with loss: {loss:.4e} (Time: {time.time()-t_start:.1f}s)\r", end=None)

                        # check for potential worker conflict before record history
                        if sql_sess is not None:
                            sql_sess.refresh(self, ['worker'])
                            if self.worker != worker:
                                msg = f"[current worker: {worker}]conflict: there's another worker ({self.worker}) working on the same training"
                                print(msg)
                                # stop the training
                                sql_sess.close()
                                # raise WorkerConflictError(msg)
                                return 1                        
                            else:
                                _checkpoint_to_db()

                        if isinstance(criteria, stopCritTest):
                            if criteria.test():
                                # end the loop by set end_step to current step
                                end_step = step
                                # update session's n_step
                                curr_se.n_step = end_step - curr_se.start_step + 1
                                print("converged")
                            elif end_step - step <= curr_se.steps_per_block:
                                # extend planned n_step for current session
                                extend = curr_se.steps_per_block
                                end_step = end_step + extend
                                curr_se.n_step = curr_se.n_step + extend
                                print(f"not converge, update session n_step to {curr_se.n_step}")
                        
                        report_step = report_step + curr_se.steps_per_block
                    
                    break # continue to next step in the for-loop

                if sig_term.kill_now:
                    # handling training termination
                    _msg = f"[{datetime.datetime.now().isoformat(timespec='milliseconds')}]: terminated at step {step}" 
                    print(_msg)
                    break

                step = step + 1
            
            curr_se = self.next_session()

        _checkpoint_to_db(terminate=True)

        if sig_term.kill_now:
            return 3
        else:
            return 0

            


class trainingSession(Base):
    """specify a training session

    ## functional objects:
        - `optimizer`: optax optimizer instance (which has `init` and `update` method
        - `make_train_step`: make_train_step(model.apply, optimizer) -> train_step: Callable[[Params, RandomKey, OptState, Inputs, Outputs], Tuple[Loss, Params, OptState]]. Loss function should be included in here
        - `datasets`: (train_dataset, test_dataset) each of them should be of trainingDataset class
        - `make_param_metric`: make_param_metric(eval_model) -> param_metric: Callable[[Params, Inputs, Outputs], jax.Array]
        `train_step` and `param_metric` should be jax-jitable 
    
    ## dry attributes:
        - id
        - name
        - training_id & parent_training
        - records
        - step_losses
        - n_step
        - steps_per_block
        - dry_optimizer
        - dry_make_train_step
        - dry_make_param_metric
        - dry_datasets

    ## materialized attributes:
        - optimizer
        - train_dataset
        - test_dataset
        - train_step
        - param_metric
    """
    __tablename__ = 'schedule'
    id: Mapped[int] = mapped_column(primary_key=True) # will auto-increment
    index: Mapped[int]

    name: Mapped[str] = mapped_column(default=lambda: f"session created at {datetime.datetime.now().isoformat(timespec='milliseconds')}")
    training_id: Mapped[int] = mapped_column(ForeignKey('training.id'))
    parent_training: Mapped["ModelTrainee"] = relationship(
        foreign_keys=[training_id], 
        back_populates='sessions',
        primaryjoin="trainingSession.training_id==ModelTrainee.id",
    )

    records: Mapped[List["trainingRecord"]] = relationship(
        back_populates='parent_session',
        order_by='trainingRecord.index',
        cascade="all, delete, delete-orphan",
    ) # it's not supposed to add records from here, since there's no ordering_list

    step_losses: Mapped[List['trainingLoss']] = relationship(
        back_populates='parent_session',
        cascade="all, delete, delete-orphan",
    )

    dry_optimizer: Mapped[Optional[dict]] = mapped_column('optimizer', _JSON_type)
    dry_make_train_step: Mapped[dict] = mapped_column('make_train_step', _JSON_type)
    dry_make_param_metric: Mapped[Optional[dict]] = mapped_column('make_param_metric', _JSON_type)
    dry_datasets: Mapped[Optional[dict]] = mapped_column('datasets', _JSON_type)
    dry_stop_criteria: Mapped[Optional[dict]] = mapped_column('stop_criteria', _JSON_type)

    n_step: Mapped[int] # total number of steps 
    steps_per_block: Mapped[int] = mapped_column(default=20) # log every x step

    @property
    def start_step(self):
        """step id for the first step in the session"""
        return sum([se.n_step for se in self.parent_training.sessions[:self.index]]) + 1

    @property
    def end_step(self):
        """step id for the last step in the session"""
        return sum([se.n_step for se in self.parent_training.sessions[:self.index+1]])
    
    def __init__(self, optimizer = None, make_train_step = None, make_param_metric = None, datasets = None, **kwargs):
        
        # Call the default __init__ provided by SQLAlchemy
        super().__init__(**kwargs)

        _optionally_assign_wrapped(self, 'dry_optimizer', optimizer)
        _optionally_assign_wrapped(self, 'dry_make_train_step', make_train_step)
        _optionally_assign_wrapped(self, 'dry_make_param_metric', make_param_metric)
        _optionally_assign_wrapped(self, 'dry_datasets', datasets)

    def asdict(self, with_default = False):
        """retrieve re-initializable items of the instance as a dictionary.
        if `with_default` is True, populate the corresponding field with default value
        """
        if with_default is False:
            keys = ['dry_optimizer', 'dry_make_train_step', 'dry_make_param_metric', 'dry_datasets', 'n_step', 'steps_per_block', 'dry_stop_criteria']
            return deepcopy({k: getattr(self, k) for k in keys})
        else:
            self_keys = ['dry_datasets', 'dry_optimizer', 'dry_make_param_metric',]
            default_keys = ['dry_default_datasets', 'dry_default_optimizer', 'dry_default_make_param_metric']
            additional_keys = ['dry_make_train_step', 'n_step', 'steps_per_block', 'dry_stop_criteria']

            dic = {k: getattr(self, k) for k in additional_keys}
            for i in range(len(self_keys)):
                if getattr(self, self_keys[i]) is None:
                    print(f"{self_keys[i]} is populated from default value {default_keys[i]} of parent ModelTrainee")
                    dic[self_keys[i]] = getattr(self.parent_training, default_keys[i])
                else:
                    dic[self_keys[i]] = getattr(self, self_keys[i])            

            return deepcopy(dic)


    def _inst_with_default(self, attr, default_attr):
        """instantiate with trainingSession dry contents. If it's None, instantiate from parent's default dry contents"""
        default_val = getattr(self.parent_training, default_attr)
        val = getattr(self, attr)

        if val is None:
            if default_val is None:
                raise ValueError(
                    f"At least one of ModelTrainee.{default_attr} and trainingSession.{attr} has to be specified"
                )
            else:
                return _pkl_instantiate(default_val)
        else:
            return _pkl_instantiate(val)
    
    def _instantiates(self):
        """instantiate functions and instances of a trainingSession
        - `train_step`
        - `optimizer`
        - datasets
        - `param_metric`
        """

        make_train_step = _pkl_instantiate(self.dry_make_train_step)
        make_param_metric = self._inst_with_default("dry_make_param_metric", "dry_default_make_param_metric")
        datasets = self._inst_with_default("dry_datasets", "dry_default_datasets") 

        self.optimizer = self._inst_with_default("dry_optimizer", "dry_default_optimizer")
        self.train_dataset:trainingDataset = datasets[0]
        self.test_dataset:trainingDataset = datasets[1]
        self.train_step:TrainStepFun = jax.jit(make_train_step(self.parent_training.model.apply, self.optimizer))
        self.param_metric:Callable[[Params, Inputs, Outputs], jax.Array] = jax.jit(make_param_metric(self.parent_training.eval_model))

        self.stop_criteria: Optional[stopCritTest] = _pkl_instantiate(self.dry_stop_criteria)

        logging.debug(f"session (id = {self.id}) is instantiated")


# TODO add additional index for training_id, step combination
class trainingRecord(Base):
    __tablename__ = 'history'
    __table_args__ = (
        Index('uix_history_training_id_step', 'training_id', 'step', unique=True),
    )
    id: Mapped[int] = mapped_column('recordID', primary_key=True) # will auto-increment
    time_added: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now)
    index: Mapped[int] # old block_id
    worker: Mapped[Optional[str]]
    
    training_id: Mapped[int] = mapped_column(ForeignKey('training.id'))
    parent_training: Mapped['ModelTrainee'] = relationship(
        back_populates='records',
    )

    session_id: Mapped[int] = mapped_column(ForeignKey('schedule.id'))
    parent_session: Mapped['trainingSession'] = relationship(back_populates='records')

    child_forks: Mapped[List['trainingFork']] = relationship(
        back_populates='forkRecord',
        cascade="all, delete, delete-orphan",
    )

    step: Mapped[int]
    parameter: Mapped[Params] = mapped_column(PyTreeType)
    train_metric: Mapped[Optional[float]]
    test_metric: Mapped[Optional[float]]

    state: Mapped[trainingState] = composite(
        'step',
        'parameter',
        mapped_column('opt_state', PickleType, nullable=True),
        mapped_column('rand_key', JaxPyTreeType, nullable=True),
    )

    def __repr__(self) -> str:
        keys = ['id', 'time_added', 'worker', 'training_id', 'session_id', 'step', 'train_metric', 'test_metric']
        return super().__repr__() + "\n" + "\n".join([f"{k}: {getattr(self, k)}" for k in keys]) 

class trainingFork(Base):
    __tablename__ = 'fork'
    id: Mapped[int] = mapped_column(ForeignKey('training.id'), primary_key=True)
    child_training: Mapped['ModelTrainee'] = relationship(back_populates='parent_fork')

    forkRecID: Mapped[int] = mapped_column(ForeignKey('history.recordID'))
    forkRecord: Mapped['trainingRecord'] = relationship(back_populates='child_forks')


class trainingLoss(Base):
    __tablename__ = 'losses'
    id: Mapped[int] = mapped_column(primary_key=True) # will auto-increment

    training_id: Mapped[int] = mapped_column(ForeignKey('training.id'))
    parent_training: Mapped['ModelTrainee'] = relationship(
        back_populates='loss_trace',
    )

    session_id: Mapped[int] = mapped_column(ForeignKey('schedule.id'))
    parent_session: Mapped['trainingSession'] = relationship(back_populates='step_losses')

    step: Mapped[int]
    value: Mapped[float]


from sqlalchemy import Engine
def create_db_session(engine: str|Engine|None, **engine_kwargs):
    """create a sqlalchemy session to a database for ModelTrainee
    if engine is existing Engine instance, the new session will connect to it;
    if engine is url, this function will create engine first and then connect;
    if engine is not provided, this function will create an in-memory sqlite database for the session to connect to
    
    """

    if isinstance(engine, str):
        engine = create_engine(engine, **engine_kwargs)
    elif engine is None:
        engine = create_engine('sqlite://', **engine_kwargs)
    else:
        if not isinstance(engine, Engine):
            raise ValueError(f"the provided engine argument is neither an engine url nor existing engine: {engine}")
    
    Base.metadata.create_all(engine)
    sess = Session(engine)

    return sess
