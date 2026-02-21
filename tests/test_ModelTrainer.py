"""Tests for ModelTrainee training workflow using disrnn_def interface.

Purpose: Verify ModelTrainee creation, training, persistence, forking, deletion,
         and CLI entry point using the latest disRNN_MP.rnn.disrnn_def API.
Method:  Each test builds a small disRNN model (latent_size=3) with short
         training sessions and exercises the train_db ORM layer against an
         in-memory SQLite database.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import subprocess
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from disRNN_MP.rnn.train_db import (
    Base, ModelTrainee, trainingSession, trainingRecord,
    trainingFork, trainingLoss,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent.parent
_DATA_PATH = _PKG_ROOT / "data" / "mp_beh_m18_500completed.npy"

# Small model for fast tests
_DRY_MODEL = {
    '_target_': 'disRNN_MP.rnn.disrnn_def.make_transformed_disrnn',
    'latent_size': 3,
    'update_mlp_shape': [3],
    'choice_mlp_shape': [2],
    'target_size': 2,
    'eval_mode': False,
}

_DRY_EVAL_MODEL = {
    '_target_': 'disRNN_MP.rnn.disrnn_def.make_transformed_disrnn',
    'latent_size': 3,
    'update_mlp_shape': [3],
    'choice_mlp_shape': [2],
    'target_size': 2,
    'eval_mode': True,
}

_DRY_DATASETS = {
    '_target_': 'disRNN_MP.dataset.train_test_datasets',
    'dat_or_path': _DATA_PATH.as_posix(),
    'n_sess_sample': 42,
    'seed': 5,
}

_DRY_OPTIMIZER = {
    '_target_': 'optax.adam',
    'learning_rate': 1e-3,
}

_DRY_MAKE_TRAIN_STEP_NO_PENALTY = {
    '_target_': 'disRNN_MP.rnn.disrnn_def.make_train_step',
    '_partial_': True,
    'penalty_scale': 0,
}

_DRY_MAKE_TRAIN_STEP_WITH_PENALTY = {
    '_target_': 'disRNN_MP.rnn.disrnn_def.make_train_step',
    '_partial_': True,
    'penalty_scale': 1e-3,
}

_DRY_MAKE_PARAM_METRIC = {
    '_target_': 'disRNN_MP.rnn.disrnn_def.make_param_metric_expLL',
    '_partial_': True,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_engine(tmpdir):
    """In-memory SQLite engine with all tables created."""
    engine = create_engine('sqlite://', echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_file_engine(tmpdir):
    """File-backed SQLite engine in tmpdir (for CLI test)."""
    db_path = str(tmpdir / "test.db")
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    Base.metadata.create_all(engine)
    return engine, db_path


# ---------------------------------------------------------------------------
# Test 1: session config — all explicit
# ---------------------------------------------------------------------------

def test_session_config_all_explicit(db_engine):
    """ModelTrainee with all config specified per-session (no defaults)."""
    se1 = trainingSession(
        dry_optimizer=_DRY_OPTIMIZER,
        dry_make_train_step=_DRY_MAKE_TRAIN_STEP_NO_PENALTY,
        dry_make_param_metric=_DRY_MAKE_PARAM_METRIC,
        dry_datasets=_DRY_DATASETS,
        n_step=20,
        steps_per_block=10,
    )
    se2 = trainingSession(
        dry_optimizer=_DRY_OPTIMIZER,
        dry_make_train_step=_DRY_MAKE_TRAIN_STEP_WITH_PENALTY,
        dry_make_param_metric=_DRY_MAKE_PARAM_METRIC,
        dry_datasets=_DRY_DATASETS,
        n_step=20,
        steps_per_block=10,
    )

    mt = ModelTrainee(
        dry_model=_DRY_MODEL,
        dry_eval_model=_DRY_EVAL_MODEL,
    )
    mt.sessions = [se1, se2]

    with Session(db_engine) as sess:
        sess.add(mt)
        sess.commit()
        tid = mt.id

    # Re-query
    with Session(db_engine) as sess:
        loaded = sess.get(ModelTrainee, tid)
        assert loaded is not None
        assert len(loaded.sessions) == 2
        assert loaded.sessions[0].index == 0
        assert loaded.sessions[1].index == 1
        assert loaded.total_steps == 40


# ---------------------------------------------------------------------------
# Test 2: session config — with defaults
# ---------------------------------------------------------------------------

def test_session_config_with_defaults(db_engine):
    """ModelTrainee defaults propagate to sessions that omit them."""
    mt = ModelTrainee(
        dry_model=_DRY_MODEL,
        dry_eval_model=_DRY_EVAL_MODEL,
        dry_default_datasets=_DRY_DATASETS,
        dry_default_optimizer=_DRY_OPTIMIZER,
        dry_default_make_param_metric=_DRY_MAKE_PARAM_METRIC,
    )

    se1 = trainingSession(
        dry_make_train_step=_DRY_MAKE_TRAIN_STEP_NO_PENALTY,
        n_step=10,
        steps_per_block=5,
    )
    se2 = trainingSession(
        dry_make_train_step=_DRY_MAKE_TRAIN_STEP_WITH_PENALTY,
        n_step=10,
        steps_per_block=5,
    )
    mt.sessions = [se1, se2]

    with Session(db_engine) as sess:
        sess.add(mt)
        sess.commit()
        tid = mt.id

    with Session(db_engine) as sess:
        loaded = sess.get(ModelTrainee, tid)
        assert loaded is not None
        assert loaded.dry_default_datasets is not None
        assert loaded.dry_default_optimizer is not None
        assert loaded.dry_default_make_param_metric is not None
        # Sessions should have None for the fields they didn't specify
        assert loaded.sessions[0].dry_optimizer is None
        assert loaded.sessions[0].dry_datasets is None
        assert loaded.total_steps == 20


# ---------------------------------------------------------------------------
# Test 3: training completes
# ---------------------------------------------------------------------------

def test_train_completes(db_engine):
    """Full training loop runs to completion with 2 short sessions."""
    mt = ModelTrainee(
        dry_model=_DRY_MODEL,
        dry_eval_model=_DRY_EVAL_MODEL,
        dry_default_datasets=_DRY_DATASETS,
        dry_default_optimizer=_DRY_OPTIMIZER,
        dry_default_make_param_metric=_DRY_MAKE_PARAM_METRIC,
    )
    mt.sessions = [
        trainingSession(
            dry_make_train_step=_DRY_MAKE_TRAIN_STEP_NO_PENALTY,
            n_step=20,
            steps_per_block=10,
        ),
        trainingSession(
            dry_make_train_step=_DRY_MAKE_TRAIN_STEP_WITH_PENALTY,
            n_step=20,
            steps_per_block=10,
        ),
    ]

    sess = Session(db_engine, expire_on_commit=False)
    sess.add(mt)
    sess.commit()

    code = mt.train(sess)

    assert code == 0
    assert mt.state.step == mt.total_steps  # 40
    assert len(mt.records) > 0
    assert len(mt.loss_trace) > 0
    sess.close()


# ---------------------------------------------------------------------------
# Test 4: reinstantiate and train
# ---------------------------------------------------------------------------

def test_reinstantiate_and_train(db_engine):
    """ModelTrainee can be re-loaded from DB and resume training."""
    # Phase 1: create with only session 1, train to completion
    mt = ModelTrainee(
        dry_model=_DRY_MODEL,
        dry_eval_model=_DRY_EVAL_MODEL,
        dry_default_datasets=_DRY_DATASETS,
        dry_default_optimizer=_DRY_OPTIMIZER,
        dry_default_make_param_metric=_DRY_MAKE_PARAM_METRIC,
    )
    mt.sessions = [
        trainingSession(
            dry_make_train_step=_DRY_MAKE_TRAIN_STEP_NO_PENALTY,
            n_step=10,
            steps_per_block=5,
        ),
    ]

    sess = Session(db_engine, expire_on_commit=False)
    sess.add(mt)
    sess.commit()
    code = mt.train(sess)
    assert code == 0
    assert mt.state.step == 10
    tid = mt.id
    sess.close()

    # Phase 2: re-open DB, re-load, add session 2, train again
    sess2 = Session(db_engine, expire_on_commit=False)
    loaded = sess2.get(ModelTrainee, tid)
    assert loaded is not None
    assert loaded.state.step == 10

    loaded.sessions.append(trainingSession(
        dry_make_train_step=_DRY_MAKE_TRAIN_STEP_WITH_PENALTY,
        n_step=10,
        steps_per_block=5,
    ))
    sess2.commit()

    code = loaded.train(sess2)
    assert code == 0
    assert loaded.state.step == 20
    assert len(loaded.records) > 0
    sess2.close()


# ---------------------------------------------------------------------------
# Test 5: fork from record
# ---------------------------------------------------------------------------

def test_fork_from_record(db_engine):
    """Forking creates a new ModelTrainee from a checkpoint record."""
    # Train original
    mt = ModelTrainee(
        dry_model=_DRY_MODEL,
        dry_eval_model=_DRY_EVAL_MODEL,
        dry_default_datasets=_DRY_DATASETS,
        dry_default_optimizer=_DRY_OPTIMIZER,
        dry_default_make_param_metric=_DRY_MAKE_PARAM_METRIC,
    )
    mt.sessions = [
        trainingSession(
            dry_make_train_step=_DRY_MAKE_TRAIN_STEP_NO_PENALTY,
            n_step=20,
            steps_per_block=10,
        ),
        trainingSession(
            dry_make_train_step=_DRY_MAKE_TRAIN_STEP_WITH_PENALTY,
            n_step=20,
            steps_per_block=10,
        ),
    ]

    sess = Session(db_engine, expire_on_commit=False)
    sess.add(mt)
    sess.commit()
    code = mt.train(sess)
    assert code == 0

    # Pick a record from the middle
    assert len(mt.records) >= 2
    rec = mt.records[1]  # second checkpoint

    # Fork
    forked = ModelTrainee.fork_from(rec)
    forked.sessions.append(trainingSession(
        dry_make_train_step=_DRY_MAKE_TRAIN_STEP_WITH_PENALTY,
        n_step=10,
        steps_per_block=5,
    ))

    sess.add(forked)
    sess.commit()

    # Train forked
    code = forked.train(sess)
    assert code == 0

    # Verify fork relationship
    assert forked.parent_fork is not None
    assert forked.parent_fork.forkRecord is rec

    # Verify forked trainee has its own records
    assert len(forked.records) > 0
    sess.close()


# ---------------------------------------------------------------------------
# Test 6: delete
# ---------------------------------------------------------------------------

def test_delete(db_engine):
    """ModelTrainee.delete removes trainee and all related records."""
    mt = ModelTrainee(
        dry_model=_DRY_MODEL,
        dry_eval_model=_DRY_EVAL_MODEL,
        dry_default_datasets=_DRY_DATASETS,
        dry_default_optimizer=_DRY_OPTIMIZER,
        dry_default_make_param_metric=_DRY_MAKE_PARAM_METRIC,
    )
    mt.sessions = [
        trainingSession(
            dry_make_train_step=_DRY_MAKE_TRAIN_STEP_NO_PENALTY,
            n_step=10,
            steps_per_block=5,
        ),
    ]

    sess = Session(db_engine, expire_on_commit=False)
    sess.add(mt)
    sess.commit()
    code = mt.train(sess)
    assert code == 0
    tid = mt.id
    assert len(mt.records) > 0
    assert len(mt.loss_trace) > 0

    # Delete
    ModelTrainee.delete(sess, tid)

    # Verify everything is gone
    assert sess.get(ModelTrainee, tid) is None
    assert sess.query(trainingSession).filter_by(training_id=tid).count() == 0
    assert sess.query(trainingRecord).filter_by(training_id=tid).count() == 0
    assert sess.query(trainingLoss).filter_by(training_id=tid).count() == 0
    sess.close()


# ---------------------------------------------------------------------------
# Test 7: CLI train_ModelTrainee
# ---------------------------------------------------------------------------

def test_cli_train(db_file_engine):
    """CLI train_ModelTrainee module can train a committed ModelTrainee."""
    engine, db_path = db_file_engine

    mt = ModelTrainee(
        dry_model=_DRY_MODEL,
        dry_eval_model=_DRY_EVAL_MODEL,
        dry_default_datasets=_DRY_DATASETS,
        dry_default_optimizer=_DRY_OPTIMIZER,
        dry_default_make_param_metric=_DRY_MAKE_PARAM_METRIC,
    )
    mt.sessions = [
        trainingSession(
            dry_make_train_step=_DRY_MAKE_TRAIN_STEP_NO_PENALTY,
            n_step=10,
            steps_per_block=5,
        ),
    ]

    with Session(engine) as sess:
        sess.add(mt)
        sess.commit()
        tid = mt.id

    result = subprocess.run(
        [
            "python", "-m", "disRNN_MP.cli.train_ModelTrainee",
            f"sql_engine.url=sqlite:///{db_path}",
            f"+trainee_id={tid}",
        ],
        capture_output=True,
        timeout=300,
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr.decode()}"

    # Verify training completed in DB
    with Session(engine) as sess:
        loaded = sess.get(ModelTrainee, tid)
        assert loaded is not None
        assert loaded.state.step == loaded.total_steps
        assert len(loaded.records) > 0


# ---------------------------------------------------------------------------
# Test 8: training determinism (snapshot-based)
# ---------------------------------------------------------------------------

def test_training_determinism(db_engine):
    """Training output matches saved snapshot for deterministic reproducibility."""
    import numpy as np
    from disRNN_MP.utils import isequal_pytree, msgpack_restore_from_file

    # Load expected snapshot
    snapshot_path = Path(__file__).parent / "test_data" / "training_snapshot.msgpack"
    expected = msgpack_restore_from_file(snapshot_path)

    # Run training with identical config
    mt = ModelTrainee(
        dry_model=_DRY_MODEL,
        dry_eval_model=_DRY_EVAL_MODEL,
        dry_default_datasets=_DRY_DATASETS,
        dry_default_optimizer=_DRY_OPTIMIZER,
        dry_default_make_param_metric=_DRY_MAKE_PARAM_METRIC,
    )
    mt.sessions = [
        trainingSession(
            dry_make_train_step=_DRY_MAKE_TRAIN_STEP_NO_PENALTY,
            n_step=10,
            steps_per_block=5,
        ),
    ]
    sess = Session(db_engine, expire_on_commit=False)
    sess.add(mt)
    sess.commit()
    code = mt.train(sess)
    assert code == 0

    # Compare params
    assert isequal_pytree(mt.state.params, expected["params"]), \
        "Final params diverge from snapshot"

    # Compare per-step losses
    actual_losses = np.array([l.value for l in mt.loss_trace])
    np.testing.assert_array_equal(actual_losses, expected["losses"],
        err_msg="Per-step losses diverge from snapshot")

    sess.close()
