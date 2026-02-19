# Recipe: Copying Database Records Between Sessions

This recipe demonstrates how to copy a `trainingRecord` and its related parent entities from one SQLAlchemy database session to another (e.g., from a production PostgreSQL DB to a local SQLite DB).

## Key Concepts

- **Eager Loading (`joinedload`)**: Ensures related records are fetched in the initial query so they are available after the object is detached.
- **Expunging (`expunge`)**: Removes the object from the source session's tracking. This is necessary to enable the object being re-attached to another session.
- **Merging (`merge`)**: Re-attaches the object (and its loaded relations) to a new session, creating or updating records as needed.

## Training Records as Model Snapshots

In the `disRNN_MP` framework, a `trainingRecord` is more than just a log entry; it represents the **trained model parameters** at a specific training step. By copying a `trainingRecord`, you are effectively migrating a **model snapshot**. This allows you to resume training, perform evaluations, or fork new models from that exact state in a different environment.

## Code Example

```python
import sqlalchemy as sa
from sqlalchemy.orm import joinedload
from disRNN_MP.rnn.train_db import create_db_session, trainingRecord

# 1. Initialize sessions for source and target databases
source_sess = create_db_session("postgresql://user:pass@host:5432/dbname")
target_sess = create_db_session('sqlite:///local_copy.db')

mt_id = 25116
step = 337802

# 2. Fetch the record with eager loading of relationships
# This is CRITICAL: without joinedload, parent_session and parent_training 
# would not be copied because they are normally lazy-loaded.
rec = source_sess.execute(
    sa.select(trainingRecord)
    .where(sa.and_(trainingRecord.training_id == mt_id, trainingRecord.step == step))
    .options(
        joinedload(trainingRecord.parent_session),
        joinedload(trainingRecord.parent_training)
    ) 
).scalar_one()

# 3. Detach from source and migrate to target
source_sess.expunge(rec)
target_sess.merge(rec)
target_sess.commit()

print(f"Successfully copied record {mt_id} at step {step} to target database.")
```

## Why use `joinedload`?
When you `expunge` a record, it becomes "detached." If you didn't use `joinedload`, any attempt to access `rec.parent_session` later would raise a `DetachedInstanceError` because SQLAlchemy can no longer go back to the source database to fetch the missing data. `joinedload` ensures all necessary data is already in memory before the move.
