from contextlib import nullcontext


def record_stage(stage_recorder, name: str):
    """Return a profiling context without affecting normal model execution.

    Production training and inference pass no recorder and therefore use a
    ``nullcontext``.  Diagnostic scripts may provide an object exposing a
    ``stage(name)`` context manager, for example one backed by CUDA events.
    """
    if stage_recorder is None:
        return nullcontext()
    return stage_recorder.stage(name)
