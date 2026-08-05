from contextlib import contextmanager

import pytest

pytest.importorskip("torch")

from gazelle.routing.timing import record_stage


class RecordingStages:
    def __init__(self):
        self.entered = []
        self.exited = []

    @contextmanager
    def stage(self, name):
        self.entered.append(name)
        try:
            yield
        finally:
            self.exited.append(name)


def test_record_stage_is_a_no_op_without_a_recorder():
    with record_stage(None, "ignored"):
        value = 3

    assert value == 3


def test_record_stage_delegates_to_the_diagnostic_recorder():
    recorder = RecordingStages()

    with record_stage(recorder, "router"):
        assert recorder.entered == ["router"]

    assert recorder.exited == ["router"]
