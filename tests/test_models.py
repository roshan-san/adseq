"""Model names stay in sync across config, models and report."""

from typing import get_args

from adseq.config import ModelName
from adseq.models import TUNED_MODELS
from adseq.report import NAMES


def test_every_model_has_a_display_name():
    assert list(NAMES) == list(get_args(ModelName))
    assert set(TUNED_MODELS) <= set(NAMES)
