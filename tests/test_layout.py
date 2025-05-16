# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from allo.memory import Layout


def test_layout_multi_digit():
    layout = Layout("S10R")
    assert layout.placement == [("S", 10), ("R", None)]


def test_layout_missing_dimension():
    with pytest.raises(ValueError):
        Layout("SR")


def test_layout_invalid_letter():
    with pytest.raises(ValueError):
        Layout("X0")


def test_layout_unexpected_index():
    with pytest.raises(ValueError):
        Layout("R0")
