from allo.memory import Layout


def test_layout_multi_digit():
    layout = Layout("S10R")
    assert layout.placement == [("S", 10), ("R", None)]
