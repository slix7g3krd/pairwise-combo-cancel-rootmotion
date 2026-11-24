import pytest
import gameplay.combo_cancel_rootmotion as mod

CASES = [
    ("order_events_by_time", (1,)),
    ("rollback_root_delta", (1,)),
    ("reset_accumulator", (1,)),
    ("single_layer_root", (1,)),
    ("window_by_time", (1,)),
    ("return_to_idle", (1,)),
    ("play_combo", (1,)),
    ("cancel", (1,)),
    ("net_rollback", (1,)),
    ("sample_root", (1,)),
    ("validate_combo", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
