
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def pekeris_env():
    from _test_helpers import pekeris_B1_rho_h

    return pekeris_B1_rho_h()


@pytest.fixture(params=[torch.float64, torch.complex128], ids=["real", "complex"])
def real_or_complex(request):
    return request.param
