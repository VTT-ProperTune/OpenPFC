# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""An optional, omitted or conditionally skipped GPU job must fail policy."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('policy', ROOT / 'scripts/check_gpu_ci_policy.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
SOURCE = (ROOT / '.github/workflows/ci.yml').read_text()


def test_current_policy():
    policy.check(SOURCE)


@pytest.mark.parametrize('name', ['cuda-compile', 'hip-compile'])
@pytest.mark.parametrize('failure', ['optional', 'omitted', 'skipped', 'unchecked'])
def test_policy_rejects_hidden_failures(name, failure):
    if failure == 'optional':
        broken = SOURCE.replace(f'  {name}:', f'  {name}:\n    continue-on-error: true')
    elif failure == 'skipped':
        broken = SOURCE.replace(f'  {name}:', f"  {name}:\n    if: github.event_name == 'push'")
    elif failure == 'omitted':
        broken = SOURCE.replace(f', {name}', '')
    else:
        broken = SOURCE.replace(f'needs.{name}.result', 'needs.unrelated.result')
    with pytest.raises(AssertionError):
        policy.check(broken)
