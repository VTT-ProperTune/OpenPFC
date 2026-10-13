# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Guard the supported SDK build contracts without requiring GPU hardware."""
import re
from pathlib import Path


def block(source, name):
    match = re.search(rf'^  {re.escape(name)}:\s*\n(.*?)(?=^  [\w-]+:|\Z)',
                      source, re.M | re.S)
    assert match, f'Missing CI job {name}'
    return match[1]


def check(source):
    status = block(source, 'ci-status')
    needs = re.search(r'^    needs:\s*\[([^]]+)\]', status, re.M)
    assert needs, 'Missing aggregate job dependencies'
    dependencies = {x.strip() for x in needs[1].split(',')}
    for name in ('cuda-compile', 'hip-compile'):
        job = block(source, name)
        assert not re.search(r'^    continue-on-error:\s*true', job, re.M), name
        assert not re.search(r'^    if:', job, re.M), f'{name} must run on PRs'
        assert name in dependencies, f'{name} omitted from aggregate status'
        assert f'needs.{name}.result' in status, f'{name} result not checked'


if __name__ == '__main__':
    check((Path(__file__).resolve().parents[1] / '.github/workflows/ci.yml').read_text())
    print('CUDA/HIP compilation and package consumers participate in CI Status')
