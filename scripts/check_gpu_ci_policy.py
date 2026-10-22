# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Guard the supported SDK build contracts without requiring GPU hardware.

CUDA and HIP compile-only jobs belong to the master tier: they must run on
every push to master of ahojukka5/OpenPFC (pull requests run only the common
build), must not be optional, and must fail CI Status when they fail.
"""
import re
from pathlib import Path

MASTER_TIER = "github.repository == 'ahojukka5/OpenPFC' && github.event_name != 'pull_request'"


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
    assert 'RESULTS: ${{ toJSON(needs) }}' in status, 'CI Status does not read every job result'
    assert 'pr and job not in required' in status, 'CI Status may skip jobs outside pull requests'
    for name in ('cuda-compile', 'hip-compile'):
        job = block(source, name)
        assert not re.search(r'^    continue-on-error:\s*true', job, re.M), name
        conditions = re.findall(r'^    if:\s*(.+)$', job, re.M)
        assert conditions == [MASTER_TIER], f'{name} must run on every push to master'
        assert name in dependencies, f'{name} omitted from aggregate status'


if __name__ == '__main__':
    check((Path(__file__).resolve().parents[1] / '.github/workflows/ci.yml').read_text())
    print('CUDA/HIP compilation and package consumers participate in CI Status')
