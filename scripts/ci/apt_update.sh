#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Refresh the apt index without letting a third-party repository fail the job.
#
# The GitHub-hosted Ubuntu images ship extra apt sources for software this
# repository never installs -- Google Chrome, Microsoft prod, and friends.
# Those repositories are published continuously and occasionally serve a
# Packages index whose hash does not match the Release file that was fetched
# moments earlier:
#
#   E: Failed to fetch .../google-chrome/.../Packages.gz  Hash Sum mismatch
#   E: Some index files failed to download.
#
# `apt-get update` then exits 100. Every workflow here runs it as the first
# line of an install step, so a transient inconsistency in Chrome's apt
# repository takes down Code Quality -- and because the build matrix is gated
# on Code Quality, the entire pipeline reports failure on branches that have
# nothing wrong with them. That happened across this repository on
# 2026-09-09 and is what this script exists to prevent.
#
# Dropping the unused sources first means the update only has to succeed for
# the Ubuntu archives we actually install from, and a real archive outage
# still fails loudly instead of being swallowed.

set -euo pipefail

for list in google-chrome microsoft-prod; do
  sudo rm -f "/etc/apt/sources.list.d/${list}.list" \
             "/etc/apt/sources.list.d/${list}.sources"
done

sudo apt-get update
