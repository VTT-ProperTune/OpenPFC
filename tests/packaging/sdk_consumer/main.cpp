// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <openpfc/kernel/utils/logging.hpp>
int main() {
  pfc::log(pfc::k_null_logger, pfc::LogLevel::Debug, "installed SDK consumer");
}
