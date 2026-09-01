# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_subdirectory(common)
if(OpenPFC_ENABLE_HEFFTE)
  add_subdirectory(tungsten)
  add_subdirectory(aluminumNew)
  add_subdirectory(cahn_hilliard)
  add_subdirectory(thin_film)
  add_subdirectory(surface_diffusion)
  add_subdirectory(kawahara)
  add_subdirectory(ehd_film)
  add_subdirectory(gradient_elasticity)
endif()
add_subdirectory(allen_cahn)
add_subdirectory(heat3d)
add_subdirectory(wave2d)
add_subdirectory(kobayashi)
add_subdirectory(alloy_pf_karma2001_benchmark)
