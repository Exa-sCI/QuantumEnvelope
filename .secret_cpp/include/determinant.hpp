#pragma once
#include <qpx.hpp>

int compute_phase_single_spin_excitation(spin_det_t d, uint64_t h, uint64_t p);
det_t apply_single_spin_excitation(det_t s, int spin, uint64_t hole, uint64_t particle);

