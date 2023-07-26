#if !defined(DOCTEST_CONFIG_DISABLE)
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#endif
#include <doctest/doctest.h>
#include <qpx.hpp>
#include <determinant.hpp>

int compute_phase_single_spin_excitation(spin_det_t d, uint64_t h, uint64_t p) {
  const auto& [i, j] = std::minmax(h, p);
  spin_det_t hpmask(d.size());
  hpmask.set(i + 1, j - i - 1, 1);
  const bool parity = (hpmask & d).count() % 2;
  return parity ? -1 : 1;
}

TEST_CASE("testing get_phase_single") {
  CHECK(compute_phase_single_spin_excitation(spin_det_t{"11000"}, 4, 2) == -1);
  CHECK(compute_phase_single_spin_excitation(spin_det_t{"10001"}, 4, 2) == 1);
  CHECK(compute_phase_single_spin_excitation(spin_det_t{"01100"}, 2, 4) == -1);
  CHECK(compute_phase_single_spin_excitation(spin_det_t{"00100"}, 2, 4) == 1);
}

det_t apply_single_spin_excitation(det_t s, int spin, uint64_t h, uint64_t p) {
  assert(s[spin][h] == 1);
  assert(s[spin][p] == 0);

  auto s2 = det_t{s};
  s2[spin][h] = 0;
  s2[spin][p] = 1;
  return s2;
}

TEST_CASE("testing apply_single_spin_excitation") {
  det_t s{spin_det_t{"11000"}, spin_det_t{"00001"}};
  CHECK(apply_single_spin_excitation(s, 0, 4, 1) ==
        det_t{spin_det_t{"01010"}, spin_det_t{"00001"}});
  CHECK(apply_single_spin_excitation(s, 1, 0, 1) ==
        det_t{spin_det_t{"11000"}, spin_det_t{"00010"}});
}
