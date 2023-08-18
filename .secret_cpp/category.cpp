#include <chrono>
#include <iostream>
#include <random>
#include <sul/dynamic_bitset.hpp>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <array>
#if !defined(DOCTEST_CONFIG_DISABLE)
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#endif
#include <doctest/doctest.h>
#include <qpx.hpp>
#include <determinant.hpp>

// Utils
uint64_t binom(int n, int k) {
  if(k == 0 || k == n) return 1;
  return binom(n - 1, k - 1) + binom(n - 1, k);
}

// We don't need it.
// But one day... Map a spin_det_t to a int
uint64_t unchoose(int n, spin_det_t S) {
  auto k = S.count();
  if((k == 0) || (k == n)) return 0;
  auto j = S.find_first();
  if(k == 1) return j;
  S >>= 1;
  if(!j) return unchoose(n - 1, S);
  return binom(n - 1, k - 1) + unchoose(n - 1, S);
}

//// i-th lexicographical bit string of lenth n with popcount k
spin_det_t combi(size_t i, size_t n, size_t k, size_t N = 0) {
  if(N == 0) N = n;
  if(k == 0) return spin_det_t(N);

  assert(i < binom(n, k));
  auto n0 = binom(n - 1, k - 1);
  if(i < n0) {
    return spin_det_t(N, 1) | (combi(i, n - 1, k - 1, N) << 1);
  } else {
    return combi(i - n0, n - 1, k, N) << 1;
  }
}

template<typename T>
spin_det_t vec_to_spin_det(std::vector<T> idx, size_t n_orb) {
  spin_det_t res(n_orb);
  for(auto i : idx) res[i] = 1;
  return res;
}


// Phase
spin_det_t get_phase_mask(spin_det_t p) {
  size_t i = 0;
  while(true) {
    spin_det_t q = (p << (1 << i++));
    if(!q.any()) return p;
    p ^= q;
  }
  /*
  for(size_t i = 0; (p << (1 << i)).any(); i++)
     { p ^= (p << (1 << i)); }
  return p;
*/
}

// Integral Driven

std::vector<unsigned> get_dets_index_statisfing_masks(std::vector<det_t>& psi,
                                                      occupancy_mask_t occupancy_mask,
                                                      unoccupancy_mask_t unoccupancy_mask) {
  std::vector<unsigned> matching_indexes;

  const auto& [alpha_omask, beta_omask] = occupancy_mask;
  const auto& [alpha_umask, beta_umask] = unoccupancy_mask;
  for(unsigned i = 0; i < psi.size(); i++) {
    const auto& [det_alpha, det_beta] = psi[i];
    bool cond_occupancy = alpha_omask.is_subset_of(det_alpha) && beta_omask.is_subset_of(det_beta);
    bool cond_unoccupancy =
        alpha_umask.is_subset_of(~det_alpha) && beta_umask.is_subset_of(~det_beta);
    if(cond_occupancy && cond_unoccupancy) matching_indexes.push_back(i);
  }
  return matching_indexes;
}

TEST_CASE("testing get_dets_index_statisfing_masks") {
  std::vector<det_t> psi{
      {spin_det_t{"11000"}, spin_det_t{"11000"}},
      {spin_det_t{"11000"}, spin_det_t{"11010"}},
  };

  SUBCASE("") {
    occupancy_mask_t occupancy_mask{spin_occupancy_mask_t{"10000"}, spin_occupancy_mask_t{"11000"}};
    unoccupancy_mask_t unoccupancy_mask{spin_unoccupancy_mask_t{"00010"},
                                        spin_unoccupancy_mask_t{"00100"}};

    CHECK(get_dets_index_statisfing_masks(psi, occupancy_mask, unoccupancy_mask) ==
          std::vector<unsigned>{0, 1});
  }

  SUBCASE("") {
    occupancy_mask_t occupancy_mask{spin_occupancy_mask_t{"10000"}, spin_occupancy_mask_t{"11000"}};
    unoccupancy_mask_t unoccupancy_mask{spin_unoccupancy_mask_t{"00010"},
                                        spin_unoccupancy_mask_t{"00010"}};

    CHECK(get_dets_index_statisfing_masks(psi, occupancy_mask, unoccupancy_mask) ==
          std::vector<unsigned>{0});
  }

  SUBCASE("") {
    occupancy_mask_t occupancy_mask{spin_occupancy_mask_t{"11000"}, spin_occupancy_mask_t{"10010"}};
    unoccupancy_mask_t unoccupancy_mask{spin_unoccupancy_mask_t{"00010"},
                                        spin_unoccupancy_mask_t{"00100"}};

    CHECK(get_dets_index_statisfing_masks(psi, occupancy_mask, unoccupancy_mask) ==
          std::vector<unsigned>{1});
  }
}

enum integrals_categorie_e { IC_A, IC_B, IC_C, IC_D, IC_E, IC_F, IC_G };

/*
for real orbitals, return same 4-tuple for all equivalent integrals
    returned (i,j,k,l) should satisfy the following:
        i <= k
        j <= l
        (k < l) or (k==l and i <= j)
*/

integrals_categorie_e integral_category(eri_4idx_t idx) {
  const auto& [i, j, k, l] = idx;
  if((i == l)) return IC_A;
  if((i == k) && (j == l)) return IC_B;
  if((i == k) || (j == l)) {
    if(j != k) return IC_C;
    return IC_D;
  }
  if(j == k) return IC_E;
  if((i == j) && (k == l)) return IC_F;
  if((i == j) || (k == l)) return IC_E;
  return IC_G;
}

typedef uint64_t det_idx_t;
typedef float phase_t;
typedef std::pair<std::pair<det_idx_t, det_idx_t>, phase_t> H_contribution_t;

std::vector<H_contribution_t> category_A(uint64_t N_orb, eri_4idx_t idx, std::vector<det_t>& psi) {
  assert(integral_category(idx) == IC_A);
  const auto& [i, j, k, l] = idx;
  auto occ                 = spin_occupancy_mask_t(N_orb);
  occ[i]                   = 1;

  auto unocc = spin_unoccupancy_mask_t(N_orb);
  std::vector<H_contribution_t> result;
  // Phase is always one
  for(auto index : get_dets_index_statisfing_masks(psi, {occ, occ}, {unocc, unocc})) {
    result.push_back({{index, index}, 1});
  }
  return result;
}

TEST_CASE("testing category A") {
  std::vector<det_t> psi{
      {spin_det_t{"11001"}, spin_det_t{"11001"}},
      {spin_det_t{"11001"}, spin_det_t{"11000"}},
      {spin_det_t{"11000"}, spin_det_t{"11001"}},
      {spin_det_t{"11001"}, spin_det_t{"11011"}},
  };
  CHECK(category_A(5, {0, 0, 0, 0}, psi) ==
        std::vector<H_contribution_t>{{{0, 0}, 1}, {{3, 3}, 1}});
}

std::vector<H_contribution_t> category_B(uint64_t N_orb, eri_4idx_t idx, std::vector<det_t>& psi) {
  assert(integral_category(idx) == IC_B);

  const auto& [i, j, k, l] = idx;
  const auto occ           = spin_occupancy_mask_t(N_orb);
  auto occ_i               = spin_occupancy_mask_t(N_orb);
  occ_i[i]                 = 1;
  auto occ_j               = spin_occupancy_mask_t(N_orb);
  occ_j[j]                 = 1;
  const auto occ_ij        = occ_i | occ_j;
  const auto unocc         = spin_unoccupancy_mask_t(N_orb);

  // Phase is always one
  std::vector<H_contribution_t> result;
  for(auto index : get_dets_index_statisfing_masks(psi, {occ, occ_ij}, {unocc, unocc}))
    result.push_back({{index, index}, 1});
  for(auto index : get_dets_index_statisfing_masks(psi, {occ_ij, occ}, {unocc, unocc}))
    result.push_back({{index, index}, 1});
  for(auto index : get_dets_index_statisfing_masks(psi, {occ_i, occ_j}, {unocc, unocc}))
    result.push_back({{index, index}, 1});
  for(auto index : get_dets_index_statisfing_masks(psi, {occ_j, occ_i}, {unocc, unocc}))
    result.push_back({{index, index}, 1});
  return result;
}

TEST_CASE("testing category B") {
  std::vector<det_t> psi{
      {spin_det_t{"00110"}, spin_det_t{"00000"}},
      {spin_det_t{"00000"}, spin_det_t{"00110"}},
      {spin_det_t{"00010"}, spin_det_t{"00100"}},
      {spin_det_t{"00100"}, spin_det_t{"00010"}},
  };
  CHECK(category_B(5, {1, 2, 1, 2}, psi) ==
        std::vector<H_contribution_t>{{{1, 1}, 1}, {{0, 0}, 1}, {{2, 2}, 1}, {{3, 3}, 1}});
}

void category_C_ijil(uint64_t N_orb, eri_4idx_t idx, std::vector<det_t>& psi,
                     std::vector<H_contribution_t>& result) {
  const auto& [i, j, k, l] = idx;
  assert(i == k);
  uint64_t a, b, c;
  a = i;
  b = j;
  c = l;

  for(size_t spin_a = 0; spin_a < N_SPIN_SPECIES; spin_a++) {
    for(size_t spin_bc = 0; spin_bc < N_SPIN_SPECIES; spin_bc++) {
      auto occ_alpha           = spin_occupancy_mask_t(N_orb);
      auto occ_beta            = spin_occupancy_mask_t(N_orb);
      auto unocc_alpha         = spin_unoccupancy_mask_t(N_orb);
      auto unocc_beta          = spin_unoccupancy_mask_t(N_orb);
      occupancy_mask_t occ     = {occ_alpha, occ_beta};
      unoccupancy_mask_t unocc = {unocc_alpha, unocc_beta};

      occ[spin_a][a]    = 1;
      occ[spin_bc][b]   = 1;
      unocc[spin_bc][c] = 1;
      for(auto index0 : get_dets_index_statisfing_masks(psi, occ, unocc)) {
        const auto det0 = psi[index0];
        const auto det1 = apply_single_spin_excitation(det0, spin_bc, b, c);

        for(uint64_t index1 = 0; index1 < psi.size(); index1++) {
          if(psi[index1] == det1) {
            result.push_back(
                {{index0, index1}, compute_phase_single_spin_excitation(det0[spin_bc], b, c)});
          }
        }
      }
    }
  }
}

std::vector<H_contribution_t> category_C(uint64_t N_orb, eri_4idx_t idx, std::vector<det_t>& psi) {
  std::vector<H_contribution_t> result;

  const auto& [i, j, k, l] = idx;
  uint64_t a, b, c;

  if(i == k) {
    a = i;
    b = j;
    c = l;
  } else {
    a = j;
    b = i;
    c = k;
  }
  const eri_4idx_t idx_bc = {a, b, a, c};
  const eri_4idx_t idx_cb = {a, c, a, b};

  category_C_ijil(N_orb, idx_bc, psi, result);
  category_C_ijil(N_orb, idx_cb, psi, result);
  return result;
}

void category_D_iiil(uint64_t N_orb, eri_4idx_t idx, std::vector<det_t>& psi,
                     std::vector<H_contribution_t>& result) {
  const auto& [i, j, k, l] = idx;
  assert(i == k);
  assert(i == j);
  uint64_t a, b;
  a = i;
  b = l;

  std::array<uint64_t, 2> ph{a, b};

  // aβ aα -> aβ bα
  // aβ aα <- aβ bα
  // aα aβ -> aα bβ
  // aα aβ <- aα bβ
  for(size_t spin_ph = 0; spin_ph < N_SPIN_SPECIES; spin_ph++) {
    size_t spin_aa = !spin_ph;
    for(size_t exc_fwd = 0; exc_fwd < N_SPIN_SPECIES; exc_fwd++) {
      size_t exc_rev = !exc_fwd;
      auto p         = ph[exc_fwd];
      auto h         = ph[exc_rev];

      auto occ_alpha           = spin_occupancy_mask_t(N_orb);
      auto occ_beta            = spin_occupancy_mask_t(N_orb);
      auto unocc_alpha         = spin_unoccupancy_mask_t(N_orb);
      auto unocc_beta          = spin_unoccupancy_mask_t(N_orb);
      occupancy_mask_t occ     = {occ_alpha, occ_beta};
      unoccupancy_mask_t unocc = {unocc_alpha, unocc_beta};

      occ[spin_ph][h]   = 1;
      occ[spin_aa][a]   = 1;
      unocc[spin_ph][p] = 1;
      for(auto index0 : get_dets_index_statisfing_masks(psi, occ, unocc)) {
        const auto det0 = psi[index0];
        const auto det1 = apply_single_spin_excitation(det0, spin_ph, h, p);

        for(uint64_t index1 = 0; index1 < psi.size(); index1++) {
          if(psi[index1] == det1) {
            result.push_back(
                {{index0, index1}, compute_phase_single_spin_excitation(det0[spin_ph], h, p)});
          }
        }
      }
    }
  }
}


std::vector<H_contribution_t> category_D(uint64_t N_orb, eri_4idx_t idx, std::vector<det_t>& psi) {
  std::vector<H_contribution_t> result;

  const auto& [i, j, k, l] = idx;
  uint64_t a, b;

  if(i == j) {
    // (ii|il)
    a = i;
    b = l;
  } else {
    // (il|ll)
    a = l;
    b = i;
  }

  category_D_iiil(N_orb, {a, a, a, b}, psi, result);
  return result;
}
