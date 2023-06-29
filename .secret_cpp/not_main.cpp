#include <chrono>
#include <iostream>
#include <random>
#include <sul/dynamic_bitset.hpp>
#include <tuple>
#include <unordered_map>
#include <vector>

typedef sul::dynamic_bitset<> spin_det_t;
template <> struct std::hash<spin_det_t> {
  std::size_t operator()(spin_det_t const &s) const noexcept {
    return std::hash<std::string>()(s.to_string());
  }
};

typedef std::pair<spin_det_t, spin_det_t> det_t;
template <> struct std::hash<det_t> {
  std::size_t operator()(det_t const &s) const noexcept {
    std::size_t h1 = std::hash<spin_det_t>{}(s.first);
    std::size_t h2 = std::hash<spin_det_t>{}(s.second);
    return h1 ^ (h2 << 1);
  }
};
std::ostream &operator<<(std::ostream &os, const det_t &obj) {
  return os << "(" << obj.first << "," << obj.second << ")";
}

typedef sul::dynamic_bitset<> spin_occupancy_mask_t;
typedef std::pair<spin_occupancy_mask_t, spin_occupancy_mask_t> occupancy_mask_t;

typedef sul::dynamic_bitset<> spin_unoccupancy_mask_t;
typedef std::pair<spin_unoccupancy_mask_t, spin_unoccupancy_mask_t> unoccupancy_mask_t;

typedef std::array<uint64_t, 4> eri_4idx_t;

std::vector<unsigned> get_dets_index_statisfing_masks(std::vector<det_t> &psi,
                                                      occupancy_mask_t occupancy_mask,
                                                      unoccupancy_mask_t unoccupancy_mask) {

  std::vector<unsigned> matching_indexes;

  const auto &[alpha_omask, beta_omask] = occupancy_mask;
  const auto &[alpha_umask, beta_umask] = unoccupancy_mask;
  for (unsigned i = 0; i < psi.size(); i++) {
    const auto &[det_alpha, det_beta] = psi[i];
    bool cond_occupancy = alpha_omask.is_subset_of(det_alpha) && beta_omask.is_subset_of(det_beta);
    bool cond_unoccupancy =
        alpha_umask.is_subset_of(~det_alpha) && beta_umask.is_subset_of(~det_beta);
    if (cond_occupancy && cond_unoccupancy) {
      matching_indexes.push_back(i);
    }
  }
  return matching_indexes;
}

spin_det_t apply_single_excitation(spin_det_t s, uint64_t hole, uint64_t particule) {
  assert(s[hole] == 1);
  assert(s[particule] == 0);

  auto s2 = spin_det_t{s};
  s2[hole] = 0;
  s2[particule] = 1;
  return s2;
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

  const auto &[i, j, k, l] = idx;
  if ((i == l))
    return IC_A;
  if ((i == k) && (j == l))
    return IC_B;
  if ((i == k) || (j == l)) {
    if (j != k)
      return IC_C;
    return IC_D;
  }
  if (j == k)
    return IC_E;
  if ((i == j) && (k == l))
    return IC_F;
  if ((i == j) || (k == l))
    return IC_E;
  return IC_G;
}

typedef uint64_t det_idx_t;
typedef float phase_t;
typedef std::pair<std::pair<det_idx_t, det_idx_t>, phase_t> H_contribution_t;

std::vector<H_contribution_t> category_A(uint64_t N_orb, eri_4idx_t idx, std::vector<det_t> &psi) {
  const auto &[i, j, k, l] = idx;

  auto occ = spin_occupancy_mask_t(N_orb);
  occ[i] = 1;

  auto unocc = spin_unoccupancy_mask_t(N_orb);
  std::vector<H_contribution_t> result;
  // Phase is always one
  for (auto index : get_dets_index_statisfing_masks(psi, {occ, occ}, {unocc, unocc})) {
    result.push_back({{index, index}, 1});
  }
  return result;
}

std::vector<H_contribution_t> category_B(uint64_t N_orb, eri_4idx_t idx, std::vector<det_t> &psi) {
  const auto &[i, j, k, l] = idx;

  const auto occ = spin_occupancy_mask_t(N_orb);
  auto occ_i = spin_occupancy_mask_t(N_orb);
  occ_i[i] = 1;
  auto occ_j = spin_occupancy_mask_t(N_orb);
  occ_j[j] = 1;
  const auto occ_ij = occ_i | occ_j;
  const auto unocc = spin_unoccupancy_mask_t(N_orb);

  // Phase is always one
  std::vector<H_contribution_t> result;
  for (auto index : get_dets_index_statisfing_masks(psi, {occ, occ_ij}, {unocc, unocc}))
    result.push_back({{index, index}, 1});
  for (auto index : get_dets_index_statisfing_masks(psi, {occ_ij, occ}, {unocc, unocc}))
    result.push_back({{index, index}, 1});
  for (auto index : get_dets_index_statisfing_masks(psi, {occ_i, occ_j}, {unocc, unocc}))
    result.push_back({{index, index}, 1});
  for (auto index : get_dets_index_statisfing_masks(psi, {occ_j, occ_i}, {unocc, unocc}))
    result.push_back({{index, index}, 1});
  return result;
}

std::vector<H_contribution_t> category_C(uint64_t N_orb, eri_4idx_t idx, std::vector<det_t> &psi) {
  const auto &[i, j, k, l] = idx;

  uint64_t a, b, c;

  if (i == k) {
    a = i;
    b = j;
    c = l;
  } else {
    a = j;
    b = i;
    c = k;
  }

  const auto occ = spin_occupancy_mask_t(N_orb);
  auto occ_a = spin_occupancy_mask_t(N_orb);
  auto occ_b = spin_occupancy_mask_t(N_orb);
  auto occ_c = spin_occupancy_mask_t(N_orb);
  const auto unocc = spin_unoccupancy_mask_t(N_orb);
  auto unocc_b = spin_unoccupancy_mask_t(N_orb);
  auto unocc_c = spin_unoccupancy_mask_t(N_orb);
  occ_a[a] = 1;
  occ_b[b] = 1;
  occ_c[c] = 1;
  unocc_b[b] = 1;
  unocc_c[c] = 1;

  const auto occ_ab = occ_a | occ_b;
  const auto occ_ac = occ_a | occ_c;

  // Phase is always one
  std::vector<H_contribution_t> result;
  for (auto index : get_dets_index_statisfing_masks(psi, {occ_ab, occ}, {unocc_c, unocc})) {
    const auto &[alpha, beta] = psi[index];
    const auto alpha2 = apply_single_excitation(alpha, b, c);
    const det_t det2{alpha2, beta};
    // Ahahahaha...
    for (uint64_t index2 = 0; index2 < psi.size(); index2++) {
      if (psi[index2] == det2) {
        result.push_back({{index, index2}, 1});
      }
    }
  }
  return result;

  for (auto index : get_dets_index_statisfing_masks(psi, {occ_ac, occ}, {unocc_b, unocc}))
    result.push_back({{index, index}, 1});
  for (auto index : get_dets_index_statisfing_masks(psi, {occ, occ_ab}, {unocc, unocc_c}))
    result.push_back({{index, index}, 1});
  for (auto index : get_dets_index_statisfing_masks(psi, {occ, occ_ac}, {unocc, unocc_b}))
    result.push_back({{index, index}, 1});

  for (auto index : get_dets_index_statisfing_masks(psi, {occ_a, occ_b}, {unocc, unocc_c}))
    result.push_back({{index, index}, 1});
  for (auto index : get_dets_index_statisfing_masks(psi, {occ_a, occ_c}, {unocc, unocc_b}))
    result.push_back({{index, index}, 1});
  for (auto index : get_dets_index_statisfing_masks(psi, {occ_b, occ_a}, {unocc_c, unocc}))
    result.push_back({{index, index}, 1});
  for (auto index : get_dets_index_statisfing_masks(psi, {occ_c, occ_a}, {unocc_b, unocc}))
    result.push_back({{index, index}, 1});

  return result;
}

int main(int argc, char **argv) {
  int N = std::stoi(argv[1]);
  std::vector<det_t> psi;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      auto d1 = spin_det_t(N, i);
      auto d2 = spin_det_t(N, j);
      psi.push_back({d1, d2});
    }
  }

  for (auto &[pair_det, phase] : category_C(N, {1, 2, 1, 3}, psi)) {
    std::cout << psi[pair_det.first] << " " << psi[pair_det.second] << std::endl;
  }
  return 0;
}
