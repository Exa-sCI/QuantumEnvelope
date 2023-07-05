#include <chrono>
#include <iostream>
#include <random>
#include <sul/dynamic_bitset.hpp>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <array>

typedef sul::dynamic_bitset<> spin_det_t;

// get phasemask
spin_det_t get_pm(spin_det_t p);

template<>
struct std::hash<spin_det_t> {
  std::size_t operator()(spin_det_t const& s) const noexcept {
    return std::hash<std::string>()(s.to_string());
  }
};

typedef std::array<spin_det_t, 2> det_t;
template<>
struct std::hash<det_t> {
  std::size_t operator()(det_t const& s) const noexcept {
    std::size_t h1 = std::hash<spin_det_t>{}(s[0]);
    std::size_t h2 = std::hash<spin_det_t>{}(s[1]);
    return h1 ^ (h2 << 1);
  }
};
std::ostream& operator<<(std::ostream& os, const det_t& obj) {
  return os << "(" << obj[0] << "," << obj[1] << ")";
}

typedef sul::dynamic_bitset<> spin_occupancy_mask_t;
typedef std::array<spin_occupancy_mask_t, 2> occupancy_mask_t;

typedef sul::dynamic_bitset<> spin_unoccupancy_mask_t;
typedef std::array<spin_unoccupancy_mask_t, 2> unoccupancy_mask_t;

typedef std::array<uint64_t, 4> eri_4idx_t;

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
  if(k == 0) { return spin_det_t(N); }
  assert(i < binom(n, k));
  auto n0 = binom(n - 1, k - 1);
  if(i < n0) {
    return spin_det_t(N, 1) | (combi(i, n - 1, k - 1, N) << 1);
  } else {
    return (combi(i - n0, n - 1, k, N) << 1);
  }
}

template<typename T>
spin_det_t vec_to_spin_det(std::vector<T> idx, size_t nbits = 0) {
  if(!nbits) nbits = *std::max_element(idx.begin(), idx.end());
  spin_det_t res(nbits);
  for(auto i : idx) res.set(i);
  return res;
}

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
    if(cond_occupancy && cond_unoccupancy) { matching_indexes.push_back(i); }
  }
  return matching_indexes;
}

spin_det_t apply_single_excitation(spin_det_t s, uint64_t hole, uint64_t particle) {
  assert(s[hole] == 1);
  assert(s[particle] == 0);

  auto s2      = spin_det_t{s};
  s2[hole]     = 0;
  s2[particle] = 1;
  return s2;
}
det_t apply_single_spin_excitation(det_t s, size_t spin, uint64_t hole, uint64_t particle) {
  auto opp_spin = (spin + 1) % 2;

  assert(s[spin][hole] == 1);
  assert(s[spin][particle] == 0);

  auto s2            = det_t{s};
  s2[spin][hole]     = 0;
  s2[spin][particle] = 1;
  return s2;
}

double get_phase_single(spin_det_t d, size_t p, size_t h) {
  auto pm     = get_pm(d);
  auto parity = pm[h] + pm[p] + (int)(h > p);
  return std::pow(-1, parity);
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
  const auto& [i, j, k, l] = idx;

  auto occ = spin_occupancy_mask_t(N_orb);
  occ[i]   = 1;

  auto unocc = spin_unoccupancy_mask_t(N_orb);
  std::vector<H_contribution_t> result;
  // Phase is always one
  for(auto index : get_dets_index_statisfing_masks(psi, {occ, occ}, {unocc, unocc})) {
    result.push_back({{index, index}, 1});
  }
  return result;
}

std::vector<H_contribution_t> category_B(uint64_t N_orb, eri_4idx_t idx, std::vector<det_t>& psi) {
  const auto& [i, j, k, l] = idx;

  const auto occ    = spin_occupancy_mask_t(N_orb);
  auto occ_i        = spin_occupancy_mask_t(N_orb);
  occ_i[i]          = 1;
  auto occ_j        = spin_occupancy_mask_t(N_orb);
  occ_j[j]          = 1;
  const auto occ_ij = occ_i | occ_j;
  const auto unocc  = spin_unoccupancy_mask_t(N_orb);

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

std::vector<H_contribution_t> category_C(uint64_t N_orb, eri_4idx_t idx, std::vector<det_t>& psi) {
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

  const auto occ   = spin_occupancy_mask_t(N_orb);
  auto occ_a       = spin_occupancy_mask_t(N_orb);
  auto occ_b       = spin_occupancy_mask_t(N_orb);
  auto occ_c       = spin_occupancy_mask_t(N_orb);
  const auto unocc = spin_unoccupancy_mask_t(N_orb);
  auto unocc_b     = spin_unoccupancy_mask_t(N_orb);
  auto unocc_c     = spin_unoccupancy_mask_t(N_orb);
  occ_a[a]         = 1;
  occ_b[b]         = 1;
  occ_c[c]         = 1;
  unocc_b[b]       = 1;
  unocc_c[c]       = 1;

  const auto occ_ab = occ_a | occ_b;
  const auto occ_ac = occ_a | occ_c;

  // aα,bα -> aα,cα
  std::vector<H_contribution_t> result;
  for(auto index : get_dets_index_statisfing_masks(psi, {occ_ab, occ}, {unocc_c, unocc})) {
    const auto& [alpha, beta] = psi[index];
    const auto alpha2         = apply_single_excitation(alpha, b, c);
    const det_t det2{alpha2, beta};
    for(uint64_t index2 = 0; index2 < psi.size(); index2++) {
      if(psi[index2] == det2) { result.push_back({{index, index2}, 1}); }
    }
  }

  // aα,bβ -> aα,cβ
  for(auto index : get_dets_index_statisfing_masks(psi, {occ_a, occ_b}, {unocc, unocc_c})) {
    const auto& [alpha, beta] = psi[index];
    const auto beta2          = apply_single_excitation(beta, b, c);
    const det_t det2{alpha, beta2};
    for(uint64_t index2 = 0; index2 < psi.size(); index2++) {
      if(psi[index2] == det2) { result.push_back({{index, index2}, 2}); }
    }
  }

  // aβ,bα -> aβ,cα
  for(auto index : get_dets_index_statisfing_masks(psi, {occ_b, occ_a}, {unocc_c, unocc})) {
    const auto& [alpha, beta] = psi[index];
    const auto alpha2         = apply_single_excitation(alpha, b, c);
    const det_t det2{alpha2, beta};
    for(uint64_t index2 = 0; index2 < psi.size(); index2++) {
      if(psi[index2] == det2) { result.push_back({{index, index2}, 3}); }
    }
  }
  // aβ,bβ -> aβ,cβ
  for(auto index : get_dets_index_statisfing_masks(psi, {occ, occ_ab}, {unocc, unocc_c})) {
    const auto& [alpha, beta] = psi[index];
    const auto beta2          = apply_single_excitation(beta, b, c);
    const det_t det2{alpha, beta2};
    for(uint64_t index2 = 0; index2 < psi.size(); index2++) {
      if(psi[index2] == det2) { result.push_back({{index, index2}, 4}); }
    }
  }

  return result;
  // rest not yet implemented in C_ijil function

  // aα,cα -> aα,bα
  for(auto index : get_dets_index_statisfing_masks(psi, {occ_ac, occ}, {unocc_b, unocc})) {
    const auto& [alpha, beta] = psi[index];
    const auto alpha2         = apply_single_excitation(alpha, c, b);
    const det_t det2{alpha2, beta};
    for(uint64_t index2 = 0; index2 < psi.size(); index2++) {
      if(psi[index2] == det2) { result.push_back({{index, index2}, 5}); }
    }
  }

  // aα,cβ -> aα,bβ
  for(auto index : get_dets_index_statisfing_masks(psi, {occ_a, occ_c}, {unocc, unocc_b})) {
    const auto& [alpha, beta] = psi[index];
    const auto beta2          = apply_single_excitation(beta, c, b);
    const det_t det2{alpha, beta2};
    for(uint64_t index2 = 0; index2 < psi.size(); index2++) {
      if(psi[index2] == det2) { result.push_back({{index, index2}, 6}); }
    }
  }

  // aβ,cα -> aβ,bα
  for(auto index : get_dets_index_statisfing_masks(psi, {occ_c, occ_a}, {unocc_b, unocc})) {
    const auto& [alpha, beta] = psi[index];
    const auto alpha2         = apply_single_excitation(alpha, c, b);
    const det_t det2{alpha2, beta};
    for(uint64_t index2 = 0; index2 < psi.size(); index2++) {
      if(psi[index2] == det2) { result.push_back({{index, index2}, 7}); }
    }
  }
  // aβ,cβ -> aβ,bβ
  for(auto index : get_dets_index_statisfing_masks(psi, {occ, occ_ac}, {unocc, unocc_b})) {
    const auto& [alpha, beta] = psi[index];
    const auto beta2          = apply_single_excitation(beta, c, b);
    const det_t det2{alpha, beta2};
    for(uint64_t index2 = 0; index2 < psi.size(); index2++) {
      if(psi[index2] == det2) { result.push_back({{index, index2}, 8}); }
    }
  }

  return result;
}


std::vector<H_contribution_t> category_C_ijil(uint64_t N_orb, eri_4idx_t idx,
                                              std::vector<det_t>& psi) {
  const auto& [i, j, k, l] = idx;
  assert(i == k);
  uint64_t a, b, c;
  a = i;
  b = j;
  c = l;

  std::vector<H_contribution_t> result;
  int tmpphase = 1;

  for(size_t spin_a = 0; spin_a < 2; spin_a++) {
    for(size_t spin_bc = 0; spin_bc < 2; spin_bc++) {
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
                {{index0, index1}, tmpphase}); //get_phase_single(det0[spin_bc], b, c)});
          }
        }
      }
      tmpphase += 1;
    }
  }

  return result;
}


spin_det_t get_pm(spin_det_t p) {
  auto n = p.size();
  for(size_t i = 0; (p << (1 << (i))).any(); i++) { p ^= (p << (1 << (i))); }
  return p;
}

int main(int argc, char** argv) {
  int Norb  = std::stoi(argv[1]);
  int Nelec = 4;
  int Ndet  = std::min(10, (int)binom(Norb, Nelec));
  std::vector<det_t> psi;
  for(int i = 0; i < Ndet; i++) {
    auto d1 = combi(i, Norb, Nelec);
    for(int j = 0; j < Ndet; j++) {
      auto d2 = combi(j, Norb, Nelec);
      psi.push_back({d1, d2});
    }
  }

  // test combi/unchoose
  for(int i = 0; i < binom(Norb, Nelec); i++) {
    assert(unchoose(Norb, combi(i, Norb, Nelec)) == i);
    std::cout << i << " " << combi(i, Norb, Nelec) << std::endl;
  }
  // return 0;

  auto res1 = category_C(Norb, {1, 2, 1, 3}, psi);
  auto res2 = category_C_ijil(Norb, {1, 2, 1, 3}, psi);

  assert(res1.size() == res2.size());

  for(size_t i = 0; i < res1.size(); i++) {
    auto& [pd1, ph1] = res1[i];
    assert(res1[i] == res2[i]);
    std::cout << i << "\t: " << psi[pd1.first] << " " << psi[pd1.second] << "\t" << ph1
              << std::endl;
  }
  return 0;
}
