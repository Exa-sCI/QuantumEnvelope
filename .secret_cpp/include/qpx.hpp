#pragma once

#include <array>
#include <sul/dynamic_bitset.hpp>
#include <functional>
#include <array>
#include <iostream>

typedef sul::dynamic_bitset<> spin_det_t;

template<>
struct std::hash<spin_det_t> {
  std::size_t operator()(spin_det_t const& s) const noexcept {
    return std::hash<std::string>()(s.to_string());
  }
};

#define N_SPIN_SPECIES 2

class det_t { // The class

public: // Access specifier
  spin_det_t alpha;
  spin_det_t beta;

  det_t(spin_det_t _alpha, spin_det_t _beta) {
    alpha = _alpha;
    beta  = _beta;
  }

  bool operator<(const det_t& b) const {
    if(alpha == b.alpha) return (beta < b.beta);
    return (alpha < b.alpha);
  }

  bool operator==(const det_t& b) const { return (alpha == b.alpha) && (beta == b.beta); }

  spin_det_t& operator[](unsigned i) {
    assert(i < N_SPIN_SPECIES);
    switch(i) {
      case 0: return alpha;
      // Avoid `non-void function does not return a value in all control paths`
      default: return beta;
    }
  }
  // https://stackoverflow.com/a/27830679/7674852 seem to recommand doing the other way arround
  const spin_det_t& operator[](unsigned i) const { return (*this)[i]; }
};

template<>
struct std::hash<det_t> {
  std::size_t operator()(det_t const& s) const noexcept {
    std::size_t h1 = std::hash<spin_det_t>{}(s.alpha);
    std::size_t h2 = std::hash<spin_det_t>{}(s.beta);
    return h1 ^ (h2 << 1);
  }
};

// Should be moved in the cpp of det
inline std::ostream& operator<<(std::ostream& os, const det_t& obj) {
  return os << "(" << obj.alpha << "," << obj.beta << ")";
}

typedef sul::dynamic_bitset<> spin_occupancy_mask_t;
typedef std::array<spin_occupancy_mask_t, N_SPIN_SPECIES> occupancy_mask_t;

typedef sul::dynamic_bitset<> spin_unoccupancy_mask_t;
typedef std::array<spin_unoccupancy_mask_t, N_SPIN_SPECIES> unoccupancy_mask_t;

typedef std::array<uint64_t, 4> eri_4idx_t;
