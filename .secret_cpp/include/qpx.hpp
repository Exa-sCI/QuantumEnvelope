#include <array>
#include <sul/dynamic_bitset.hpp>
#include <unordered_map>


typedef sul::dynamic_bitset<> spin_det_t;

template<>
struct std::hash<spin_det_t> {
  std::size_t operator()(spin_det_t const& s) const noexcept {
    return std::hash<std::string>()(s.to_string());
  }
};
#define N_SPIN_SPECIES 2
//typedef std::array<spin_det_t, 2> det_t;

class det_t {       // The class

  public:          // Access specifier
	spin_det_t alpha;
	spin_det_t beta;

	  det_t(spin_det_t _alpha, spin_det_t _beta) {
		alpha = _alpha;
		beta = _beta;
	}

    bool operator < (   const det_t &b  ) const {
	return (alpha < b.alpha) && (beta < b.beta) ;
    }

    bool operator == (   const det_t &b  ) const {
        return (alpha == b.alpha) && (beta == b.beta) ;
    }

};

template<>
struct std::hash<det_t> {
  std::size_t operator()(det_t const& s) const noexcept {
    std::size_t h1 = std::hash<spin_det_t>{}(s.alpha);
    std::size_t h2 = std::hash<spin_det_t>{}(s.beta);
    return h1 ^ (h2 << 1);
  }
};
std::ostream& operator<<(std::ostream& os, const det_t& obj) {
  return os << "(" << obj.alpha << "," << obj.beta << ")";
}

typedef sul::dynamic_bitset<> spin_occupancy_mask_t;
typedef std::array<spin_occupancy_mask_t, N_SPIN_SPECIES> occupancy_mask_t;

typedef sul::dynamic_bitset<> spin_unoccupancy_mask_t;
typedef std::array<spin_unoccupancy_mask_t, N_SPIN_SPECIES> unoccupancy_mask_t;

typedef std::array<uint64_t, 4> eri_4idx_t;


