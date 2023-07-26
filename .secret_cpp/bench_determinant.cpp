#include "qpx.hpp"
#include <determinant.hpp>
#include <benchmark/benchmark.h>

static void ComputePhaseAndApplyExcitation(benchmark::State& state) {

  const auto n_orbital = 1024;
  const auto n_elec = 129;

  auto alpha = spin_det_t(n_orbital);
  auto beta = spin_det_t(n_elec);
  alpha.set(0, n_elec, 1);
  
  det_t s{alpha, beta};
  for(auto _ : state) {
      	 auto d = apply_single_spin_excitation(s, 0, 0, n_elec+1);
	 benchmark::DoNotOptimize(d);
  }
}

BENCHMARK(ComputePhaseAndApplyExcitation);
BENCHMARK_MAIN();
