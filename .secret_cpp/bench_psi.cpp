#include <random>
#include <algorithm>
#include "qpx.hpp"
#include "psi.hpp"
#include <benchmark/benchmark.h>
#include <unordered_set>
#include <tsl/hopscotch_set.h>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>

spin_det_t parse_line(std::string s) {
  spin_det_t sd;
  for(int i = 0; i < s.size(); i++) {
    if(s[i] == '-') sd.push_back(0);
    if(s[i] == '+') sd.push_back(1);
  }
  return sd;
}

std::vector<det_t> setup() {
  std::vector<det_t> psi;

  std::ifstream fs("/home/applenco/QuantumEnvelope/data/c2_eq_hf_dz_15.101224det.wf");
  while(true) {
    std::string coef_str_or_empty;
    if(!std::getline(fs, coef_str_or_empty)) break;

    std::string alpha_str;
    std::getline(fs, alpha_str);
    parse_line(alpha_str);
    std::string beta_str;
    std::getline(fs, beta_str);
    std::string emptyline_str;
    std::getline(fs, emptyline_str);

    psi.push_back({parse_line(alpha_str), parse_line(beta_str)});
  }
  return psi;
}

bool get_element_naive(det_t target, std::vector<det_t>& psi) {
  for(auto& det : psi) {
    if(det == target) return true;
  }
  return false;
}

static void NaiveLookup(benchmark::State& state) {
  auto psi = setup();
  std::sort(psi.begin(), psi.end());


  std::vector<det_t> psi_random{psi.begin(), psi.end()};
  std::shuffle(psi_random.begin(), psi_random.end(), std::default_random_engine{});

  for(auto _ : state) {
    for(auto d : psi_random) {
      bool f = get_element_naive(d, psi);
      benchmark::DoNotOptimize(f);
    }
  }
  state.counters["LookupRate"] =
      benchmark::Counter(psi.size(), benchmark::Counter::kIsIterationInvariantRate);
}

static void BinarySearchLookup(benchmark::State& state) {
  auto psi = setup();
  std::sort(psi.begin(), psi.end());


  std::vector<det_t> psi_random{psi.begin(), psi.end()};
  std::shuffle(psi_random.begin(), psi_random.end(), std::default_random_engine{});

  for(auto _ : state) {
    for(auto d : psi_random) {
      bool f = std::binary_search(psi.begin(), psi.end(), d);
      benchmark::DoNotOptimize(f);
    }
  }
  state.counters["LookupRate"] =
      benchmark::Counter(psi.size(), benchmark::Counter::kIsIterationInvariantRate);
}

static void BinarySearchesLookup(benchmark::State& state) {
  auto psi = setup();
  std::sort(psi.begin(), psi.end());

  std::vector<det_t> vals{psi.begin(), psi.end()};
  std::vector<bool> result(vals.size(), false);

  for(auto _ : state) { binary_searchs(psi, vals, result); }

  state.counters["LookupRate"] =
      benchmark::Counter(psi.size(), benchmark::Counter::kIsIterationInvariantRate);
}


static void stdHashLookup(benchmark::State& state) {
  auto psi = setup();
  std::unordered_set<det_t> psi_s;
  for(auto d : psi) { psi_s.insert(d); }

  std::vector<det_t> psi_random{psi.begin(), psi.end()};
  std::shuffle(psi_random.begin(), psi_random.end(), std::default_random_engine{});

  for(auto _ : state) {
    for(auto d : psi_random) {
      bool f = (psi_s.find(d) != psi_s.end());
      benchmark::DoNotOptimize(f);
    }
  }
  state.counters["LookupRate"] =
      benchmark::Counter(psi.size(), benchmark::Counter::kIsIterationInvariantRate);
}

static void tslHashLookup(benchmark::State& state) {
  auto psi = setup();
  tsl::hopscotch_set<det_t> psi_s;
  for(auto d : psi) { psi_s.insert(d); }

  std::vector<det_t> psi_random{psi.begin(), psi.end()};
  std::shuffle(psi_random.begin(), psi_random.end(), std::default_random_engine{});

  for(auto _ : state) {
    for(auto d : psi_random) {
      bool f = (psi_s.find(d) != psi_s.end());
      benchmark::DoNotOptimize(f);
    }
  }
  state.counters["LookupRate"] =
      benchmark::Counter(psi.size(), benchmark::Counter::kIsIterationInvariantRate);
}

// Register the function as a benchmark
BENCHMARK(NaiveLookup);
BENCHMARK(BinarySearchLookup);
BENCHMARK(BinarySearchesLookup);
BENCHMARK(stdHashLookup);
BENCHMARK(tslHashLookup);
// Run the benchmark
BENCHMARK_MAIN();
