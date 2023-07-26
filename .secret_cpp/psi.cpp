#include "qpx.hpp"
#include "psi.hpp"
#include <algorithm>
#include <iterator>
#if !defined(DOCTEST_CONFIG_DISABLE)
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#endif
#include <doctest/doctest.h>
#include <sstream>

namespace doctest {
template<typename T>
struct StringMaker<std::vector<T>> {
  static String convert(const std::vector<T>& in) {
    std::ostringstream oss;

    oss << "[";
    for(auto it = in.begin(); it < in.end(); ++it) {
      oss << *it;
      if(it + 1 != in.end()) oss << ", ";
    }
    oss << "]";

    return oss.str().c_str();
  }
};
} // namespace doctest

TEST_CASE("testing binary_searches") {
  std::vector<int> source{1, 2, 3, 4, 6};
  SUBCASE("1 element") {
    std::vector<int> vals(source.begin(), source.begin() + 1);
    std::vector<bool> results(vals.size(), false);
    std::vector<bool> expected(vals.size(), true);
    binary_searchs(source, vals, results);
    CHECK(results == expected);
  }


  SUBCASE("2 elements") {
    std::vector<int> vals(source.begin(), source.begin() + 2);
    std::vector<bool> results(vals.size(), false);
    std::vector<bool> expected(vals.size(), true);
    binary_searchs(source, vals, results);
    CHECK(results == expected);
  }


  SUBCASE("3 elements") {
    std::vector<int> vals(source.begin(), source.begin() + 3);
    std::vector<bool> results(vals.size(), false);
    std::vector<bool> expected(vals.size(), true);
    binary_searchs(source, vals, results);
    CHECK(results == expected);
  }

  SUBCASE("1 element non present") {
    std::vector<int> vals{0};
    std::vector<bool> results(vals.size(), false);
    std::vector<bool> expected{false};
    binary_searchs(source, vals, results);
    CHECK(results == expected);
  }

  SUBCASE("Shortcut") {
    std::vector<int> vals{0, 1, 5, 6, 7, 8, 9};
    std::vector<bool> results(vals.size(), false);
    std::vector<bool> expected{false, true, false, true, false, false, false};
    binary_searchs(source, vals, results);
    CHECK(results == expected);
  }
}
