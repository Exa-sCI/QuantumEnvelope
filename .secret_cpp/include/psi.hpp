#pragma once

template<class ForwardIterator, class ForwardIteratorBool>
void binary_searchs_imp(ForwardIterator first, ForwardIterator last, ForwardIterator v_first,
                        ForwardIterator v_last, ForwardIterator v, ForwardIteratorBool i) {
  // Return:       *v ∈? [first, last).
  // Precondition: v ∈ [v_first, v_last)

  // pos is pointer ∈ [first, last) pointing `*v`.
  auto pos = std::lower_bound(first, last, *v);
  // Lazy: only set i, if found
  if(pos != last && !(*v < *pos)) *i = true;

  // Find the new `v` to search for
  // v_first     v                v_last
  // |-----------|----------------|
  {
    // Botom Half [v_first, v)
    // End the recursion when v_first == v, we search for all our values
    // if distance == 1, need to search for v_first
    auto d = std::distance(v_first, v);
    if(pos != first && d > 0)
      binary_searchs_imp(first, pos, v_first, v - 1, v - (d + 1) / 2, i - (d + 1) / 2);
  }
  {
    // Top Half (v, v_last)
    // We cannot deference v_last, so we stop the recursion when d == 1
    auto d = std::distance(v, v_last);
    if(pos != last && d > 1) binary_searchs_imp(pos, last, v + 1, v_last, v + d / 2, i + d / 2);
  }
}

template<class T1, class T2>
void binary_searchs(T1& enumerable, T1& values, T2& result) {
  binary_searchs_imp(enumerable.begin(), enumerable.end(), values.begin(), values.end(),
                     values.begin() + values.size() / 2, result.begin() + values.size() / 2);
}
