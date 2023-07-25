#pragma once

template<class ForwardIterator, class ForwardIteratorBool>
void binary_searchs(ForwardIterator first, ForwardIterator last, ForwardIterator v_first,
                    ForwardIterator v_last, ForwardIterator v, ForwardIteratorBool i) {
  // Return:  `*v` ∈? [first, last)
  // Precondition: `v` ∈ `v_first`, `v_last`.

  // pos of `*v` inside `v` array using binary search
  auto pos = std::lower_bound(first, last, *v);
  // Find it or not find it?
  *i = (pos != last && !(*v < *pos));
  // Find new values to search for
  // v_first     v                v_last
  // |-----------|----------------|
  {
    // Botom Half [v_first, v)
    // End the recursion when v_first == v, we search for all our values
    // if distance == 1, need to search for v_first
    auto d = std::distance(v_first, v);
    if(pos != first && d > 1) binary_searchs(first, pos, v_first, v - 1, v - d / 2, i - d / 2);
  }
  {
    // Top Half (v, v_last)
    // We cannot deference v_last, so we stop the recursion when d == 1
    auto d = std::distance(v, v_last);
    if(pos != last && d > 1) binary_searchs(pos, last, v + 1, v_last, v + d / 2, i + d / 2);
  }
}

template<class T1, class T2, class T3>
void binary_searchs(T1& enumerable, T2& values, T3& result) {
  binary_searchs(enumerable.begin(), enumerable.end(), values.begin(), values.end(),
                 values.begin() + values.size() / 2, result.begin() + values.size() / 2);
}
