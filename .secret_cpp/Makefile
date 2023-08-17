CXXFLAGS_TEST_MAIN=-fsanitize=address -fno-omit-frame-pointer -Wno-tautological-constant-compare
CXXFLAGS_TEST_NO_MAIN=$(CXXFLAGS_TEST_MAIN) -DDOCTEST_CONFIG_DISABLE

CXXFLAGS_BENCH=-O3 -DDOCTEST_CONFIG_DISABLE -I include/ -I ~/benchmark/include/ -L ~/benchmark/build/src/ -lbenchmark -lpthread

test_determinant.o: determinant.cpp
	icpx $(CXXFLAGS_TEST_NO_MAIN) -I include/ $^ -c -o $@

test_category.o: category.cpp
	icpx $(CXXFLAGS_TEST_NO_MAIN) -I include/ $^ -c -o $@

test_psi.o: psi.cpp
	icpx $(CXXFLAGS_TEST_NO_MAIN) -I include/ $^ -c -o $@


test_determinant: determinant.cpp
	icpx $(CXXFLAGS_TEST_MAIN) -I include/ $^ -o $@

test_category: category.cpp test_determinant.o
	icpx $(CXXFLAGS_TEST_MAIN)  -I include/ $^ -o $@

test_psi: psi.cpp
	icpx $(CXXFLAGS_TEST_MAIN)  -I include/ $^ -o $@

run_test: test_determinant test_category test_psi
	./test_determinant
	./test_category
	./test_psi

bench_psi: bench_psi.cpp psi.cpp
	icpx $^ $(CXXFLAGS_BENCH)  -o $@

bench_determinant: bench_determinant.cpp determinant.cpp
	icpx $^ $(CXXFLAGS_BENCH) -o $@

run_bench: bench_psi bench_determinant
	./bench_psi
	./bench_determinant
