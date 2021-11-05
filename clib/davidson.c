#include <cblas.h>
#include <lapacke.h>
#include <mpi.h>
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define MPI_CHECK(expr)                \
do {                                   \
	assert(MPI_SUCCESS == (expr)); \
} while (0)

struct davidson_state {
	size_t problem_size;
	MPI_Comm comm;
	int rank;
	int world_size;
	int local_size;
	int *recvcounts;
	int *displs;
};

void split_problem(size_t problem_size, int world_size, int *recvcounts) {
	size_t ws = (size_t)world_size;
	size_t min = (problem_size/ws);
	size_t max = min + 1;
	size_t rest = problem_size % ws;
	assert(max <= INT_MAX);
	for (size_t i = 0; i < rest; i++)
		recvcounts[i] = (int)max;
	for (size_t i = rest; i < ws ; i++)
		recvcounts[i] = (int)min;
}

void fill_displs(int world_size, int *recvcounts, int *displs) {
	displs[0] = 0;
	for (size_t i = 1; i < (size_t)world_size; i++)
		displs[i] = recvcounts[i-1] + displs[i-1];
}

struct davidson_state * qpxDavidsonInit(MPI_Comm comm, size_t problem_size) {
	int world_size, rank;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &world_size);
	struct davidson_state *state =
		(struct davidson_state *)malloc(sizeof(struct davidson_state) +
		world_size * 2 * sizeof(int));
	state->problem_size = problem_size;
	state->comm = comm;
	state->rank = rank;
	state->world_size = world_size;
	state->recvcounts = (int *)(state + 1);
	state->displs = state->recvcounts + world_size;
	split_problem(problem_size, world_size, state->recvcounts);
	state->local_size = state->recvcounts[state->rank];
	fill_displs(world_size, state->recvcounts, state->displs);
	return state;
}

void qpxDavidsonDestroy(struct davidson_state *state) {
	free(state);
}

/*  A_i[state->problem_size][state->local_size]
*/
struct argsort {
	double v;
	double *pvec;
};

int cmp_argsort(void * a, void * b) {
	struct argsort *as = (struct argsort *)a;
	struct argsort *bs = (struct argsort *)b;
	return as->v < bs->v ? -1 : as->v > bs->v ? 1 : 0;
}

void qpxDavidsonDistributed(struct davidson_state *state, double *A_i, int n_eig, double eps, int max_iter, int m, int q) {
	double *V_ik, *Wik, *r_ik, *t_ik;
	double *x_1;
	size_t local_size = state->local_size;

	// not max_iter but m + q
	V_ik = (double *)malloc(local_size * max_iter * sizeof(double));
	W_ik = (double *)malloc(local_size * max_iter * sizeof(double));
	r_ik = (double *)malloc(local_size * sizeof(double));
	t_ik = (double *)malloc(local_size * sizeof(double));

	double * r_k, V_k;
	r_k = (double *)malloc(state->problem_size * sizeof(double));
	V_k = (double *)malloc(state->problem_size * sizeof(double));

	double *L_k, *Y_k;
	L_k = (double *)malloc((m + q) * sizeof(double));
	Y_k = (double *)malloc((m + q) * (m + q) * sizeof(double));
	struct argsort * argsort_vals = (struct argsort *)malloc((m + q) * sizeof(struct argsort));

	lapack_int seed = 1234;
	lapack_int res;
	res = LAPACKE_dlarnv(3, &seed, local_size, V_ik);
	assert(res == 0);
	double nrm = cblas_dnrm2(local_size, V_ik, 1);
	assert(nrm > 0.0);
	cblas_dscal(local_size, 1.0/nrm, V_ik, 1);
//CblasColMajor
	int dim_S = 0;
	for (int j = 0; j < n_eig; j++) {
		for (int k = 0; k < max_iter; k++) {
			MPI_CHECK(MPI_Allgatherv(V_ik + dim_S * local_size, local_size, MPI_DOUBLE, V_k,
			                         state->recvcounts, state->displs, MPI_DOUBLE, state->comm));
			clbas_dgemv(CblasRowMajor, CblasTrans, state->problem_size, local_size, 1.0, A_i, state->problem_size,
			            V_k, 1, 0.0, W_ik + dim_S * local_size, 1);

			double s_k = cblas_ddot(local_size, V_ik + dim_S * local_size, 1, W_ik + dim_S * local_size, 1);
			double sum;
			MPI_CHECK(MPI_Alleeduce(&s_k, &sum, 1, MPI_DOUBLE, MPI_SUM, state->comm));

			if (k == 0) {
				L_k[0] = s_k;
				Y_k[0] = 1.0;
			} else
				parallel_arrowhead_decomposition(state, dim_S, Y_k, s_k, L_k);
			dim_S += 1;

			parallel_restart(m, q, &dim_S, L_k, Y_k, &dim_S, V_ik, W_ik);

			int lmin = dim_S < j ? dim_S : j;
			for (int i = 0; i < dim_S; i++)
				argsort_vals[i] = {L_k[i], Y_k + dim_S * i};
			qsort(argsort_vals, dim_S, sizeof(struct argsort), &cmp_argsort);
			
			clbas_dgemv(CblasColMajor, CblasTrans, dim_S, local_size, 1.0, W_ik, dim_S,
			            argsort_vals[dim_S - 1].pvec, 1, 0.0, r_ik, 1);
			clbas_dgemv(CblasColMajor, CblasTrans, dim_S, local_size, -argsort_vals[dim_S - 1].v, Vik, dim_S,
			            argsort_vals[dim_S - 1].pvec, 1, 1.0, r_ik, 1);
			MPI_CHECK(MPI_Allgatherv(r_ik, local_size, MPI_DOUBLE, r_k,
			          state->recvcounts, state->displs, MPI_DOUBLE, state->comm));

			nrm = cblas_dnrm2(state->problem_size, r_k, 1);
			if (nrm < eps)
				break;
			preconditioning(state, A_i, argsort_vals[dim_S - 1].pvec, r_ik, t_ik);
			mgs(state, dim_S, V_ik, t_ik);
		}
	}
	return 
}

int main(int argc, char *argv[]) {
	MPI_Init( &argc, &argv );
	struct davidson_state *state = qpxDavidsonInit(MPI_COMM_WORLD, (1 << 8) - 3);
	if (state->rank == 0)
		for (int i = 0; i < state->world_size; i++)
			printf("%d (%d) \n", state->recvcounts[i], state->displs[i]);
	MPI_Finalize();
	return 0;
}
