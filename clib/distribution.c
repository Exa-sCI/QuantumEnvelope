#include "mpi.h"

/* scalapack doesn't provide a header, so simplify our life a bit
 * WARNING: needs to use FC_FUNC in the future */
void blacs_get_(int *, int *, int *);
void blacs_grindinfo_(int *, int *, int *, int *, int *);
int numroc_(int *, int *, int *, int *, int *);

struct dist_ctxt {
	int slctxt;
	int nrow;
	int ncol;
};

/* scalapack context init */
int qpx_context_init(void *ctxt, int nrow, int ncol)
{
	/* fortran fun, need variables holding specific values */
	int zero = 0;
	int layout = 'R';
	ctxt = malloc(sizeof(struct dist_ctxt));
	ctxt.nrow = nrow;
	ctxt.ncol = ncol;
	
	/* retrieve the default context */
	blacs_get_(&zero, &zero, &ctxt.slctxt);

	/* initialize grid, row major */
	blacs_gridinit_(&ctxt->slctxt, &layout, &ctxt->nrow, &ctxt->ncol);

	return 0;
}

/* process grid info */
int qpx_grid_info(void *ctxt, int *myrow, int *mycol)
{
	int zero = 0;
	blacs_gridinfo_(&ctxt->slctxt, &ctxt->nrow, &ctxt->ncol, myrow, mycol);
	return 0;
}

/* from process grid coordinates to local block dimensions */
int qpx_numroc(void *ctxt, int n, int bsize, int r, int c, int *rowsize, int *colsize)
{
	int zero = 0;
	/* in theory blacs can handle partial blocks, so that should be fine */
	*rowsize = numroc_(&n, &bsize, &r, &zero, &ctxt->nrow);
	*colsize = numroc_(&n, &bsize, &c, &zero, &ctxt->ncol);
	return 0;
}

int qpx_startroc(void *ctxt, int n, int bsize, int r, int c, int *startr, int
		 *startc)
{
	*startr = r * bsize;
	*startc = c * bsize;
	return 0;
}
