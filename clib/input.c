#define _GNU_SOURCE
#include <archive.h>
#include <archive_entry.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

double *one_e_integral;
double *two_e_integral;


int load_archive(char *path, char **file, size_t *length)
{
	int n_orb;
	int i,j,k,l;
	int res;

	struct archive *a;
	struct archive_entry *entry;

	const struct stat *stats;
	int64_t len;

	/* open an archive, of any type */
	a = archive_read_new();
	archive_read_support_filter_all(a);
	archive_read_support_format_all(a);
	res = archive_read_open_filename(a, path, 4096);
	assert(res == ARCHIVE_OK);

	/* we only care about the first file */
	res = archive_read_next_header(a, &entry);
	assert(res == ARCHIVE_OK);

	/* extra byte for NULL terminator */
	stats = archive_entry_stat(entry);
	*file = malloc(stats->st_size + 1);
	assert(*file != NULL);

	len = archive_read_data(a, *file, stats->st_size);
	assert(len == stats->st_size);

	*length = len;
	*file[len] = 0;
	return 0;
}
