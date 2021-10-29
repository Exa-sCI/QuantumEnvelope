CC=mpicc
LD=mpicc

CFLAGS= -fPIC -Wall -std=c99 -pedantic -Werror
LDFLAGS= -fPIC

CLIB_SOURCES= clib/davidson.c clib/excitation.c
CLIB_OBJS= $(patsubst %.c,%.o,$(CLIB_SOURCES))

all: libqpx.so

libqpx.so: $(CLIB_OBJS)
	$(CC) -o $@ -shared $^ $(LDFLAGS)
%.o: %.c
	$(CC) -c $< $(CFLAGS)
