CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LDFLAGS = -lm

symnmf: symnmf.o
	$(CC) -o symnmf symnmf.o $(CFLAGS) $(LDFLAGS)

symnmf.o: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -c symnmf.c 

clean:
	rm -f *.o symnmf