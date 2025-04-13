# Excov

This package is an implementation of [Knuth's Algorithm
X](https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X) using
[DLX](https://en.wikipedia.org/wiki/Dancing_Links).

This is just my little starter project to explore Rust.  That said,
the implementation is meant to be relatively simple and efficient, and
I'd deeply appreciate any comments on how to make the code better.

Knuth's Algorithm X solves [exact
cover](https://en.wikipedia.org/wiki/Exact_cover) problems -- see the
wikipedia article for details.  But basically: if you have a set X,
and a set S of subsets of X (so, each element of S contains some
number of elements of X), can you find a subset of S s.t. each element
of X is contained in exactly one subset?  Essentially, this partitions
X, with no elements in X left over.

This is useful for things like solving
[Sudoku](https://en.wikipedia.org/wiki/Exact_cover#Sudoku) puzzles or
the [N Queens](https://en.wikipedia.org/wiki/Eight_queens_puzzle)
puzzle.
