#![doc = include_str!("../README.md")]
//!
//!
//! To solve an exact cover problem with this implementation,
//! instantiate a [`Problem`] using [`Problem::new()`], add
//! constraints to describe the problem, and call [`solve()`] to
//! produce one or more solutions.
//!
//! An example, based on Wikipedia's [Algorithm
//! X](https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X) article:
//!
//! ```
//! // This is the example problem from
//! // <https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X>; it
//! // makes for a nice test case, since the article explains
//! // every step of the execution.
//!
//! let mut problem = excov::Problem::new(excov::DenseOptMapper{});
//!
//! problem.add_constraint([0, 1]);
//! problem.add_constraint([4, 5]);
//! problem.add_constraint([3, 4]);
//! problem.add_constraint([0, 1, 2]);
//! problem.add_constraint([2, 3]);
//! problem.add_constraint([2, 3]);
//! problem.add_constraint([3, 4]);
//! problem.add_constraint([0, 2, 4, 5]);
//!
//! // And then, actually solve the problem.
//! let mut solution_count = 0;
//! for soln in excov::solve(problem) {
//!     solution_count += 1;
//!     assert_eq!(
//!         vec![1usize, 3, 5],
//!         soln,
//!         "solution: expected(left) != actual(right)"
//!     )
//! };
//!
//! assert_eq!(1, solution_count);
//! ```
//!
//! For more expressiveness in option identifiers, it may be useful to
//! use [`HashOptMapper`], which lets you use any option
//! identifier that can be stored as a key in a
//! [`std::collections::HashMap`].

use std::{collections, hash, iter};

/// A trait for types that map between opaque solution-specific option
/// identifiers and the dense option identifier space the problem
/// solver requires for representing options.
pub trait OptMapper {
    type InOptID;
    type OutOptID;

    fn to_dense_id(&mut self, obj_id: Self::InOptID) -> usize;
    fn to_option_id(&mut self, dense_id: usize) -> Self::OutOptID;
}

/// An implemention of [`OptMapper`], where the option identifiers are
/// already dense [`usize`] values; this implementation simply uses
/// the opaque option identifier as the dense integer value.
pub struct DenseOptMapper {}

impl OptMapper for DenseOptMapper {
    type InOptID = usize;
    type OutOptID = usize;

    fn to_dense_id(&mut self, obj_id: Self::InOptID) -> usize {
        obj_id
    }
    fn to_option_id(&mut self, dense_id: usize) -> Self::OutOptID {
        dense_id
    }
}

/// An implemention of [`OptMapper`], using a
/// [`std::collections::HashMap`] to manage the option mapping.
pub struct HashOptMapper<K>
where
    K: Copy + Eq + hash::Hash,
{
    opt_to_dense: collections::HashMap<K, usize>,
    dense_to_opt: Vec<K>,
}

impl<K> HashOptMapper<K>
where
    K: Copy + Eq + hash::Hash,
{
    pub fn new() -> HashOptMapper<K> {
        HashOptMapper {
            opt_to_dense: collections::HashMap::new(),
            dense_to_opt: Vec::new(),
        }
    }
}

impl<K> OptMapper for HashOptMapper<K>
where
    K: Copy + Eq + hash::Hash,
{
    type InOptID = K;
    type OutOptID = K;

    fn to_dense_id(&mut self, obj_id: Self::InOptID) -> usize {
        if let Some(dense_id) = self.opt_to_dense.get(&obj_id) {
            *dense_id
        } else {
            let dense_id = self.dense_to_opt.len();
            self.dense_to_opt.push(obj_id);
            self.opt_to_dense.insert(obj_id, dense_id);
            dense_id
        }
    }

    fn to_option_id(&mut self, dense_id: usize) -> Self::OutOptID {
        self.dense_to_opt[dense_id]
    }
}

struct Location {
    constraint: usize,
    node: usize,
}

/// An iterator yielding solutions to an exact cover problem.  This is
/// typically created by [`solve()`].
pub struct SolutionIterator<M>
where
    M: OptMapper,
{
    problem: Problem<M>,
    location: Vec<Location>,
    solution: Vec<usize>,
}

/// Represention of an exact cover problem to be solved by [`solve()`].
///
/// A problem consists of a set of _constraints_, and a set of
/// _options_; each option covers (satisfies) some number of
/// constraints.
///
/// A solution to the exact cover problem is a subset of the options
/// s.t. all of the mandatory constraints are covered by exactly one
/// option, and all optional constraints are covered by at most one
/// option.
///
/// For example:
///
///   In the classic [N Queens
///   puzzle](https://en.wikipedia.org/wiki/Eight_queens_puzzle), an
///   NxN chessboard is to be covered by N queens.
///
///   Each square on the board is an option for a queen; a queen in a
///   square covers its row, its column, and all squares reachable on
///   the diagonals.
///
///   So the rows and columns of the board are the constraints that
///   must be covered to have a valid solution (each row and column
///   must have a queen); diagonals are not required to have queens,
///   but if they do, they must have at most one, making them optional
///   constraints.
///
///   So to solve the N Queens problem with this implementation, one
///   would:
///
///   * Instantiate a [`Problem`],
///
///   * Call [`Problem::add_constraint()`] for each row and column
///     (specifying the squares in that row/column as the options that
///     cover that row/column),
///
///   * Call [`Problem::add_optional_constraint`] for each diagonal
///     (specifying the squares that make up that diagonal)
///
///   * Call [`solve()`] to generate the solutions.
pub struct Problem<M: OptMapper> {
    // NB In any other language, I'd probably implement this using
    // cross-threaded doubly-linked lists.  In Rust, the borrowing
    // rules for that get Tricky (the nodes need to be mutable from a
    // variety of directions).  There are ways around this (arenas,
    // graph packages...), but this is a pretty simple situation, so
    // we just manage the datastructure directly.
    //
    // So instead of using pointers, our "links" are all indices into
    // a vector, which acts as the node owner.  The algorithm is
    // otherwise unchanged.
    //
    // Note that as long as we're using a vector as the allocation
    // arena anyway, we may as well allocate all of the nodes
    // contiguously within that vector; the resize expense is
    // neglibible, and this has at least the possibility of packing
    // the node data nicely into the processor cache.
    /// The OptID mapper for this Problem, used to translate between
    /// the solution's options and the dense usize space we use for
    /// indexing those options.
    mapper: M,

    /// The node allocation arena.  NB: Index zero is *always* the
    /// index of the head node, which keeps track of the active constraint
    /// set.
    nodes: Vec<Node>,

    /// The options -- a mapping from a dense integer option identifier
    /// to the index of the first node that's a member of the option.
    options: Vec<Option<usize>>,
}

/// Node is a node in the sparse graph associating the options with
/// the constraints.
struct Node {
    constraint_or_node_count: usize,
    option: usize,
    up: usize,
    down: usize,
    left: usize,
    right: usize,
}

impl<M> Problem<M>
where
    M: OptMapper,
{
    /// Initializes a new, empty, [`Problem`] instance.
    pub fn new(mapper: M) -> Problem<M> {
        let mut problem = Problem {
            mapper,
            nodes: Vec::new(),
            options: Vec::new(),
        };
        problem.nodes.push(Node {
            constraint_or_node_count: 0,
            option: 0,
            up: 0,    // The index of the previous node in the same constraint
            down: 0,  // The index of the next node in the same constraint
            left: 0,  // The index of the previous node in the same option
            right: 0, // The index of the next node in the same option
        });
        problem
    }

    // Basic accessor methods, hiding most of the direct accesses of
    // the nodes vector (which would otherwise make the code a little
    // harder to read & understand).

    fn node_count(&mut self, node: usize) -> usize {
        self.nodes[node].constraint_or_node_count
    }

    fn set_node_count(&mut self, node: usize, val: usize) {
        self.nodes[node].constraint_or_node_count = val
    }

    fn incr_node_count(&mut self, node: usize) {
        self.nodes[node].constraint_or_node_count += 1
    }

    fn decr_node_count(&mut self, node: usize) {
        self.nodes[node].constraint_or_node_count -= 1
    }

    fn constraint(&mut self, node: usize) -> usize {
        self.nodes[node].constraint_or_node_count
    }

    fn option_id(&mut self, node: usize) -> usize {
        self.nodes[node].option
    }

    fn up(&mut self, node: usize) -> usize {
        self.nodes[node].up
    }

    fn set_up(&mut self, node: usize, val: usize) {
        self.nodes[node].up = val
    }

    fn down(&mut self, node: usize) -> usize {
        self.nodes[node].down
    }

    fn set_down(&mut self, node: usize, val: usize) {
        self.nodes[node].down = val
    }

    fn left(&mut self, node: usize) -> usize {
        self.nodes[node].left
    }

    fn set_left(&mut self, node: usize, val: usize) {
        self.nodes[node].left = val
    }

    fn right(&mut self, node: usize) -> usize {
        self.nodes[node].right
    }

    fn set_right(&mut self, node: usize, val: usize) {
        self.nodes[node].right = val
    }

    // A few utility link/unlink functions.

    fn unlink_from_option(&mut self, node: usize) {
        let left = self.left(node);
        let right = self.right(node);
        self.set_right(left, right);
        self.set_left(right, left);
    }

    fn relink_into_option(&mut self, node: usize) {
        let left = self.left(node);
        let right = self.right(node);
        self.set_left(right, node);
        self.set_right(left, node);
    }

    fn unlink_from_constraint(&mut self, node: usize) {
        let up = self.up(node);
        let down = self.down(node);
        self.set_down(up, down);
        self.set_up(down, up);
    }

    fn relink_into_constraint(&mut self, node: usize) {
        let up = self.up(node);
        let down = self.down(node);
        self.set_up(down, node);
        self.set_down(up, node);
    }

    /// Adds a mandatory constraint to the solution search
    /// space.
    ///
    /// A constraint is a set of options of the overall problem
    /// universe; each option is identified by a [`usize`].  Each
    /// member of the set is incompatible with the others within the
    /// constraint.
    ///
    /// For a mandatory constraint, exactly one option within the
    /// constraint *must* be selected by a valid solution.
    pub fn add_constraint<OI>(&mut self, options: OI)
    where
        OI: IntoIterator<Item = M::InOptID>,
    {
        let constraint = self.internal_add_constraint(options);

        // Link this constraint into the list of mandatory constraints
        // (whose head is at node 0).
        let left = self.left(0);
        self.set_left(constraint, left);
        self.set_right(constraint, 0);
        self.relink_into_option(constraint);
    }

    /// Adds an optional constraint to the solution search space.
    ///
    /// A constraint is a set of options of the overall problem
    /// universe; each option is identified by a [`usize`].  Each
    /// member of the set is incompatible with the others within the
    /// constraint.
    ///
    /// For an optional constraint, only one option covering the
    /// constraint may be selected by a valid solution; it's also
    /// valid for the constraint to not be covered all.
    pub fn add_optional_constraint<OI>(&mut self, options: OI)
    where
        OI: IntoIterator<Item = M::InOptID>,
    {
        self.internal_add_constraint(options);
    }

    fn internal_add_constraint<OI>(&mut self, options: OI) -> usize
    where
        OI: IntoIterator<Item = M::InOptID>,
    {
        // Allocate a constraint state node for this constraint.
        let constraint = self.nodes.len();
        self.nodes.push(Node {
            constraint_or_node_count: 0,
            option: 0,
            up: constraint,
            down: constraint,
            left: constraint,
            right: constraint,
        });

        // For each option, add a node to the constraint.
        let mut node_count: usize = 0;
        for option_id in options {
            node_count += 1;
            let option = self.mapper.to_dense_id(option_id);

            // Allocate a node for this option
            let node = constraint + node_count;
            self.nodes.push(Node {
                constraint_or_node_count: constraint,
                option,
                up: node - 1,
                down: node + 1,
                left: node,
                right: node,
            });

            // Link the node into the list of nodes with the same option.
            if self.options.len() <= option {
                self.options.extend(iter::repeat_n(
                    Option::None,
                    option - self.options.len() + 1,
                ));
            }
            match self.options[option] {
                None => self.options[option] = Option::Some(node),
                Some(found_node) => {
                    // Link the current node into the option, to the
                    // left of the found node (=> end of the node list
                    // for this option).
                    let left = self.left(found_node);
                    self.set_left(node, left);
                    self.set_right(node, found_node);
                    self.relink_into_option(node);
                }
            }
        }

        self.set_node_count(constraint, node_count);
        if node_count != 0 {
            self.set_down(constraint, constraint + 1);
        }
        self.set_up(constraint, constraint + node_count);
        self.set_down(constraint + node_count, constraint);

        constraint
    }
}

// Algorithm implementation utility methods

/// Finds the constraint with the minimium node count, if any active
/// constraint exists.
fn find_min_node_constraint<M>(problem: &mut Problem<M>) -> Option<usize>
where
    M: OptMapper,
{
    // Start from node zero, which is always the head node. This
    // links to the nodes representing the active constraints.
    let mut constraint = problem.right(0);

    if constraint == 0 {
        // There are no active constraints => mandatory constraints to
        // be covered.
        return None;
    }

    // Scan through the active constraints to find the constraint with the
    // lowest number of nodes.
    let mut min_node_count = problem.node_count(constraint);
    let mut min_node_count_constraint = constraint;
    loop {
        constraint = problem.right(constraint);
        if constraint == 0 {
            break;
        }
        let constraint_node_count = problem.node_count(constraint);
        if constraint_node_count < min_node_count {
            min_node_count = constraint_node_count;
            min_node_count_constraint = constraint;
        }
    }

    Some(min_node_count_constraint)
}

/// Unlinks a covered constraint (one which is covered by the
/// current solution) from the problem matrix (making it inactive
/// & removing the need to explore it).  This also removes every
/// option that would also cover the constraint, as those options are
/// by definition incompatible with the current solution (since
/// the constraint's already been covered).
fn unlink_constraint_and_conflicting_options<M>(problem: &mut Problem<M>, covered_constraint: usize)
where
    M: OptMapper,
{
    // Note that although the rest of the problem matrix is no
    // longer linked to the removed constraint and options that cover
    // that constraint, the information needed to re-link the constraint
    // and its options remains in the link fields of the constraint
    // itself.
    problem.unlink_from_option(covered_constraint);
    let mut covered_node = covered_constraint;
    loop {
        covered_node = problem.down(covered_node);
        if covered_node == covered_constraint {
            break;
        }
        let mut conflicting_option_node = covered_node;
        loop {
            conflicting_option_node = problem.right(conflicting_option_node);
            if conflicting_option_node == covered_node {
                break;
            }
            problem.unlink_from_constraint(conflicting_option_node);
            let decr_constraint = problem.constraint(conflicting_option_node);
            problem.decr_node_count(decr_constraint);
        }
    }
}

/// Performs the inverse operation of
/// unlink_constraint_and_conflicting_options, re-linking the constraint
/// and the options that cover the constraint back into the problem
/// matrix.
fn relink_constraint_and_conflicting_options<M>(problem: &mut Problem<M>, covered_constraint: usize)
where
    M: OptMapper,
{
    // This code very carefully executes in the exact reverse
    // sequence of the unlink operation, because it's using the
    // information that's in the removed nodes' links in order to
    // perform the re-linking.
    let mut covered_node = covered_constraint;
    loop {
        covered_node = problem.up(covered_node);
        if covered_node == covered_constraint {
            break;
        }
        let mut conflicting_option_node = covered_node;
        loop {
            conflicting_option_node = problem.left(conflicting_option_node);
            if conflicting_option_node == covered_node {
                break;
            }
            let incr_constraint = problem.constraint(conflicting_option_node);
            problem.incr_node_count(incr_constraint);
            problem.relink_into_constraint(conflicting_option_node);
        }
    }
    problem.relink_into_option(covered_constraint);
}

/// Unlinks all constraints covered by a particular option (and all
/// options that cover *those* constraints) from the problem matrix.
fn unlink_option_constraints_and_conflicting_options<M>(problem: &mut Problem<M>, node: usize)
where
    M: OptMapper,
{
    // The consequence of adding this option to our proposed
    // solution is that all other constraints linked into the
    // option are now covered, and all options linked into
    // *those* constraints must be removed, since they conflict
    // with the proposed solution.
    let mut option_node = node;
    loop {
        option_node = problem.right(option_node);
        if option_node == node {
            // We've looped around the current option,
            // unlinking it, the constraint it covers, and the
            // options that cover those constraints.
            break;
        }
        let covered_constraint = problem.constraint(option_node);
        unlink_constraint_and_conflicting_options(problem, covered_constraint);
    }
}

/// Relinks all constraints covered by a particular option (and all
/// options that cover *those* constraints) back into the problem
/// matrix.
fn relink_option_constraints_and_conflicting_options<M>(problem: &mut Problem<M>, node: usize)
where
    M: OptMapper,
{
    // This logic very carefully re-links the solution graph in
    // exactly the reverse order of the unlinking done by
    // unlink_option_constraints_and_conflicts().

    let mut option_node = node;
    loop {
        option_node = problem.left(option_node);
        if option_node == node {
            break;
        }
        let covered_constraint = problem.constraint(option_node);
        relink_constraint_and_conflicting_options(problem, covered_constraint);
    }
}

impl<M> Iterator for SolutionIterator<M>
where
    M: OptMapper,
{
    // The iterator item type.
    //
    // NB: We use a vec here because we need to capture the solution
    // -- there doesn't appear to be a good safe way to say "Hey, I'm
    // giving you back a reference to some of my internal state, don't
    // call me again until you're done with it, 'cause I'm gonna
    // mutate it out from under you."
    type Item = Vec<M::OutOptID>;

    fn next(&mut self) -> Option<Self::Item> {
        // At this point in the code, we're generating the next
        // solution -- either we're just starting off the iteration,
        // or we've already generated at least one solution.
        //
        // If we're just starting the iteration:
        // * The solution vector is empty.
        // * The location vector contains one location, which
        //   references the min-option constraint.
        // * The current constraint is unlinked from the problem.
        //
        // If we've already generated at least one solution:
        // * The solution vector contains that solution.
        // * The location vector indicates where we're at in the
        //   exploration; the last entry is the constraint+option that
        //   generated the solution.
        // * The current constraint is unlinked from the problem.
        // * The current option is unlinked from the problem.

        'walk_location_stack: loop {
            // This is the loop over the location stack --
            // essentially, we pop the location stack and continue
            // from there.  So this is doing one step of unwinding per
            // iteration.

            let Some(mut loc) = self.location.pop() else {
                // If we're here, and there're no locations in the
                // location stack, we're done with the problem.
                return None;
            };

            loop {
                // This loop repeatedly checks the current node, and
                // moves to the next constraint.

                if loc.node != loc.constraint {
                    // We need to put this node back into the problem.
                    // This also means there's a solution that needs
                    // to be popped.
                    relink_option_constraints_and_conflicting_options(&mut self.problem, loc.node);
                    self.solution.pop();
                }

                loc.node = self.problem.down(loc.node);
                if loc.node == loc.constraint {
                    // We've looped back around to the constraint header,
                    // which means we've tried every node in this constraint
                    // and reported the resulting solutions (if any).  So
                    // there are no further solutions that can be obtained
                    // via this constraint in the problem's current state.
                    relink_constraint_and_conflicting_options(&mut self.problem, loc.constraint);
                    continue 'walk_location_stack;
                }

                // Otherwise, we're exploring this node, whose option
                // becomes part of our proposed solution.
                self.solution.push(self.problem.option_id(loc.node));

                // Unlink the node's option, the associated constraints, and
                // conflicting options.
                unlink_option_constraints_and_conflicting_options(&mut self.problem, loc.node);

                // Put this location onto the stack, re-establishing
                // our invariants.  At this point, we could produce a
                // value to the iterator, returning to our caller.
                self.location.push(loc);

                // Select the next active constraint.
                let Some(constraint) = find_min_node_constraint(&mut self.problem) else {
                    // If there are no active constraints, then we've found a solution.
                    return Some(
                        self.solution
                            .to_vec()
                            .into_iter()
                            .map(|dense_id: usize| -> M::OutOptID {
                                self.problem.mapper.to_option_id(dense_id)
                            })
                            .collect(),
                    );
                };

                if self.problem.node_count(constraint) == 0 {
                    // Some constraint had a zero node count -- so the
                    // current state is not compatible with finding a
                    // solution.  So we need to relink the current
                    // node, and continue on to the next -- i.e. we
                    // need to continue the walk_constraint loop.
                    continue 'walk_location_stack;
                }

                // Otherwise, we're ready to process this constraint.
                unlink_constraint_and_conflicting_options(&mut self.problem, constraint);
                loc = Location {
                    constraint,
                    node: constraint,
                };
            }
        }
    }
}

/// Computes solutions to the exact cover problem described by the
/// supplied [`Problem`].
///
/// The problem constraints should have already been established via
/// the [`Problem::add_constraint()`] and
/// [`Problem::add_optional_constraint()`] methods.
pub fn solve<M>(problem: Problem<M>) -> SolutionIterator<M>
where
    M: OptMapper,
{
    let mut it = SolutionIterator {
        problem,
        location: Vec::new(),
        solution: Vec::new(),
    };
    if let Some(constraint) = find_min_node_constraint(&mut it.problem) {
        // Establish initial invariants.
        unlink_constraint_and_conflicting_options(&mut it.problem, constraint);
        it.location.push(Location {
            constraint,
            node: constraint,
        });
    }
    it
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() {
        // This is the example problem from
        // <https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X>; it
        // makes for a nice test case, since the article explains
        // every step of the execution.

        let mut problem = Problem::new(DenseOptMapper {});

        problem.add_constraint([0, 1]);
        problem.add_constraint([4, 5]);
        problem.add_constraint([3, 4]);
        problem.add_constraint([0, 1, 2]);
        problem.add_constraint([2, 3]);
        problem.add_constraint([2, 3]);
        problem.add_constraint([3, 4]);
        problem.add_constraint([0, 2, 4, 5]);

        // And then, actually solve the problem.
        let mut count: usize = 0;
        for soln in solve(problem) {
            count += 1;
            assert_eq!(
                vec![1usize, 3, 5],
                soln,
                "solution: expected(left) != actual(right)"
            );
        }

        assert_eq!(1, count, "solution count: expected(left) != actual(right)");
    }

    #[test]
    fn eight_queens() {
        const N: usize = 8;

        #[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
        struct Square {
            file: usize,
            rank: usize,
        }

        let mut problem = Problem::new(HashOptMapper::new());

        // For N queens: we have N^2 squares on the board.  Each of
        // these squares is a possible place to put a queen; this
        // prevents the placement of a queen on all squares in the
        // same rank, all squares in the same file, and all squares
        // sharing the square's diagonals.
        //
        // In terms of a DLX problem: each square represents one
        // member of the set of values being covered by the problem,
        // and also identifies a option that may be part of the
        // overall solution, where the option associated with a square
        // is exactly those squares that are attacked by a queen in
        // that square.
        //
        // Each rank and file maps to a mandatory constraint -- there
        // must be a queen in each rank and file.  The diagonals map
        // to optional constraints: it is acceptable for a solution to
        // contain diagonals that are not covered by a queen, as long
        // as there is at most one queen per diagonal.

        // Add the per-rank constraints.
        for rank in 0..N {
            problem.add_constraint((0..N).map(|file| Square { file, rank }))
        }

        // Add the per-file constraints.
        for file in 0..N {
            problem.add_constraint((0..N).map(|rank| Square { file, rank }))
        }

        // Add the per-diagonal constraints along the files (so,
        // imagine starting at the bottom of the board, and scanning
        // up and to the right).  Each diagonal starts at rank=0, and
        // is numbered by the file it starts on.
        //
        // Note that file N-1 contains only a single element (at
        // rank=0), and so does not need to have an optional
        // constraint attached to it.
        for file_diagonal in 0..(N - 1) {
            problem.add_optional_constraint((0..N - file_diagonal).map(|rank| Square {
                file: file_diagonal + rank,
                rank,
            }))
        }

        // Add the per-reverse-diagonal constraints along the files
        // (so, imagine starting at the bottom of the board, and
        // scanning up and to the left).  Each reverse diagonal starts
        // at rank=0, and is numbered by the file it starts on.
        //
        // Note that file 0 contains only a single element (at
        // rank=0), and so does not need to have an optional
        // constraint attached to it.
        for file_reverse_diagonal in 1..N {
            problem.add_optional_constraint((0..file_reverse_diagonal + 1).map(|rank| Square {
                file: file_reverse_diagonal - rank,
                rank,
            }))
        }

        // Add the per-diagonal constraints along the ranks (so,
        // imagine starting at the left side of the board, and
        // scanning up and to the right).  Each diagonal starts at
        // file=0, and is numbered by the rank it starts on.
        //
        // Note that rank N-1 contains only a single element (at
        // file=0), and so does not need to have an optional
        // constraint attached to it.  Additionally, rank 0 is already
        // covered by the per-file diagonals.
        for rank_diagonal in 1..(N - 1) {
            problem.add_optional_constraint((0..N - rank_diagonal).map(|file| Square {
                file,
                rank: rank_diagonal + file,
            }))
        }

        // Add the per-reverse-diagonal constraints along the ranks
        // (so, imagine starting at the right side of the board, and
        // scanning up and to the left).  Each reverse diagonal starts
        // at file=N-1, and is numbered by the rank it starts on.
        //
        // Note that rank N-1 contains only a single element (at
        // file=N-1), and so does not need to have an optional
        // constraint attached to it.  Additionally, rank 0 is already
        // covered by the per-file reverse diagonals.
        for rank_reverse_diagonal in 1..(N - 1) {
            problem.add_optional_constraint((0..N - rank_reverse_diagonal).map(|file| Square {
                file: N - 1 - file,
                rank: rank_reverse_diagonal + file,
            }))
        }

        // And then, actually solve the problem, printing the
        // solutions as we go.
        let mut count: usize = 0;
        for soln in solve(problem) {
            count += 1;
            let mut board = [[false; N]; N];
            print!("Found solution: [");
            for square in soln {
                let file = square.file;
                let rank = square.rank;
                let file_char = (b'a' + (square.file as u8)) as char;
                board[square.rank][file] = true;
                print!(" {file_char}{rank}");
            }
            println!("]");
            if N == 8 {
                println!("");
                println!("                   A B C D E F G H");
                println!("                 +-----------------+");
                for rank in (0..N).rev() {
                    print!("                {rank}|");
                    for file in 0..N {
                        if board[rank][file] {
                            print!(" Q");
                        } else {
                            print!(" _");
                        }
                    }
                    println!(" |{rank}");
                }
                println!("                 +-----------------+");
                println!("                   A B C D E F G H");
                println!("");
                println!("");
            }
        }

        if N == 8 {
            assert_eq!(92, count, "solution count: expected(left) != actual(right)");
        }
    }

    #[test]
    fn sudoku() {
        // For Sudoku, we have a 9x9 grid of numbers 1-9, with four
        // types of rules:
        // * All the squares in a given row must have different numbers.
        // * All the squares in a given column must have different numbers.
        // * All the squares in a given 3x3 subgrid must have different numbers.
        // * Any given square can only have one number.
        //
        // In terms of a DLX problem, each 1..=9 possibility for each
        // square represents one option.  Selecting one option is
        // equivalent to determining the contents of that square, and
        // thus covering that possibility in all other squares in the
        // same row/column/subgrid, as well as covering that all other
        // possibilities for the current square.

        #[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
        struct GridVal {
            row: usize, // 1..9
            col: usize, // 1..9
            val: usize, // 1..9
        }

        let mut problem = Problem::new(HashOptMapper::new());

        // Add the row constraints.
        for row in 1..=9 {
            for val in 1..=9 {
                problem.add_constraint(
                    (1..=9)
                        .map(|col| GridVal { row, col, val })
                );
            }
        }

        // Add the column constraints.
        for col in 1..=9 {
            for val in 1..=9 {
                problem.add_constraint(
                    (1..=9)
                        .map(|row| GridVal { row, col, val })
                );
            }
        }

        // Add the subgrid constraints.
        for supergrid_row in 0..=2 {
            for supergrid_col in 0..=2 {
                for val in 1..=9 {
                    problem.add_constraint(
                        (0..=8)
                            .map(|rc| GridVal {
                                row: supergrid_row * 3 + rc / 3 + 1,
                                col: supergrid_col * 3 + rc % 3 + 1,
                                val,
                            })
                    );
                }
            }
        }

        // Add the per-square constraints.
        for row in 1..=9 {
            for col in 1..=9 {
                problem.add_constraint(
                    (1..=9)
                        .map(|val| GridVal { row, col, val })
                );
            }
        }

        // Next, we add the puzzle-specific constraints.
        // This set was the NYT Hard sudoku puzzle on 2025-04-04,
        // numbering rows from the bottom up and columns from left to
        // right.
        [
            GridVal {
                row: 1,
                col: 6,
                val: 7,
            },
            GridVal {
                row: 1,
                col: 8,
                val: 8,
            },
            GridVal {
                row: 2,
                col: 5,
                val: 2,
            },
            GridVal {
                row: 3,
                col: 1,
                val: 2,
            },
            GridVal {
                row: 3,
                col: 2,
                val: 4,
            },
            GridVal {
                row: 3,
                col: 4,
                val: 6,
            },
            GridVal {
                row: 4,
                col: 3,
                val: 7,
            },
            GridVal {
                row: 4,
                col: 4,
                val: 2,
            },
            GridVal {
                row: 4,
                col: 6,
                val: 1,
            },
            GridVal {
                row: 4,
                col: 7,
                val: 6,
            },
            GridVal {
                row: 5,
                col: 2,
                val: 1,
            },
            GridVal {
                row: 5,
                col: 2,
                val: 1,
            },
            GridVal {
                row: 5,
                col: 5,
                val: 6,
            },
            GridVal {
                row: 5,
                col: 6,
                val: 3,
            },
            GridVal {
                row: 5,
                col: 8,
                val: 7,
            },
            GridVal {
                row: 6,
                col: 2,
                val: 5,
            },
            GridVal {
                row: 6,
                col: 7,
                val: 9,
            },
            GridVal {
                row: 7,
                col: 3,
                val: 9,
            },
            GridVal {
                row: 7,
                col: 6,
                val: 5,
            },
            GridVal {
                row: 8,
                col: 4,
                val: 4,
            },
            GridVal {
                row: 8,
                col: 5,
                val: 7,
            },
            GridVal {
                row: 8,
                col: 9,
                val: 9,
            },
            GridVal {
                row: 9,
                col: 1,
                val: 8,
            },
            GridVal {
                row: 9,
                col: 8,
                val: 4,
            },
        ]
        .map(|id| problem.add_constraint([id]));

        let mut count: usize = 0;
        for _ in solve(problem) {
            count += 1
        }

        assert_eq!(1, count, "solution count: expected(left) != actual(right)");
    }
}
