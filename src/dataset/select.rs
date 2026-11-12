//! Choosing rows, for [`Dataset::select_rows`](super::Dataset::select_rows):
//! a contiguous range is borrowed, any other choice of rows copied once.

use std::ops::Range;

use ndarray::{ArrayView, Axis, RemoveAxis, Slice};

use super::CovariantCowArray;

/// The rows a selection asks for, with what their shape allows.
pub(super) struct Selection<'r> {
    pub(super) rows: &'r [usize],
    /// `Some` when the rows are consecutive and ascending: a range that every
    /// part except a CSC block can borrow instead of copying.
    pub(super) range: Option<Range<usize>>,
    /// Ascending with no repeats, so the rows kept from a CSC column are
    /// already in order.
    pub(super) increasing: bool,
}

impl<'r> Selection<'r> {
    pub(super) fn new(rows: &'r [usize]) -> Self {
        let consecutive = rows.windows(2).all(|pair| pair[1] == pair[0] + 1);
        let range = consecutive.then(|| {
            let start = rows.first().copied().unwrap_or(0);
            start..start + rows.len()
        });
        Selection {
            rows,
            range,
            increasing: rows.windows(2).all(|pair| pair[0] < pair[1]),
        }
    }
}

/// The chosen rows of `x` along its first axis: a borrowed slice for a range,
/// otherwise a copy.
pub(super) fn select_array<'b, A: Clone, D: RemoveAxis>(
    x: ArrayView<'b, A, D>,
    rows: &[usize],
    range: Option<Range<usize>>,
) -> CovariantCowArray<'b, A, D> {
    match range {
        Some(range) => x.slice_axis_move(Axis(0), Slice::from(range)).into(),
        None => x.select(Axis(0), rows).into(),
    }
}
