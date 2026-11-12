//! The storage CSR and CSC matrices share: the stored entries, one lane after
//! another. A lane is a row for CSR and a column for CSC; within a lane, each
//! entry's position along the other axis is its index.

use std::borrow::Cow;

use ndarray::Array2;

use super::select::Selection;
use crate::error::check_len;
use crate::{Float, PredictError, Result};

/// What an entry a sparse matrix doesn't store means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Absent {
    /// The entry is 0, as in scikit-learn and in the maths: the sparse matrix is
    /// the dense matrix with its zeros left out.
    #[default]
    Zero,
    /// The entry is missing, like a NaN, as in XGBoost. The matrix can then only
    /// be validated as `Finite` if it stores every entry.
    Missing,
}

impl Absent {
    /// The value an entry that isn't stored stands for.
    pub(super) fn value<F: Float>(self) -> F {
        match self {
            Absent::Zero => F::zero(),
            Absent::Missing => F::nan(),
        }
    }

    /// Whether `value` is what an entry that isn't stored stands for, so a
    /// sparse matrix with this meaning can leave it out. `-0.0` counts as 0.
    pub(super) fn stands_for<F: Float>(self, value: F) -> bool {
        match self {
            Absent::Zero => value == F::zero(),
            Absent::Missing => value.is_nan(),
        }
    }
}

/// How a compressed matrix's parts are named in errors: CSR has row offsets and
/// column indices, CSC column offsets and row indices.
pub(super) struct Names {
    pub(super) offsets: &'static str,
    pub(super) indices: &'static str,
    pub(super) lane: &'static str,
    pub(super) index: &'static str,
}

/// Lanes of stored entries: lane `i`'s entries are positions
/// `offsets[i]..offsets[i + 1]` of `indices` and `values`.
#[derive(Debug, Clone)]
pub(super) struct Compressed<'a, F: Float> {
    // The length of a lane, which the indices can't show: positions with no
    // stored entry leave no trace.
    lane_len: usize,
    offsets: Cow<'a, [usize]>,
    indices: Cow<'a, [usize]>,
    values: Cow<'a, [F]>,
    pub(super) absent: Absent,
}

impl<'a, F: Float> Compressed<'a, F> {
    /// The parts, their structure checked, in this order: `offsets` (non-empty,
    /// start at 0, never decrease), one index and one value per stored entry,
    /// then each lane's indices (below `lane_len`, strictly increasing).
    pub(super) fn new(
        names: &Names,
        lane_len: usize,
        offsets: Cow<'a, [usize]>,
        indices: Cow<'a, [usize]>,
        values: Cow<'a, [F]>,
    ) -> Result<Self> {
        let matrix = Compressed {
            lane_len,
            offsets,
            indices,
            values,
            absent: Absent::Zero,
        };
        matrix.check_structure(names)?;
        Ok(matrix)
    }

    /// Parts already known to be well formed: those of a matrix that passed
    /// [`new`](Self::new), selected or rebuilt by this module.
    fn from_checked(
        lane_len: usize,
        offsets: Cow<'a, [usize]>,
        indices: Cow<'a, [usize]>,
        values: Cow<'a, [F]>,
        absent: Absent,
    ) -> Self {
        Compressed {
            lane_len,
            offsets,
            indices,
            values,
            absent,
        }
    }

    /// The same matrix, borrowing its three arrays.
    pub(super) fn view(&self) -> Compressed<'_, F> {
        Compressed {
            lane_len: self.lane_len,
            offsets: Cow::Borrowed(&self.offsets),
            indices: Cow::Borrowed(&self.indices),
            values: Cow::Borrowed(&self.values),
            absent: self.absent,
        }
    }

    pub(super) fn n_lanes(&self) -> usize {
        self.offsets.len() - 1
    }

    pub(super) fn lane_len(&self) -> usize {
        self.lane_len
    }

    pub(super) fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    pub(super) fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub(super) fn values(&self) -> &[F] {
        &self.values
    }

    fn check_structure(&self, names: &Names) -> Result<()> {
        let offsets = &self.offsets[..];
        let Some(&first) = offsets.first() else {
            return malformed(
                names.offsets,
                format!("must have one entry per {} plus one", names.lane),
            );
        };
        if first != 0 {
            return malformed(names.offsets, format!("must start at 0, got {first}"));
        }
        for (lane, pair) in offsets.windows(2).enumerate() {
            if pair[1] < pair[0] {
                return malformed(
                    names.offsets,
                    format!(
                        "must not decrease, got {} after {} at {} {lane}",
                        pair[1], pair[0], names.lane
                    ),
                );
            }
        }

        let n_stored = offsets[offsets.len() - 1];
        check_len(names.indices, n_stored, self.indices.len())?;
        check_len("values", n_stored, self.values.len())?;

        for lane in 0..self.n_lanes() {
            let (indices, _) = self.lane(lane);
            for (k, &index) in indices.iter().enumerate() {
                if index >= self.lane_len {
                    return malformed(
                        names.indices,
                        format!(
                            "{} {lane}: {} {index} is out of range for {} {}s",
                            names.lane, names.index, self.lane_len, names.index
                        ),
                    );
                }
                if k > 0 && index <= indices[k - 1] {
                    return malformed(
                        names.indices,
                        format!(
                            "{} {lane}: {} {index} does not come after {} {}",
                            names.lane,
                            names.index,
                            names.index,
                            indices[k - 1]
                        ),
                    );
                }
            }
        }
        Ok(())
    }

    /// The chosen lanes, in the order given (CSR's rows).
    ///
    /// A range of lanes is one run of the indices and values, so those are
    /// borrowed and only the offsets, one per lane, are rebuilt. Any other
    /// choice copies each lane's entries, repeats included.
    pub(super) fn select_lanes(&self, selection: &Selection<'_>) -> Compressed<'_, F> {
        if let Some(range) = &selection.range {
            let (start, end) = (self.offsets[range.start], self.offsets[range.end]);
            let offsets = self.offsets[range.start..=range.end]
                .iter()
                .map(|offset| offset - start)
                .collect::<Vec<_>>();
            return Compressed::from_checked(
                self.lane_len,
                Cow::Owned(offsets),
                Cow::Borrowed(&self.indices[start..end]),
                Cow::Borrowed(&self.values[start..end]),
                self.absent,
            );
        }
        let mut offsets = Vec::with_capacity(selection.rows.len() + 1);
        offsets.push(0);
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        for &lane in selection.rows {
            let (lane_indices, lane_values) = self.lane(lane);
            indices.extend_from_slice(lane_indices);
            values.extend_from_slice(lane_values);
            offsets.push(indices.len());
        }
        Compressed::from_checked(
            self.lane_len,
            Cow::Owned(offsets),
            Cow::Owned(indices),
            Cow::Owned(values),
            self.absent,
        )
    }

    /// The chosen positions within every lane, renumbered in the order given
    /// (CSC's rows). Always a copy: each lane keeps a different run of its
    /// entries, and their indices change.
    ///
    /// A position chosen several times gives its entry once for each time, and
    /// each lane's entries are sorted by their new index unless the positions
    /// were ascending without repeats, which keeps them in order already.
    pub(super) fn select_indices(&self, selection: &Selection<'_>) -> Compressed<'static, F> {
        // Where each old position went: a counting sort of the new positions by
        // old position, so old position `p`'s new ones are
        // `new_positions[starts[p]..starts[p + 1]]`.
        let mut starts = vec![0; self.lane_len + 1];
        for &old in selection.rows {
            starts[old + 1] += 1;
        }
        for p in 0..self.lane_len {
            starts[p + 1] += starts[p];
        }
        let mut next = starts.clone();
        let mut new_positions = vec![0; selection.rows.len()];
        for (new, &old) in selection.rows.iter().enumerate() {
            new_positions[next[old]] = new;
            next[old] += 1;
        }

        let mut offsets = Vec::with_capacity(self.n_lanes() + 1);
        offsets.push(0);
        let mut entries: Vec<(usize, F)> = Vec::new();
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        for lane in 0..self.n_lanes() {
            entries.clear();
            let (lane_indices, lane_values) = self.lane(lane);
            for (&old, &value) in lane_indices.iter().zip(lane_values) {
                for &new in &new_positions[starts[old]..starts[old + 1]] {
                    entries.push((new, value));
                }
            }
            if !selection.increasing {
                entries.sort_unstable_by_key(|&(new, _)| new);
            }
            indices.extend(entries.iter().map(|&(new, _)| new));
            values.extend(entries.iter().map(|&(_, value)| value));
            offsets.push(indices.len());
        }
        Compressed::from_checked(
            selection.rows.len(),
            Cow::Owned(offsets),
            Cow::Owned(indices),
            Cow::Owned(values),
            self.absent,
        )
    }

    /// Every stored entry as `(lane, index, value)`, lane by lane and in index
    /// order within a lane.
    pub(super) fn for_each_stored(&self, mut visit: impl FnMut(usize, usize, F)) {
        for lane in 0..self.n_lanes() {
            let (indices, values) = self.lane(lane);
            for (&index, &value) in indices.iter().zip(values) {
                visit(lane, index, value);
            }
        }
    }

    /// Lane `lane`'s stored indices and values.
    fn lane(&self, lane: usize) -> (&[usize], &[F]) {
        let range = self.offsets[lane]..self.offsets[lane + 1];
        (&self.indices[range.clone()], &self.values[range])
    }

    /// `(lane, index, value)` of the first entry, lane by lane and in index
    /// order within a lane, that `is_rejected` picks out. With
    /// [`Absent::Missing`], entries that aren't stored are checked as NaN.
    pub(super) fn first_rejected(
        &self,
        is_rejected: impl Fn(F) -> bool,
    ) -> Option<(usize, usize, F)> {
        let check_absent = self.absent == Absent::Missing && is_rejected(F::nan());
        (0..self.n_lanes()).find_map(|lane| {
            let (indices, values) = self.lane(lane);
            let stored = indices
                .iter()
                .zip(values)
                .find(|&(_, &value)| is_rejected(value))
                .map(|(&index, &value)| (index, value));
            let absent = check_absent
                .then(|| first_absent(indices, self.lane_len))
                .flatten()
                .map(|index| (index, F::nan()));
            stored
                .into_iter()
                .chain(absent)
                .min_by_key(|&(index, _)| index)
                .map(|(index, value)| (lane, index, value))
        })
    }

    /// Every entry written out, absent ones as 0 or NaN, one lane per row of
    /// the result: CSR's shape, and the transpose of CSC's.
    pub(super) fn to_dense_by_lanes(&self) -> Array2<F> {
        let mut dense = Array2::from_elem((self.n_lanes(), self.lane_len), self.absent.value());
        self.for_each_stored(|lane, index, value| dense[[lane, index]] = value);
        dense
    }
}

/// The first index a lane doesn't store. The stored indices are strictly
/// increasing, so it is the first position `k` whose index isn't `k`.
fn first_absent(indices: &[usize], lane_len: usize) -> Option<usize> {
    match indices.iter().enumerate().find(|&(k, &index)| index != k) {
        Some((k, _)) => Some(k),
        None => (indices.len() < lane_len).then_some(indices.len()),
    }
}

fn malformed(context: &'static str, reason: String) -> Result<()> {
    Err(PredictError::Malformed { context, reason })
}
