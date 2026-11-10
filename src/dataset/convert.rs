//! Writing a dataset's blocks out as one block of a chosen kind: what the
//! `into_*` conversions do. The simplest correct versions: each allocates the
//! result once and fills it in one walk over the blocks (two for sparse).

use ndarray::{Array2, CowArray, Ix2, ShapeBuilder, s};

use super::block::Block;
use super::{Absent, CscMatrix, CsrMatrix};
use crate::Float;

/// The blocks as one dense matrix with each row contiguous, or each column if
/// `col_major`. A single block already of that kind is moved, not copied.
pub(super) fn into_dense<'a, F: Float>(
    mut blocks: Vec<Block<'a, F>>,
    n_rows: usize,
    col_major: bool,
) -> CowArray<'a, F, Ix2> {
    if blocks.len() == 1 {
        match blocks.pop() {
            Some(Block::DenseRowMajor(x)) if !col_major => return x,
            Some(Block::DenseColMajor(x)) if col_major => return x,
            Some(other) => blocks.push(other),
            None => {}
        }
    }
    let n_cols = blocks.iter().map(Block::n_cols).sum();
    let mut dense = Array2::zeros((n_rows, n_cols).set_f(col_major));
    let mut start = 0;
    for block in &blocks {
        let end = start + block.n_cols();
        block.write_dense(dense.slice_mut(s![.., start..end]));
        start = end;
    }
    dense.into()
}

/// The blocks as one CSR matrix whose absent entries mean `absent`. A single
/// CSR block with that meaning is moved, not copied.
pub(super) fn into_csr<'a, F: Float>(
    mut blocks: Vec<Block<'a, F>>,
    n_rows: usize,
    absent: Absent,
) -> CsrMatrix<'a, F> {
    if blocks.len() == 1 {
        match blocks.pop() {
            Some(Block::Csr(m)) if m.absent() == absent => return m,
            Some(other) => blocks.push(other),
            None => {}
        }
    }
    let n_cols = blocks.iter().map(Block::n_cols).sum();
    let (offsets, indices, values) = compress(&blocks, n_rows, absent, |row, col| (row, col));
    CsrMatrix::new(n_cols, offsets, indices, values)
        .expect("built row by row with increasing columns")
        .with_absent(absent)
}

/// The blocks as one CSC matrix whose absent entries mean `absent`. A single
/// CSC block with that meaning is moved, not copied.
pub(super) fn into_csc<'a, F: Float>(
    mut blocks: Vec<Block<'a, F>>,
    n_rows: usize,
    absent: Absent,
) -> CscMatrix<'a, F> {
    if blocks.len() == 1 {
        match blocks.pop() {
            Some(Block::Csc(m)) if m.absent() == absent => return m,
            Some(other) => blocks.push(other),
            None => {}
        }
    }
    let n_cols = blocks.iter().map(Block::n_cols).sum();
    let (offsets, indices, values) = compress(&blocks, n_cols, absent, |row, col| (col, row));
    CscMatrix::new(n_rows, offsets, indices, values)
        .expect("built column by column with increasing rows")
        .with_absent(absent)
}

/// The compressed arrays of the blocks side by side, by a counting sort over
/// lanes: `lane_of(row, col)` gives an entry's `(lane, index)`, rows for CSR
/// and columns for CSC. One walk counts each lane's entries, a running sum
/// turns the counts into offsets, and a second walk writes each entry at its
/// lane's next free position. Blocks are walked left to right, each in its
/// own order, so indices arrive in increasing order within every lane and need
/// no sorting.
fn compress<F: Float>(
    blocks: &[Block<'_, F>],
    n_lanes: usize,
    absent: Absent,
    lane_of: impl Fn(usize, usize) -> (usize, usize),
) -> (Vec<usize>, Vec<usize>, Vec<F>) {
    let walk = |visit: &mut dyn FnMut(usize, usize, F)| {
        let mut col_offset = 0;
        for block in blocks {
            block.for_each_entry(absent, |row, col, value| {
                let (lane, index) = lane_of(row, col_offset + col);
                visit(lane, index, value);
            });
            col_offset += block.n_cols();
        }
    };

    let mut offsets = vec![0; n_lanes + 1];
    walk(&mut |lane, _, _| offsets[lane + 1] += 1);
    for lane in 0..n_lanes {
        offsets[lane + 1] += offsets[lane];
    }

    let n_stored = offsets[n_lanes];
    let mut indices = vec![0; n_stored];
    let mut values = vec![F::zero(); n_stored];
    let mut next = offsets[..n_lanes].to_vec();
    walk(&mut |lane, index, value| {
        indices[next[lane]] = index;
        values[next[lane]] = value;
        next[lane] += 1;
    });
    (offsets, indices, values)
}
