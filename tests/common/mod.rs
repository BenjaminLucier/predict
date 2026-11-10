//! Helpers shared by the integration tests. Each test file compiles this module
//! on its own and uses only some of it.
#![allow(dead_code)]

use ndarray::{Array2, ArrayView2, s};
use predict::dataset::{Absent, BlockView, Blocks, Density, Orientation, State, TargetState};
use predict::{Dataset, Float};

/// Every value of `data` as one row-major matrix, written out from its blocks:
/// entries a sparse block doesn't store as 0, or NaN when they mean missing.
/// Test-only, so the copy doesn't matter; the library has no such read.
pub fn dense<'a, F: Float, O: Orientation, D: Density, S: State, Y: TargetState>(
    data: &Dataset<'a, F, Blocks<'a, F, O, D>, S, Y>,
) -> Array2<F> {
    let mut out = Array2::zeros((data.n_samples(), data.n_features()));
    let mut start = 0;
    for block in data.blocks() {
        match block {
            BlockView::DenseRowMajor(x) | BlockView::DenseColMajor(x) => {
                out.slice_mut(s![.., start..start + x.ncols()]).assign(&x);
                start += x.ncols();
            }
            BlockView::Csr(m) => {
                let mut part = out.slice_mut(s![.., start..start + m.n_cols()]);
                part.fill(absent_value(m.absent()));
                let offsets = m.row_offsets();
                for row in 0..m.n_rows() {
                    for k in offsets[row]..offsets[row + 1] {
                        part[[row, m.col_indices()[k]]] = m.values()[k];
                    }
                }
                start += m.n_cols();
            }
            BlockView::Csc(m) => {
                let mut part = out.slice_mut(s![.., start..start + m.n_cols()]);
                part.fill(absent_value(m.absent()));
                let offsets = m.col_offsets();
                for col in 0..m.n_cols() {
                    for k in offsets[col]..offsets[col + 1] {
                        part[[m.row_indices()[k], col]] = m.values()[k];
                    }
                }
                start += m.n_cols();
            }
            _ => unreachable!("a kind of block these helpers don't know"),
        }
    }
    out
}

/// The single dense block of `data`, borrowed: for checking that something was
/// moved rather than copied, or which memory order it has.
///
/// # Panics
///
/// Unless `data` is one dense block.
pub fn only_dense<'b, 'a, F: Float, O: Orientation, D: Density, S: State, Y: TargetState>(
    data: &'b Dataset<'a, F, Blocks<'a, F, O, D>, S, Y>,
) -> ArrayView2<'b, F> {
    let mut blocks = data.blocks();
    match (blocks.next(), blocks.next()) {
        (Some(BlockView::DenseRowMajor(x) | BlockView::DenseColMajor(x)), None) => x,
        _ => panic!("expected one dense block"),
    }
}

fn absent_value<F: Float>(absent: Absent) -> F {
    match absent {
        Absent::Zero => F::zero(),
        Absent::Missing => F::nan(),
        _ => unreachable!("a meaning these helpers don't know"),
    }
}
