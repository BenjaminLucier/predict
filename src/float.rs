//! The floating-point types this crate works with.

use ndarray::NdFloat;

mod sealed {
    // faer's real-number trait sits on this private supertrait, so generic
    // code can hand `F` to faer (the least squares SVD) while no faer type or
    // trait appears in this crate's signatures.
    pub trait Sealed: faer::traits::RealField {}
}

/// Implemented by `f32` and `f64`, and nothing else.
///
/// Every type that holds feature values or regression targets is generic over
/// `F: Float`, with `f64` as the default. `f32` halves memory and is faster
/// where the arithmetic allows it.
///
/// It builds on ndarray's [`NdFloat`], which gives the arithmetic operators,
/// formatting, and the methods of `num_traits::Float` (`is_finite`, `sqrt`,
/// ...). `Into<f64>` lets an error report a value whatever its type (widening
/// `f32` to `f64` is exact). The trait is sealed, so it can gain methods without a
/// breaking change.
pub trait Float: sealed::Sealed + NdFloat + Into<f64> {}

impl sealed::Sealed for f32 {}
impl sealed::Sealed for f64 {}

impl Float for f32 {}
impl Float for f64 {}
