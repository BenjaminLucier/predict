//! Markers for what is known about a dataset's feature values.
//!
//! Each state is a zero-sized type used only at the type level: a
//! `Dataset<Finite>` and a `Dataset<Unchecked>` hold the same data, but only the
//! first can be passed to a model that needs finite input. The three states are
//! ordered by what they guarantee:
//!
//! `Unchecked` (nothing) ⊂ `NoInfinity` (no ±inf, NaN allowed) ⊂ `Finite` (neither).

use std::fmt::Debug;

mod sealed {
    pub trait Sealed {}
}

/// Implemented by [`Unchecked`], [`NoInfinity`] and [`Finite`], and nothing else.
///
/// The trait is sealed: code outside this crate can name it in bounds but cannot
/// implement it, so a state can only be reached through a check this crate made.
pub trait State: sealed::Sealed + Debug + Clone + Copy + Send + Sync + 'static {
    /// The state reached when infinity is also ruled out: `Unchecked` becomes
    /// `NoInfinity`, and the two infinity-free states stay as they are.
    ///
    /// For outputs: a transformer that checks what it writes for infinity (a
    /// scaler) outputs `S::WithoutInfinity`, so scaling `Finite` data still gives
    /// `Finite` data.
    type WithoutInfinity: InfinityFree;
}

/// The states with no ±infinity: [`NoInfinity`] and [`Finite`].
///
/// For inputs: a model that handles missing values (NaN) but not infinity is
/// bounded by `S: InfinityFree`.
pub trait InfinityFree: State {}

/// Nothing is known about the values: they may include NaN or ±infinity.
#[derive(Debug, Clone, Copy)]
pub struct Unchecked;

/// No value is ±infinity; NaN, meaning a missing value, is allowed.
#[derive(Debug, Clone, Copy)]
pub struct NoInfinity;

/// Every value is finite: no NaN and no ±infinity.
#[derive(Debug, Clone, Copy)]
pub struct Finite;

impl sealed::Sealed for Unchecked {}
impl sealed::Sealed for NoInfinity {}
impl sealed::Sealed for Finite {}

impl State for Unchecked {
    type WithoutInfinity = NoInfinity;
}
impl State for NoInfinity {
    type WithoutInfinity = NoInfinity;
}
impl State for Finite {
    type WithoutInfinity = Finite;
}

impl InfinityFree for NoInfinity {}
impl InfinityFree for Finite {}
