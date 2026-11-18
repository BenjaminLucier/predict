//! # predict
//!
//! Machine learning for Rust with a consistent, scikit-learn-style API.
//!
//! **Early development:** every `0.0.x` release may break the API.
//!
//! Today it has [datasets](dataset) with dense and sparse features, targets
//! and sample weights; [synthetic data](synthetic); [metrics]; [model
//! selection](selection) (splits and cross-validation); and [linear
//! regression](linear). The book at <https://predict.rs> explains the design
//! and the maths.

pub mod dataset;
mod error;
mod float;
pub mod linear;
pub mod metrics;
mod seed;
pub mod selection;
pub mod synthetic;
mod traits;

pub use dataset::Dataset;
pub use error::{PredictError, Result};
pub use float::Float;
pub use seed::Seed;
pub use traits::{Fit, FitTransform, Predict, PredictProbability, Transform, ValidateConfig};
