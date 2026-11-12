//! # predict
//!
//! Machine learning for Rust with a consistent, scikit-learn-style API.
//!
//! **This crate is a work in progress.** This early release reserves the
//! crate name; the real API is under development and will change soon.

pub mod dataset;
mod error;
mod float;
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
