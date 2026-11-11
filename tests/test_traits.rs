use ndarray::{Array2, array};
use predict::dataset::{DenseRowMajor, Finite, TargetState};
use predict::{Dataset, Fit, FitTransform, PredictError, Result, Transform, ValidateConfig};

mod common;
use common::only_dense;

/// A transformer for the tests: learns each column's mean and subtracts it.
#[derive(Debug, Default)]
struct Center {
    fail: bool,
}

#[derive(Debug)]
struct FittedCenter {
    means: ndarray::Array1<f64>,
}

impl ValidateConfig for Center {
    fn validate_config(&self) -> Result<()> {
        if self.fail {
            return Err(PredictError::InvalidParameter {
                name: "fail",
                reason: "set for the test".into(),
            });
        }
        Ok(())
    }
}

impl<'a, Y: TargetState> Fit<Dataset<'a, f64, DenseRowMajor<'a>, Finite, Y>> for Center {
    type Fitted = FittedCenter;

    fn fit(&self, data: &Dataset<'a, f64, DenseRowMajor<'a>, Finite, Y>) -> Result<FittedCenter> {
        self.validate_config()?;
        let means = only_dense(data).mean_axis(ndarray::Axis(0)).unwrap();
        Ok(FittedCenter { means })
    }
}

impl<'a, Y: TargetState> Transform<Dataset<'a, f64, DenseRowMajor<'a>, Finite, Y>>
    for FittedCenter
{
    type Output = Array2<f64>;

    fn transform(&self, x: Dataset<'a, f64, DenseRowMajor<'a>, Finite, Y>) -> Result<Array2<f64>> {
        Ok(&only_dense(&x) - &self.means)
    }
}

impl<'a, Y: TargetState> FitTransform<Dataset<'a, f64, DenseRowMajor<'a>, Finite, Y>> for Center {}

fn finite(x: &Array2<f64>) -> Dataset<'_, f64, DenseRowMajor<'_>, Finite> {
    Dataset::new(x).validate_finite().unwrap()
}

// --- Fit, transform, fit_transform ---------------------------------------------

#[test]
fn fit_transform_is_fit_then_transform() {
    let x = array![[1.0, 10.0], [3.0, 30.0]];
    let fitted = Center::default().fit(&finite(&x)).unwrap();
    let separately = fitted.transform(finite(&x)).unwrap();

    let (fitted, together) = Center::default().fit_transform(finite(&x)).unwrap();
    assert_eq!(fitted.means, array![2.0, 20.0]);
    assert_eq!(together, separately);
    assert_eq!(together, array![[-1.0, -10.0], [1.0, 10.0]]);
}

#[test]
fn fit_checks_the_config_first() {
    let x = array![[1.0]];
    let err = Center { fail: true }.fit(&finite(&x)).unwrap_err();
    assert!(matches!(
        err,
        PredictError::InvalidParameter { name: "fail", .. }
    ));
    assert!(Center { fail: true }.fit_transform(finite(&x)).is_err());
}

#[test]
fn the_same_config_fits_with_or_without_targets() {
    let (x, y) = (array![[1.0], [3.0]], array![0.0, 1.0]);
    let with_y = finite(&x).with_y(&y).unwrap();
    assert_eq!(Center::default().fit(&with_y).unwrap().means, array![2.0]);
}

// --- view --------------------------------------------------------------------------

#[test]
fn a_view_borrows_features_targets_and_weights_without_copying() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let (y, w) = (array![1.0, 2.0], array![1.0, 3.0]);
    // Owned, so a copy would show up as a different pointer.
    let data = Dataset::new(x.clone())
        .validate_finite()
        .unwrap()
        .with_y(y.clone())
        .unwrap()
        .with_sample_weights(w.clone())
        .unwrap();

    let view = data.view();
    assert_eq!(only_dense(&view).as_ptr(), only_dense(&data).as_ptr());
    assert_eq!(view.y().as_ptr(), data.y().as_ptr());
    assert_eq!(
        view.sample_weights().unwrap().as_ptr(),
        data.sample_weights().unwrap().as_ptr()
    );

    // The view keeps the state and targets, so it fits like the dataset.
    let _: &Dataset<'_, f64, DenseRowMajor<'_>, Finite, _> = &view;
    assert_eq!(view.y(), y);
    assert_eq!(view.sample_weights().unwrap(), w);
}

#[test]
fn a_view_can_be_transformed_leaving_the_dataset_untouched() {
    let x = array![[1.0], [3.0]];
    let data = Dataset::new(x.clone()).validate_finite().unwrap();
    let fitted = Center::default().fit(&data).unwrap();
    // `transform` takes ownership; a view lets the dataset be used again.
    let once = fitted.transform(data.view()).unwrap();
    let twice = fitted.transform(data.view()).unwrap();
    assert_eq!(once, twice);
    assert_eq!(only_dense(&data), x);
}
