use ndarray::{Array1, Array2, ArrayView1, ShapeBuilder, array, s};
use predict::dataset::{ColMajorMatrix, DenseRowMajor, Finite, Targets};
use predict::linear::{FittedLinearRegressor, LinearRegression};
use predict::metrics::R2;
use predict::selection::{CrossValidation, KFold};
use predict::synthetic::MakeRegression;
use predict::{Dataset, Fit, Predict, PredictError, Result, Seed};

type Data<'a> = Dataset<'a, f64, DenseRowMajor<'a>, Finite, Targets<'a, f64>>;

fn data<'a>(x: &'a Array2<f64>, y: &'a Array1<f64>) -> Data<'a> {
    Dataset::new(x)
        .validate_finite()
        .unwrap()
        .with_y(y)
        .unwrap()
}

fn assert_close(actual: ArrayView1<'_, f64>, expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= tol, "{actual} vs {expected:?}");
    }
}

fn fit(data: &Data<'_>) -> FittedLinearRegressor {
    LinearRegression::new().fit(data).unwrap()
}

/// Shared by the scikit-learn cases: six rows, two columns, slightly noisy.
fn six_rows() -> (Array2<f64>, Array1<f64>) {
    let x = array![
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0],
        [5.0, 7.0],
        [6.0, 5.0]
    ];
    (x, array![3.1, 2.9, 7.2, 6.8, 12.5, 10.9])
}

// --- The solution -------------------------------------------------------------

#[test]
fn noiseless_targets_give_back_the_true_coefficients() {
    let x = array![
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 1.0],
        [2.0, 2.0, 0.0],
        [3.0, 1.0, 1.0],
        [1.0, 4.0, 2.0]
    ];
    let (coef, intercept) = (array![2.0, -1.0, 0.5], 3.0);
    let y = x.dot(&coef) + intercept;
    let model = fit(&data(&x, &y));
    assert_close(model.coef(), coef.as_slice().unwrap(), 1e-12);
    assert!((model.intercept() - intercept).abs() < 1e-12);
}

#[test]
fn residuals_are_perpendicular_to_every_column_and_sum_to_zero() {
    let (x, y) = six_rows();
    let model = fit(&data(&x, &y));
    let residuals = &y - &model.predict(&data(&x, &y)).unwrap();
    // The normal equations, Xᵀr = 0, and with an intercept Σr = 0.
    assert_close(x.t().dot(&residuals).view(), &[0.0, 0.0], 1e-12);
    assert!(residuals.sum().abs() < 1e-12);
}

// Expected values from scikit-learn 1.x `LinearRegression` on the same inputs
// (a local script in the git-ignored `reference/python/linear.py`).

#[test]
fn matches_scikit_learn_on_tall_data() {
    let (x, y) = six_rows();
    let model = fit(&data(&x, &y));
    assert_close(
        model.coef(),
        &[0.8380743982494527, 1.189606126914661],
        1e-12,
    );
    assert!((model.intercept() - -0.06181619256017523).abs() < 1e-12);
}

#[test]
fn matches_scikit_learn_with_weights() {
    let (x, y) = six_rows();
    let w = array![1.0, 2.0, 0.5, 3.0, 1.0, 0.0];
    let weighted = data(&x, &y).with_sample_weights(&w).unwrap();
    let model = fit(&weighted);
    assert_close(
        model.coef(),
        &[0.8217257318952236, 1.1924499229583976],
        1e-12,
    );
    assert!((model.intercept() - -0.01926040061633394).abs() < 1e-12);
}

#[test]
fn matches_scikit_learn_without_an_intercept() {
    let (x, y) = six_rows();
    let model = LinearRegression::new()
        .without_intercept()
        .fit(&data(&x, &y))
        .unwrap();
    assert_close(
        model.coef(),
        &[0.8276073619631901, 1.1858895705521477],
        1e-12,
    );
    assert_eq!(model.intercept(), 0.0);
}

#[test]
fn a_duplicated_column_splits_its_coefficient_evenly() {
    let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [5.0, 5.0]];
    let y = array![1.0, 2.5, 2.9, 5.2];
    let model = fit(&data(&x, &y));
    // The minimum-norm solution, as scikit-learn gives.
    assert_close(
        model.coef(),
        &[0.5028571428571429, 0.502857142857143],
        1e-12,
    );
    assert!((model.intercept() - 0.13428571428571434).abs() < 1e-12);
}

#[test]
fn more_features_than_rows_gives_the_minimum_norm_solution() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 0.0, 1.0]];
    let y = array![1.0, 2.0];
    let model = LinearRegression::new()
        .without_intercept()
        .fit(&data(&x, &y))
        .unwrap();
    assert_close(
        model.coef(),
        &[0.4603174603174604, 0.031746031746031834, 0.1587301587301588],
        1e-12,
    );
    // Wide data fits exactly.
    assert_close(
        model.predict(&data(&x, &y)).unwrap().view(),
        &[1.0, 2.0],
        1e-12,
    );
}

// --- Weights -------------------------------------------------------------------

#[test]
fn integer_weights_fit_like_repeated_rows_and_zero_like_a_dropped_row() {
    let (x, y) = six_rows();
    let w = array![2.0, 1.0, 0.0, 1.0, 3.0, 1.0];
    let weighted = fit(&data(&x, &y).with_sample_weights(&w).unwrap());

    let rows = [0, 0, 1, 3, 4, 4, 4, 5];
    let repeated_x = x.select(ndarray::Axis(0), &rows);
    let repeated_y = y.select(ndarray::Axis(0), &rows);
    let repeated = fit(&data(&repeated_x, &repeated_y));

    assert_close(weighted.coef(), repeated.coef().as_slice().unwrap(), 1e-12);
    assert!((weighted.intercept() - repeated.intercept()).abs() < 1e-12);
}

#[test]
fn weights_adding_up_to_zero_are_undefined() {
    let (x, y) = six_rows();
    let w = Array1::zeros(6);
    let result = LinearRegression::new().fit(&data(&x, &y).with_sample_weights(&w).unwrap());
    assert!(matches!(
        result,
        Err(PredictError::Undefined {
            context: "LinearRegressor",
            ..
        })
    ));
}

#[test]
fn no_rows_is_an_empty_input() {
    let (x, y) = six_rows();
    let full = data(&x, &y);
    let result = LinearRegression::new().fit(&full.select_rows(&[]));
    assert!(matches!(
        result,
        Err(PredictError::EmptyInput {
            context: "LinearRegressor"
        })
    ));
}

// --- Layouts and types ---------------------------------------------------------

#[test]
fn every_dense_layout_gives_the_same_fit() {
    let (x, y) = six_rows();
    let expected = fit(&data(&x, &y));

    let mut col_major = Array2::zeros((6, 2).f());
    col_major.assign(&x);
    let cols = Dataset::from_col_major(ColMajorMatrix::new(&col_major).unwrap())
        .validate_finite()
        .unwrap()
        .with_y(&y)
        .unwrap();
    let from_cols = LinearRegression::new().fit(&cols).unwrap();

    // One row-major column beside one column-major column.
    let left = x.slice(s![.., ..1]);
    let mut right = Array2::zeros((6, 1).f());
    right.assign(&x.slice(s![.., 1..]));
    let mixed = Dataset::new(left)
        .hstack(Dataset::from_col_major(
            ColMajorMatrix::new(&right).unwrap(),
        ))
        .unwrap()
        .validate_finite()
        .unwrap()
        .with_y(&y)
        .unwrap();
    let from_mixed = LinearRegression::new().fit(&mixed).unwrap();

    for model in [&from_cols, &from_mixed] {
        assert_close(model.coef(), expected.coef().as_slice().unwrap(), 1e-12);
        assert!((model.intercept() - expected.intercept()).abs() < 1e-12);
    }
    assert_eq!(
        from_mixed.predict(&mixed).unwrap(),
        expected.predict(&data(&x, &y)).unwrap()
    );
}

#[test]
fn f32_fits_to_f32_accuracy() {
    let x = array![[1.0_f32, 0.0], [0.0, 1.0], [2.0, 2.0], [3.0, 1.0]];
    let y = x.dot(&array![2.0_f32, -1.0]) + 0.5;
    let model = LinearRegression::new()
        .fit(
            &Dataset::new(&x)
                .validate_finite()
                .unwrap()
                .with_y(&y)
                .unwrap(),
        )
        .unwrap();
    let coef: ArrayView1<'_, f32> = model.coef();
    assert!((coef[0] - 2.0).abs() < 1e-5 && (coef[1] + 1.0).abs() < 1e-5);
    assert!((model.intercept() - 0.5).abs() < 1e-5);
}

#[test]
fn predict_checks_the_number_of_features() {
    let (x, y) = six_rows();
    let model = fit(&data(&x, &y));
    let narrow = Array2::<f64>::zeros((2, 1));
    let result = model.predict(&Dataset::new(&narrow).validate_finite().unwrap());
    assert!(matches!(
        result,
        Err(PredictError::ShapeMismatch {
            context: "x",
            expected: 2,
            actual: 1
        })
    ));
}

// --- With the rest of the crate ------------------------------------------------

#[test]
fn cross_validates_near_perfectly_on_low_noise_data() -> Result<()> {
    let generated = MakeRegression::new()
        .with_n_samples(200)
        .with_noise(0.1)
        .with_seed(3)
        .generate()?;
    let cv = CrossValidation::new(KFold::new().with_shuffle(Seed::Fixed(1)));
    let scores = cv.score(&LinearRegression::new(), &generated.dataset()?, (), &R2)?;
    assert!(scores.iter().all(|&r2| r2 > 0.99), "{scores}");
    Ok(())
}
