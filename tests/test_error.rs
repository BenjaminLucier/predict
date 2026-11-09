use predict::PredictError;

#[test]
fn messages_name_the_source_and_the_location() {
    let err = PredictError::NonFinite {
        context: "y",
        row: 41,
        col: 0,
        value: f64::NAN,
    };
    assert_eq!(
        err.to_string(),
        "non-finite value NaN in y at row 41, column 0"
    );

    let err = PredictError::InvalidValue {
        context: "sample_weights",
        row: 3,
        col: 0,
        value: -1.0,
        reason: "must be non-negative".to_string(),
    };
    assert_eq!(
        err.to_string(),
        "invalid value -1 in sample_weights at row 3, column 0: must be non-negative"
    );

    let err = PredictError::ShapeMismatch {
        context: "y",
        expected: 100,
        actual: 99,
    };
    assert_eq!(err.to_string(), "y: expected 100, got 99");

    let err = PredictError::InvalidParameter {
        name: "max_iter",
        reason: "must be at least 1".to_string(),
    };
    assert_eq!(
        err.to_string(),
        "invalid parameter `max_iter`: must be at least 1"
    );
}

/// `Send + Sync + 'static` is what `anyhow`'s `?` conversion, `std::thread::spawn`
/// results and `Box<dyn Error + Send + Sync>` require. The compiler derives the
/// three automatically from the fields, so nothing in `PredictError` says so; this
/// check makes a future variant that loses one (a borrowed `&'a str`, a boxed error
/// without `Send + Sync`) fail here instead of in users' code. It is checked at
/// compile time: the call does nothing at run time.
#[test]
fn errors_cross_threads_and_box_into_dyn_error() {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<PredictError>();

    let boxed: Box<dyn std::error::Error + Send + Sync> =
        Box::new(PredictError::EmptyInput { context: "x" });
    assert_eq!(boxed.to_string(), "x: empty input");
}
