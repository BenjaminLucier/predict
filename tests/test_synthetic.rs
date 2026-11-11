use ndarray::{Array1, Axis, concatenate, s};
use predict::PredictError;
use predict::synthetic::{MakeBlobs, MakeClassification, MakeRegression, Samples};
use std::collections::HashSet;

fn regression() -> MakeRegression {
    MakeRegression::new().with_seed(7)
}

// --- MakeRegression: the truth -----------------------------------------------------

#[test]
fn the_defaults_draw_100_rows_of_10_features_5_informative() {
    let data = regression().generate().unwrap();
    assert_eq!(data.x.dim(), (100, 10));
    assert_eq!(data.y.len(), 100);
    assert!(
        data.coef
            .slice(s![..5])
            .iter()
            .all(|&c| c > 0.0 && c <= 100.0)
    );
    assert!(data.coef.slice(s![5..]).iter().all(|&c| c == 0.0));
    assert_eq!(data.intercept, 0.0);
}

#[test]
fn without_noise_the_targets_are_the_linear_function() {
    let data = regression().with_bias(3.5).generate().unwrap();
    let expected = data.x.dot(&data.coef) + 3.5;
    // Per-row dot products and a matrix product round differently.
    assert!((&data.y - &expected).iter().all(|d| d.abs() < 1e-9));
}

#[test]
fn the_noise_has_the_requested_standard_deviation() {
    let data = regression()
        .with_n_samples(20_000)
        .with_noise(2.0)
        .generate()
        .unwrap();
    let residuals = &data.y - &data.x.dot(&data.coef);
    // The sample standard deviation of 20 000 draws is within 2% of the
    // true one with overwhelming probability, and the seed is fixed.
    let std = residuals.std(0.0);
    assert!((std - 2.0).abs() < 0.04, "{std}");
}

#[test]
fn the_features_are_the_same_at_every_noise_level() {
    let quiet = regression().generate().unwrap();
    let noisy = regression().with_noise(10.0).generate().unwrap();
    assert_eq!(quiet.x, noisy.x);
    assert_eq!(quiet.coef, noisy.coef);
    assert_ne!(quiet.y, noisy.y);
}

// --- MakeRegression: seeds ---------------------------------------------------------

#[test]
fn a_seed_repeats_the_data_and_another_seed_changes_it() {
    let a = regression().generate().unwrap();
    let b = regression().generate().unwrap();
    assert_eq!((&a.x, &a.y, &a.coef), (&b.x, &b.y, &b.coef));

    let c = MakeRegression::new().with_seed(8).generate().unwrap();
    assert_ne!(a.x, c.x);
}

#[test]
fn without_a_seed_every_run_differs() {
    let a = MakeRegression::new().generate().unwrap();
    let b = MakeRegression::new().generate().unwrap();
    assert_ne!(a.x, b.x);
}

// --- MakeRegression: streams and batches -----------------------------------------

#[test]
fn generate_is_the_start_of_the_stream_however_it_is_drawn() {
    let data = regression().with_n_samples(6).generate().unwrap();

    // Row by row.
    let rows: Vec<(Array1<f64>, f64)> = regression().stream().unwrap().take(6).collect();
    for (i, (x, y)) in rows.iter().enumerate() {
        assert_eq!(x, &data.x.row(i));
        assert_eq!(*y, data.y[i]);
    }

    // In uneven batches.
    let mut stream = regression().stream().unwrap();
    let (first, second) = (stream.next_batch(4), stream.next_batch(2));
    let x = concatenate(Axis(0), &[first.x.view(), second.x.view()]).unwrap();
    let y = concatenate(Axis(0), &[first.y.view(), second.y.view()]).unwrap();
    assert_eq!((x, y), (data.x, data.y));

    // And the stream knows the same truth.
    assert_eq!(stream.coef(), data.coef);
    assert_eq!(stream.intercept(), data.intercept);
}

#[test]
fn batches_are_consecutive_and_the_stream_continues_after_them() {
    let data = regression().with_n_samples(7).generate().unwrap();
    let mut stream = regression().stream().unwrap();
    let batches: Vec<Samples<f64>> = stream.batches(3).take(2).collect();
    assert_eq!(batches[0].x, data.x.slice(s![..3, ..]));
    assert_eq!(batches[1].x, data.x.slice(s![3..6, ..]));
    let (x, y) = stream.next().unwrap();
    assert_eq!((x.view(), y), (data.x.row(6), data.y[6]));
}

#[test]
#[should_panic(expected = "batch_size must be positive")]
fn batches_of_no_rows_panic() {
    let mut stream = regression().stream().unwrap();
    let _ = stream.batches(0);
}

// --- MakeRegression: datasets ------------------------------------------------------

#[test]
fn the_data_and_batches_become_datasets_without_copying() {
    let data = regression().generate().unwrap();
    let dataset = data.dataset().unwrap();
    assert_eq!(dataset.y().as_ptr(), data.y.as_ptr());
    assert_eq!(dataset.n_features(), 10);

    let batch = regression().stream().unwrap().next_batch(5);
    assert_eq!(batch.dataset().unwrap().n_samples(), 5);
}

#[test]
fn an_overflowing_target_is_caught_when_making_the_dataset() {
    // Finite settings, but `f64::MAX` times a normal draw above 1 is infinite.
    let data = regression().with_noise(f64::MAX).generate().unwrap();
    let err = data.dataset().unwrap_err();
    assert!(matches!(err, PredictError::NonFinite { context: "y", .. }));
}

// --- MakeRegression: settings ------------------------------------------------------

#[test]
fn bad_settings_name_the_setting() {
    let cases = [
        ("n_samples", regression().with_n_samples(0)),
        ("n_features", regression().with_n_features(0)),
        ("n_informative", regression().with_n_informative(11)),
        ("noise", regression().with_noise(-1.0)),
        ("noise", regression().with_noise(f64::NAN)),
        ("bias", regression().with_bias(f64::INFINITY)),
    ];
    for (expected, config) in cases {
        match config.generate() {
            Err(PredictError::InvalidParameter { name, .. }) => assert_eq!(name, expected),
            other => panic!("{expected}: {other:?}"),
        }
    }
}

#[test]
fn no_informative_features_leave_only_the_bias_and_noise() {
    let data = regression()
        .with_n_informative(0)
        .with_bias(4.0)
        .generate()
        .unwrap();
    assert!(data.coef.iter().all(|&c| c == 0.0));
    assert!(data.y.iter().all(|&y| y == 4.0));
}

#[test]
fn a_stream_does_not_need_n_samples() {
    assert!(regression().with_n_samples(0).stream().is_ok());
}

// --- MakeBlobs -----------------------------------------------------------------------

fn blobs() -> MakeBlobs {
    MakeBlobs::new().with_seed(7)
}

fn counts(y: &Array1<usize>, n_classes: usize) -> Vec<usize> {
    let mut counts = vec![0; n_classes];
    for &class in y {
        counts[class] += 1;
    }
    counts
}

#[test]
fn blobs_are_split_evenly_and_drawn_around_their_centres() {
    let data = blobs().with_n_samples(3_000).generate().unwrap();
    assert_eq!(data.x.dim(), (3_000, 2));
    assert_eq!(data.centers.dim(), (3, 2));
    assert!(data.centers.iter().all(|&c| (-10.0..10.0).contains(&c)));
    assert_eq!(counts(&data.y, 3), vec![1_000; 3]);

    // Each cluster's mean is near its centre: within 0.1 for 1 000 rows of
    // standard deviation 1 (more than 3 standard errors).
    for class in 0..3 {
        let rows: Vec<usize> = (0..3_000).filter(|&i| data.y[i] == class).collect();
        let mean = data.x.select(Axis(0), &rows).mean_axis(Axis(0)).unwrap();
        let error = &mean - &data.centers.row(class);
        assert!(error.iter().all(|e| e.abs() < 0.1), "{class}: {error}");
    }
}

#[test]
fn leftover_rows_go_to_the_lowest_classes_and_rows_are_shuffled() {
    let data = blobs().with_n_samples(10).generate().unwrap();
    assert_eq!(counts(&data.y, 3), vec![4, 3, 3]);
    let data = blobs().with_n_samples(300).generate().unwrap();
    assert!(data.y.windows(2).into_iter().any(|w| w[0] > w[1]));
}

#[test]
fn a_blobs_stream_keeps_the_centres_and_picks_centres_at_random() {
    let data = blobs().generate().unwrap();
    let mut stream = blobs().stream().unwrap();
    assert_eq!(stream.centers(), data.centers);

    let batch = stream.next_batch(3_000);
    let counts = counts(&batch.y, 3);
    // About 1 000 each, but not exactly: the stream doesn't fix sizes.
    assert!(
        counts.iter().all(|&n| (900..1_100).contains(&n)),
        "{counts:?}"
    );
    assert_eq!(batch.dataset().unwrap().n_samples(), 3_000);
}

#[test]
fn blobs_streams_give_the_same_rows_however_they_are_batched() {
    let mut whole = blobs().stream().unwrap();
    let all = whole.next_batch(5);
    let mut split = blobs().stream().unwrap();
    let parts: Vec<Samples<usize>> = split.batches(2).take(2).collect();
    let (x, y) = split.next().unwrap();
    assert_eq!(parts[0].x, all.x.slice(s![..2, ..]));
    assert_eq!(parts[1].y, all.y.slice(s![2..4]));
    assert_eq!((x.view(), y), (all.x.row(4), all.y[4]));
}

#[test]
fn zero_spread_puts_every_row_on_its_centre() {
    let data = blobs().with_cluster_std(0.0).generate().unwrap();
    for (row, &class) in data.x.rows().into_iter().zip(&data.y) {
        assert_eq!(row, data.centers.row(class));
    }
}

#[test]
fn bad_blob_settings_name_the_setting() {
    let cases = [
        ("n_samples", blobs().with_n_samples(0)),
        ("n_features", blobs().with_n_features(0)),
        ("n_centers", blobs().with_n_centers(0)),
        ("cluster_std", blobs().with_cluster_std(-1.0)),
        ("center_box", blobs().with_center_box(1.0, 1.0)),
        ("center_box", blobs().with_center_box(f64::NAN, 1.0)),
        ("center_box", blobs().with_center_box(-f64::MAX, f64::MAX)),
    ];
    for (expected, config) in cases {
        match config.generate() {
            Err(PredictError::InvalidParameter { name, .. }) => assert_eq!(name, expected),
            other => panic!("{expected}: {other:?}"),
        }
    }
}

// --- MakeClassification ---------------------------------------------------------------

fn classification() -> MakeClassification {
    MakeClassification::new().with_seed(7)
}

#[test]
fn classes_sit_on_distinct_corners_at_class_sep() {
    let data = classification()
        .with_n_classes(8)
        .with_n_informative(3)
        .with_class_sep(2.5)
        .generate()
        .unwrap();
    assert_eq!(data.centers.dim(), (8, 3));
    assert!(data.centers.iter().all(|&c| c.abs() == 2.5));
    // 8 classes on the 8 corners of a cube: every corner, once each.
    let corners: HashSet<Vec<bool>> = data
        .centers
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|&c| c > 0.0).collect())
        .collect();
    assert_eq!(corners.len(), 8);
}

#[test]
fn the_columns_are_informative_redundant_repeated_then_noise() {
    let data = classification()
        .with_n_features(9)
        .with_n_informative(3)
        .with_n_redundant(2)
        .with_n_repeated(2)
        .with_flip_y(0.0)
        .generate()
        .unwrap();
    let informative = data.x.slice(s![.., ..3]);
    let redundant = informative.dot(&data.redundant);
    assert!(data.redundant.iter().all(|&w| (-1.0..1.0).contains(&w)));
    assert!(
        (&data.x.slice(s![.., 3..5]) - &redundant)
            .iter()
            .all(|d| d.abs() < 1e-12)
    );
    for (offset, &source) in data.repeated.iter().enumerate() {
        assert!(source < 5);
        assert_eq!(data.x.column(5 + offset), data.x.column(source));
    }
    // The last two are noise, unrelated to the class: means near 0 either way.
    let class_mean = |class, col| {
        let rows: Vec<usize> = (0..100).filter(|&i| data.y[i] == class).collect();
        data.x.select(Axis(0), &rows).column(col).mean().unwrap()
    };
    assert!(class_mean(0, 8).abs() < 0.5 && class_mean(1, 8).abs() < 0.5);
}

#[test]
fn class_weights_set_the_sizes_exactly_before_flipping() {
    let data = classification()
        .with_n_samples(100)
        .with_class_weights(vec![0.9, 0.1])
        .with_flip_y(0.0)
        .generate()
        .unwrap();
    assert_eq!(counts(&data.y, 2), vec![90, 10]);
}

#[test]
fn flip_y_changes_only_classes_and_about_the_expected_share() {
    let config = classification().with_n_samples(20_000);
    let kept = config.clone().with_flip_y(0.0).generate().unwrap();
    let flipped = config.with_flip_y(0.2).generate().unwrap();
    assert_eq!(kept.x, flipped.x);
    // A flip picks a random class, the same one half the time with 2 classes,
    // so about 10% of classes change.
    let changed = kept
        .y
        .iter()
        .zip(&flipped.y)
        .filter(|(a, b)| a != b)
        .count();
    assert!((1_800..2_200).contains(&changed), "{changed}");
}

#[test]
fn larger_class_sep_separates_the_classes_further() {
    // Nearest corner is the best possible classifier here (unit noise, round
    // classes, no flips), and its accuracy approaches 1 - Φ(-sep) per axis.
    let accuracy = |sep: f64| {
        let data = classification()
            .with_n_samples(4_000)
            .with_n_informative(1)
            .with_n_redundant(0)
            .with_n_features(1)
            .with_class_sep(sep)
            .with_flip_y(0.0)
            .generate()
            .unwrap();
        let positive_class = usize::from(data.centers[[1, 0]] > 0.0);
        let correct = (0..4_000)
            .filter(|&i| {
                let guess = if data.x[[i, 0]] > 0.0 {
                    positive_class
                } else {
                    1 - positive_class
                };
                guess == data.y[i]
            })
            .count();
        correct as f64 / 4_000.0
    };
    // Φ(-1) ≈ 0.159 and Φ(-2) ≈ 0.023: accuracies near 0.841 and 0.977.
    assert!((accuracy(1.0) - 0.841).abs() < 0.02);
    assert!((accuracy(2.0) - 0.977).abs() < 0.01);
}

#[test]
fn a_classification_stream_keeps_the_truth_and_batches_alike() {
    let data = classification().generate().unwrap();
    let mut stream = classification().stream().unwrap();
    assert_eq!(stream.centers(), data.centers);
    assert_eq!(stream.redundant(), data.redundant);
    assert_eq!(stream.repeated(), &data.repeated[..]);

    let mut again = classification().stream().unwrap();
    let all = again.next_batch(4);
    let parts: Vec<Samples<usize>> = stream.batches(2).take(2).collect();
    assert_eq!(parts[0].x, all.x.slice(s![..2, ..]));
    assert_eq!(parts[1].y, all.y.slice(s![2..]));
    assert!(parts[0].dataset().is_ok());
}

#[test]
fn many_dimensions_and_many_classes_are_fine() {
    // Beyond the 63 dimensions an integer can index, and 2^16 classes on 16.
    let wide = classification()
        .with_n_informative(70)
        .with_n_redundant(0)
        .with_n_features(70);
    assert!(wide.generate().is_ok());
    let many = classification()
        .with_n_samples(1)
        .with_n_classes(1 << 16)
        .with_n_informative(16);
    assert_eq!(many.generate().unwrap().centers.nrows(), 1 << 16);
}

#[test]
fn bad_classification_settings_name_the_setting() {
    let cases = [
        ("n_samples", classification().with_n_samples(0)),
        ("n_classes", classification().with_n_classes(0)),
        ("n_informative", classification().with_n_informative(0)),
        ("n_features", classification().with_n_features(3)),
        ("n_features", classification().with_n_repeated(usize::MAX)),
        ("n_classes", classification().with_n_classes(5)),
        ("class_sep", classification().with_class_sep(0.0)),
        ("flip_y", classification().with_flip_y(1.5)),
        (
            "class_weights",
            classification().with_class_weights(vec![1.0]),
        ),
        (
            "class_weights",
            classification().with_class_weights(vec![1.0, -1.0]),
        ),
        (
            "class_weights",
            classification().with_class_weights(vec![0.0, 0.0]),
        ),
        (
            "class_weights",
            classification().with_class_weights(vec![f64::MAX, f64::MAX]),
        ),
    ];
    for (expected, config) in cases {
        match config.generate() {
            Err(PredictError::InvalidParameter { name, .. }) => assert_eq!(name, expected),
            other => panic!("{expected}: {other:?}"),
        }
    }
}
