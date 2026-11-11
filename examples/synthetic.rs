//! Synthetic data with a known truth. Run with `cargo run --example synthetic`.

use ndarray::Axis;
use predict::Result;
use predict::synthetic::{MakeBlobs, MakeClassification, MakeRegression};

fn main() -> Result<()> {
    // ANCHOR: regression
    let data = MakeRegression::new()
        .with_n_samples(200)
        .with_noise(0.5)
        .with_seed(42)
        .generate()?;

    // The truth the targets were drawn from: a model fitted on
    // `data.dataset()?` should recover it.
    println!("true coefficients: {:.1}", data.coef);
    let dataset = data.dataset()?; // finite, with targets, borrowing x and y
    // ANCHOR_END: regression
    println!(
        "{} rows of {} features",
        dataset.n_samples(),
        dataset.n_features()
    );

    // ANCHOR: stream
    let mut stream = MakeRegression::new().with_seed(42).stream()?;
    let truth = stream.coef().to_owned();

    // One row at a time...
    let (x, y) = stream.next().expect("streams never end");

    // ...or in batches, each a matrix that becomes a dataset.
    for batch in stream.batches(1_000).take(3) {
        let dataset = batch.dataset()?;
        assert_eq!(dataset.n_samples(), 1_000);
    }
    // ANCHOR_END: stream
    println!("first streamed row: {x:.2} -> {y:.2}; truth {truth:.1}");

    // ANCHOR: blobs
    let blobs = MakeBlobs::new()
        .with_n_centers(4)
        .with_cluster_std(0.5)
        .with_seed(1)
        .generate()?;
    // `blobs.y` says which of `blobs.centers` each row was drawn around.
    // ANCHOR_END: blobs
    println!("blob centres:\n{:.2}", blobs.centers);

    // ANCHOR: classification
    let classes = MakeClassification::new()
        .with_n_classes(3)
        .with_n_informative(2)
        .with_class_sep(2.0)
        .with_class_weights(vec![0.6, 0.3, 0.1])
        .with_seed(7)
        .generate()?;
    // ANCHOR_END: classification
    let sizes: Vec<usize> = (0..3)
        .map(|class| classes.y.iter().filter(|&&y| y == class).count())
        .collect();
    println!(
        "class sizes (after 1% flips): {sizes:?}; feature means {:.2}",
        classes.x.mean_axis(Axis(0)).expect("rows")
    );
    Ok(())
}
