//! Regression and classification metrics, weighted and not, and metrics as
//! values. Run with `cargo run --example metrics`.

use ndarray::{ArrayView1, array};
use predict::Result;
use predict::metrics::{self, Average, ConfusionMatrix, Direction, MSE, Metric, R2};

fn main() -> Result<()> {
    // ANCHOR: regression
    let y_true = array![3.0, -0.5, 2.0, 7.0];
    let y_pred = array![2.5, 0.0, 2.0, 8.0];

    let mse = metrics::mse(y_true.view(), y_pred.view())?; // 0.375
    let mae = metrics::mae(y_true.view(), y_pred.view())?; // 0.5
    let r2 = metrics::r2_score(y_true.view(), y_pred.view())?; // 0.949
    // ANCHOR_END: regression
    println!("mse {mse}, mae {mae}, r2 {r2:.3}");

    // ANCHOR: weighted
    // Row 0 counts twice, row 2 not at all.
    let weights = array![2.0, 1.0, 0.0, 1.0];
    let weighted = metrics::mse_weighted(y_true.view(), y_pred.view(), weights.view())?;
    // ANCHOR_END: weighted
    println!("weighted mse {weighted}");

    // ANCHOR: classification
    let y_true = array![0, 0, 1, 1, 2, 2, 2];
    let y_pred = array![0, 1, 1, 1, 2, 0, 2];

    let confusion = ConfusionMatrix::from_predictions(y_true.view(), y_pred.view())?;
    // [[1, 1, 0],
    //  [0, 2, 0],
    //  [1, 0, 2]]   rows: true class, columns: predicted class

    let accuracy = metrics::accuracy(y_true.view(), y_pred.view())?; // 5 / 7
    let f1 = metrics::f1_score(y_true.view(), y_pred.view(), Average::Macro)?; // 0.7
    // ANCHOR_END: classification
    println!(
        "{}\naccuracy {accuracy:.3}, macro F1 {f1:.3}",
        confusion.matrix()
    );

    // ANCHOR: metric
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.5, 2.0, 2.0];
    let no_weights: Option<ArrayView1<'_, f64>> = None;

    for (name, metric) in [("MSE", &MSE as &dyn Metric<f64, f64>), ("R2", &R2)] {
        let score = metric.compute(y_true.view(), y_pred.view(), no_weights)?;
        let better = match metric.direction() {
            Direction::HigherIsBetter => "higher",
            Direction::LowerIsBetter => "lower",
        };
        println!("{name} = {score:.3} ({better} is better)");
    }
    // ANCHOR_END: metric
    Ok(())
}
