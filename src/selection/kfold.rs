//! K-fold splits: the rows cut into K folds, each the test side of one split.

use std::collections::BTreeMap;

use rand::seq::SliceRandom;

use super::{Classes, Split, Splitter};
use crate::error::{check_len, require};
use crate::{PredictError, Result, Seed};

/// The number of folds unless set, as in scikit-learn.
const DEFAULT_N_SPLITS: usize = 5;

/// K consecutive folds of rows; split `i` tests on fold `i` and trains on the
/// rest, so every row is tested exactly once.
///
/// When `n_samples` isn't a multiple of K, the first `n_samples % K` folds get
/// one row more. Rows are taken in order unless shuffled with
/// [`with_shuffle`](KFold::with_shuffle): shuffle when the rows are sorted
/// (by class, by date), or each fold sees a different slice of the data.
///
/// # Errors
///
/// `InvalidParameter` with `name = "n_splits"` unless `2 <= n_splits <=
/// n_samples`.
#[derive(Debug, Clone)]
pub struct KFold {
    n_splits: usize,
    shuffle: Option<Seed>,
}

impl KFold {
    /// 5 folds, rows in order.
    pub fn new() -> Self {
        KFold {
            n_splits: DEFAULT_N_SPLITS,
            shuffle: None,
        }
    }

    /// The number of folds, and so of splits: at least 2.
    pub fn with_n_splits(self, n_splits: usize) -> Self {
        KFold { n_splits, ..self }
    }

    /// Shuffle the rows with `seed` before cutting them into folds.
    pub fn with_shuffle(self, seed: Seed) -> Self {
        KFold {
            shuffle: Some(seed),
            ..self
        }
    }
}

impl Default for KFold {
    fn default() -> Self {
        Self::new()
    }
}

impl Splitter for KFold {
    type By<'a> = ();

    fn splits(&self, n_samples: usize, _by: ()) -> Result<Vec<Split>> {
        check_n_splits(self.n_splits, n_samples)?;
        let mut rows: Vec<usize> = (0..n_samples).collect();
        if let Some(seed) = self.shuffle {
            rows.shuffle(&mut seed.rng());
        }
        let (base, n_larger) = (n_samples / self.n_splits, n_samples % self.n_splits);
        let mut rest = &rows[..];
        let splits = (0..self.n_splits)
            .map(|fold| {
                let (test, after) = rest.split_at(base + usize::from(fold < n_larger));
                rest = after;
                Split::holding_out(n_samples, test.to_vec())
            })
            .collect();
        Ok(splits)
    }
}

/// [`KFold`] keeping the class proportions in every fold, reading
/// [`Classes`].
///
/// Each class's rows are dealt to the folds in turn, each class carrying on
/// from the fold where the last one stopped, so fold sizes differ by at most
/// one row. For classes `0 0 0 0 1 1 1 1` and 3 folds, the test sides are rows
/// `0 3 6`, `1 4 7` and `2 5`. (scikit-learn gives each class consecutive
/// blocks with the same counts instead.) With
/// [`with_shuffle`](StratifiedKFold::with_shuffle), each class's rows are
/// shuffled before dealing.
///
/// # Errors
///
/// Checked in this order:
///
/// - `ShapeMismatch` with `context = "classes"` unless there is one class per
///   row.
/// - `InvalidParameter` with `name = "n_splits"` unless `2 <= n_splits <=
///   n_samples`.
/// - `InvalidParameter` with `name = "classes"` if a class has fewer rows than
///   folds: some test side would lack it, so its precision or recall there
///   would have nothing to measure. (scikit-learn only warns.)
#[derive(Debug, Clone)]
pub struct StratifiedKFold {
    n_splits: usize,
    shuffle: Option<Seed>,
}

impl StratifiedKFold {
    /// 5 folds, rows in order.
    pub fn new() -> Self {
        StratifiedKFold {
            n_splits: DEFAULT_N_SPLITS,
            shuffle: None,
        }
    }

    /// The number of folds, and so of splits: at least 2.
    pub fn with_n_splits(self, n_splits: usize) -> Self {
        StratifiedKFold { n_splits, ..self }
    }

    /// Shuffle each class's rows with `seed` before dealing them to the folds.
    pub fn with_shuffle(self, seed: Seed) -> Self {
        StratifiedKFold {
            shuffle: Some(seed),
            ..self
        }
    }
}

impl Default for StratifiedKFold {
    fn default() -> Self {
        Self::new()
    }
}

impl Splitter for StratifiedKFold {
    type By<'a> = Classes<'a>;

    fn splits(&self, n_samples: usize, Classes(classes): Classes<'_>) -> Result<Vec<Split>> {
        check_len("classes", n_samples, classes.len())?;
        check_n_splits(self.n_splits, n_samples)?;

        let mut rows_by_class: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (row, &class) in classes.iter().enumerate() {
            rows_by_class.entry(class).or_default().push(row);
        }
        if let Some((class, rows)) = rows_by_class
            .iter()
            .find(|(_, rows)| rows.len() < self.n_splits)
        {
            return Err(PredictError::InvalidParameter {
                name: "classes",
                reason: format!(
                    "class {class} has {} rows, fewer than the {} folds",
                    rows.len(),
                    self.n_splits
                ),
            });
        }

        // One generator for every class, drawn in class order.
        let mut rng = self.shuffle.map(Seed::rng);
        let mut folds = vec![Vec::new(); self.n_splits];
        let mut fold = 0;
        for mut rows in rows_by_class.into_values() {
            if let Some(rng) = &mut rng {
                rows.shuffle(rng);
            }
            for row in rows {
                folds[fold].push(row);
                fold = (fold + 1) % self.n_splits;
            }
        }
        Ok(folds
            .into_iter()
            .map(|test| Split::holding_out(n_samples, test))
            .collect())
    }
}

/// The K-fold bounds on `n_splits`: at least 2, and no more than the rows.
fn check_n_splits(n_splits: usize, n_samples: usize) -> Result<()> {
    require(
        (2..=n_samples).contains(&n_splits),
        "n_splits",
        &format!("must be in 2..={n_samples}, got {n_splits}"),
    )
}
