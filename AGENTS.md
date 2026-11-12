# AGENTS.md

Guidance for AI coding agents working in this repository. Read this first.

## What this project is

`predict` is a Rust crate for machine learning with a consistent,
scikit-learn-style API. It aims to be the fastest and easiest-to-use traditional
ML library on a single machine, written only in Rust. It is a library for
external use, so public API stability and semver matter.

The book in `docs/src/` (published at <https://predict.rs>) is the source of
truth for the architecture. This file summarises it. If the two disagree, the
book wins, and the disagreement is a bug: fix the book and the code together.

## Working agreement

The developer must understand every line in this repository well enough to have
written it from scratch. Speed of producing code is not the goal.

- **Explain before and after.** Before writing code, say what you'll build and
  why it comes next. Afterwards, explain how it works, its invariants, and how
  to verify it.
- **Teach the ML, not just the Rust.** Explain the algorithm and the maths: what
  problem it solves, why it works, how it fails, how it compares with
  scikit-learn.
- **Simple and legible first.** The first version of anything is the most
  obvious correct implementation. Optimisation comes later, explained when it's
  added.
- **Small steps.** Prefer many small, understandable diffs to one large one. If
  a change would be hard to follow, stop and split it or explain it first.
- **Show it working.** Pair each step with a test or a tiny runnable
  demonstration. Never claim something works without running it.
- **Surface decisions.** When there is a real design choice, present the options
  and trade-offs and let the developer decide.
- **Stay in scope.** Do exactly the agreed step, nothing neighbouring.

## Git

- Commits on `main` are small, each one idea, and each compiles and passes the
  checks below on its own.
- Commit or push only when the developer explicitly approves that change. An
  approval covers that one commit, not the next.
- Never force-push `main` or rewrite pushed history. If CI breaks, fix forward.
- Message: imperative subject under 72 characters, prefixed with the area
  (`dataset: add Dataset`), and a body saying what and why.

## Architecture

Grows as modules land; the book's Architecture pages have the detail. The crate
is flat: no `core` module (it would shadow Rust's built-in `core` crate at the
root); foundation modules sit at the top level with everyday items re-exported
at the root.

- **Float type** (`src/float.rs`). Everything holding feature values or
  regression targets is generic over `F: Float` (sealed; `f32` and `f64`, built
  on ndarray's `NdFloat`, plus `Into<f64>`), default `f64`, as the first type
  parameter. `ndarray` is the public array type; `faer` stays internal.
- **Dataset** (`src/dataset.rs`, book: `data.md`). `Dataset<'a, F = f64, L =
  DenseRowMajor<'a, F>, S = Unchecked, Y = NoTargets>`. `L` is the layout, a
  sealed `Layout<'a, F>` (`src/dataset/layout.rs`) held in a plain `features: L`
  field; its methods sit on a private supertrait, so outside code can bound on
  it but not call or implement them. Every layout is `Blocks<'a, F, O, D>`:
  column blocks side by side (private `Block` in `src/dataset/block.rs`:
  `DenseRowMajor` / `DenseColMajor` `CowArray`, `Csr`, `Csc`), never empty, each with `n_samples`
  rows, 1..n of them (count is runtime, never typed). `O: Orientation` is
  `Rows`, `Cols` or `MixedOrientation`; `D: Density` is `Dense`, `Sparse` or
  `MixedDensity` (sealed zero-sized markers, a promise like `S`). Aliases:
  `DenseRowMajor`, `DenseColMajor`, `SparseRowMajor`, `SparseColMajor`.
  Dense memory order is parsed once into `RowMajorMatrix` / `ColMajorMatrix`
  (`src/dataset/dense.rs`; `new` returns `Malformed`, `context = "x"`, unless
  each row / column is contiguous: unit stride along it, the other stride at
  least its length, one row or column always passes).
  `from_row_major` / `from_col_major` / `from_csr` / `from_csc` are
  infallible and make one block each, never copying; `new` takes `impl
  Into<CowArray>` (`&x` or a view borrows, `x` moves in) and panics with the
  `RowMajorMatrix` error. `hstack` appends a right-hand
  dataset's blocks (same rows, same `S`, `NoTargets`, no weights, else
  `ShapeMismatch` / `Conflict`), copying and converting nothing; each axis of
  the result is the sealed `Union` of both sides' markers (same stays,
  different is mixed), so the result type depends only on the input types.
  A third check refuses sparse blocks with different `Absent` on the two
  sides (`Conflict`), so a dataset has at most one meaning. Conversions
  (`src/dataset/convert.rs`): `into_dense_row_major` / `into_dense_col_major`
  / `into_csr` / `into_csc` / `into_csr_with(absent)` / `into_csc_with(absent)`
  take ownership and return one block of that kind, moving it when it already
  is one, otherwise allocating once (sparse: counting sort over lanes, blocks
  walked in their own order so indices arrive sorted). Values never change, so
  `S`, targets and weights are kept; to sparse, a value is left out iff
  `absent` stands for it. `with_absent` (only on `Unchecked`) relabels every
  sparse block without copying.
  The left's targets and weights are kept. `select_rows(&[usize])` (private
  helper `src/dataset/select.rs`) returns the chosen rows, in order, repeats
  allowed, as `Dataset<'_, F, Blocks<'_, ..>, S, Y::Borrowed<'_>>`: every block
  keeps its kind; one ascending range is borrowed (dense slices, targets and
  weights; CSR borrows indices and values, rebuilds offsets), anything else is
  copied once; CSC always copies (counting sort of chosen rows, per-column
  sort unless ascending without repeats). Out-of-range rows panic.
  `TargetState::select_rows` is `#[doc(hidden)]`. `Blocks::append`
  `debug_assert!`s matching rows. Transformers return owned data. The only read
  of the features is `blocks()`: borrowed `BlockView`s (public,
  `#[non_exhaustive]`: `ArrayView2` or `&CsrMatrix` / `&CscMatrix`), never a
  copy; there is no `to_dense` (it would densify sparse data out of sight).
  Tests read values through `tests/common` (`dense`, `only_dense`).
  Validation errors give the column across the whole dataset. `S` is a sealed `State`: `Unchecked`,
  `NoInfinity` (NaN allowed), `Finite`. `validate_no_infinity` /
  `validate_finite` scan once and move up; states never move down, and only the
  private `from_parts` assembles a dataset in a given state. `InfinityFree`
  bounds inputs that accept NaN but not infinity; `S::WithoutInfinity` is the
  output state of a transformer that checks what it writes for infinity.
- **Targets** (`src/dataset/targets.rs`, book: `data.md`). `Y` is a sealed
  `TargetState`: `NoTargets` or `Targets<'a, T, D = Ix1>`. `T: Target` is sealed:
  `f32` / `f64` (must be finite) and `usize` (class indices). `D: TargetDim` is
  `Ix1` or `Ix2` (rows × outputs). `with_y` checks rows, then outputs, then
  values, once; `S` describes only the features. Features and targets share one
  lifetime. Fields use `CovariantCowArray` (`CowArray` with its element type
  spelled out), so `Dataset` and `Targets` are covariant over `'a` despite
  ndarray 0.17's projection default; `datasets_shorten_like_references` guards
  it.
- **Sample weights** (`src/dataset.rs`, book: `data.md`). A field,
  `Option<CovariantCowArray<'a, F, Ix1>>`, not a type parameter: every
  estimator supports weights. `None` means every row weighs 1 (estimators take
  the unweighted fast path). `with_sample_weights` works in any state and
  target state, replaces earlier weights, and checks length, then each value
  (finite, non-negative) at the first bad row. A positive total is checked in
  `fit`, not here.

- **Sparse** (`src/dataset/csr.rs`, `csc.rs`, book: `data.md`). `CsrMatrix<'a,
  F>` (`n_cols`, `row_offsets`, `col_indices`, `values`) and `CscMatrix<'a, F>`
  (`n_rows`, `col_offsets`, `row_indices`, `values`) wrap one private
  `Compressed` storage (`src/dataset/compressed.rs`: lanes of `Cow` slices,
  `usize` indices; error names per wrapper). `new` checks structure only, in
  order: offsets (non-empty, start at 0, non-decreasing), lengths, then per
  lane indices `<` the lane length and strictly increasing (canonical form
  required). Validation scans lane by lane. Values are checked by `validate_*`.
  `Absent` (`#[non_exhaustive]`): `Zero` (default, scikit-learn) or `Missing`
  (XGBoost); with `Missing`, absent entries are NaN when converted to dense
  and in validation, so the matrix reaches `Finite` only if fully stored.

- **Estimator traits** (`src/traits.rs`, private, re-exported at the root;
  book: `estimators.md`). An estimator is a config type and a fitted type.
  `ValidateConfig` (default accepts) checks settings and every `fit` calls it
  first. `Fit<In>` (config, borrows, `type Fitted`), `Predict<In>` (fitted,
  borrows, generic over `Y` so it can't read targets), `PredictProbability<In>:
  Predict<In>` (`type Probabilities`, `n_classes`), `Transform<In>` (takes
  ownership, keeps rows, targets and weights), `FitTransform<In>: Fit<In,
  Fitted: Transform<In>>` (default body, explicit empty opt-in impl).
  `#[diagnostic::on_unimplemented]` on `Fit`, `Predict`, `Transform` names
  state, targets and layout; it fires only when an estimator has two or more
  impls (with one, rustc reports a type mismatch). Impls name the input's `L`,
  `S` and `Y`, one per layout an estimator handles well. `Dataset::view()`
  borrows everything (targets via `TargetState::Borrowed<'b>`), copying
  nothing, so a dataset can be transformed and kept. `from_parts`
  `debug_assert!`s targets and weights have one row per sample. The example
  estimator (`examples/estimator.rs`, nearest centroid) is not in the library.
- **Seed** (`src/seed.rs`, private, `Seed` re-exported at the root).
  `#[non_exhaustive] enum Seed { Entropy (default), Fixed(u64) }`; configs
  store one and set it with `with_seed(u64)`. Crate-private `rng()` gives a
  `rand` `Xoshiro256PlusPlus` (fast, the same on every platform, not
  cryptographic), never public. `rand` default features are off (no ChaCha);
  `rand_distr` gives `StandardNormal`. Same seed, same draws for the same
  `rand` and crate versions; not promised across upgrades. No `child(index)`
  yet: it lands with forests or selection.
- **Synthetic data** (`src/synthetic.rs` + `synthetic/`, book: `synthetic.md`).
  `MakeRegression`, `MakeBlobs`, `MakeClassification`: configs with `with_*`,
  checked by the crate-private `require` (`InvalidParameter`) when drawing.
  `generate()` returns `*Data` (public `x`, `y` and truth fields) with
  `dataset()`, which runs the public `validate_finite` scan (fields are
  public, so no assumed state). `stream()` returns `*Stream`: endless
  `Iterator<Item = (Array1<f64>, target)>`, `next_batch(n)` and `batches(n)`
  (panics on 0) return `Samples<T> { x, y }` with `dataset()`; truth getters.
  One private per-row `fill_row` per generator serves every path; batching
  never changes the rows. Regression's `generate` is the stream's first
  `n_samples` rows; the class generators' `generate` fixes class sizes
  (largest remainder, ties to the lower class) and shuffles, streams draw each
  class. Noise and flips are drawn even at 0, so a seed's features don't depend
  on `noise` / `flip_y`. Classification: one corner per class (distinct
  integers' bits, `n_classes <= 2^min(n_informative, 63)`), columns
  informative, redundant, repeated, useless, then `flip_y`; no clusters per
  class, covariance or column shuffling yet.
- **Errors** (`src/error.rs`, book: `errors.md`). One public enum,
  `PredictError` (`thiserror`, `#[non_exhaustive]`), and the alias `Result<T>`,
  both re-exported at the crate root from a private `error` module. No `Clone` /
  `PartialEq`; tests use `matches!`. A variant is added in the commit that first
  raises it. `InvalidValue` covers a value outside its input's range (negative
  weight); `Malformed` a broken structure (unsorted sparse row); `Conflict`
  two valid inputs that can't be combined (weights on both sides of
  `hstack`); `Undefined` a quantity with no value for the inputs (R² of
  constant targets, all-zero weights). `context` names an input (`"y"`), a
  transformer (`"StandardScaler"`) or a function or method
  (`"validate_finite"`, `"r2_score"`). Settings are checked in `fit`, never in
  `with_*`.

- **Metrics** (`src/metrics.rs`, `src/metrics/`, book: `metrics.md`). Free
  functions over `ArrayView1` `y_true` / `y_pred`: `mse`, `rmse`, `mae`,
  `r2_score` (inputs `F: Float`), `accuracy`, `precision`, `recall`,
  `fbeta_score`, `f1_score` (`usize` classes), each with a `_weighted` form
  taking `sample_weights` of its own float type `W` (checked like `Dataset`
  weights, total positive and finite). Inputs are checked first (lengths, empty,
  finite); every result is `f64`, summed in `f64`. Overflow gives ±infinity,
  never NaN; `Undefined` for a constant `y_true` in R². Classes are those in
  `y_true` or `y_pred`; a 0/0 per-class score is 0. `Average`
  (`#[non_exhaustive]`): `Binary { positive }`, `Macro`, `Micro`, `BySupport`
  (scikit-learn's `"weighted"`). `ConfusionMatrix` holds `usize` counts; a
  private `Tally` holds weights. `Metric<T, W>` (open, dyn-compatible:
  `direction()`, `compute(.., Option<weights>)`) with `Direction`, implemented
  by `MSE`, `RMSE`, `MAE`, `R2`, `Accuracy`, `Precision`, `Recall`, `F1`,
  `FBeta`.

## Conventions

- Rust edition 2024. Adding a dependency needs the developer's agreement.
- Explicit names: trait names say what they do and for which setting.
- Short idiomatic method names: `fit`, `predict`, `transform`.
- `x` / `y` in code, "targets" in prose. "Classes" for class indices, never
  "labels"; "labels" only for cluster assignments.
- Acronyms keep their capitals in type names (`KMeans`, `DBSCAN`).
- Public enums that may grow are `#[non_exhaustive]`.
- Don't add comments that restate code; comment intent and trade-offs.
- Tests go in `tests/` and use only the public API. A `#[cfg(test)] mod tests`
  in a source file is only for behaviour the public API can't reach yet; move
  it to `tests/` once it can.

## Documentation

- Docs and code change together: a change to behaviour, naming or architecture
  updates the book in the same commit.
- The book describes what's in the repository, not plans.
- `predict` in backticks reads as the method, so in prose call the library
  "this crate" (or "predict" in a heading or title), and write "the `predict`
  method" when you mean the method.
- Code shown in the book lives in `examples/` and is pulled in with
  `{{#include}}`, so `cargo check --all-targets` compiles it.
- Maths in the book: `$...$` inline and `$$...$$` for display, rendered at build
  time by mdbook-katex; diagrams in Mermaid. Rustdoc has no maths rendering, so
  doc comments use plain notation and link to the book for the derivation.
- Links: book pages use relative links; book to API docs uses
  `/api/predict/...`; rustdoc to book uses `https://predict.rs/...`.

## Commands

```bash
cargo check --all-targets --all-features    # fast compile check
cargo test --all-features
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
mdbook build docs
```

## Definition of done

1. It does what was agreed, no more.
2. Format, clippy, tests, docs and the book build all pass (CI runs the same),
   and you report honestly what you ran.
3. There is a minimal demonstration of it working.
4. The developer has had the explanation needed to understand it.
5. The book and this file are up to date.
