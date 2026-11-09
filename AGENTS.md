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
- **Dataset** (`src/dataset.rs`, book: `data.md`). `Dataset<'a, F = f64, S =
  Unchecked, Y = NoTargets>` wraps a `CowArray`: `new` takes `impl
  Into<CowArray>`, so `&x` or a view borrows and `x` moves in, never copying;
  transformers return owned data. `S` is a sealed `State`: `Unchecked`,
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

- **Errors** (`src/error.rs`, book: `errors.md`). One public enum,
  `PredictError` (`thiserror`, `#[non_exhaustive]`), and the alias `Result<T>`,
  both re-exported at the crate root from a private `error` module. No `Clone` /
  `PartialEq`; tests use `matches!`. A variant is added in the commit that first
  raises it. `context` names an input (`"y"`), a transformer (`"StandardScaler"`)
  or a method (`"validate_finite"`). Settings are checked in `fit`, never in
  `with_*`.

## Conventions

- Rust edition 2024. Adding a dependency needs the developer's agreement.
- Explicit names: trait names say what they do and for which setting.
- Short idiomatic method names: `fit`, `predict`, `transform`.
- `x` / `y` in code, "targets" in prose. "Classes" for class indices, never
  "labels"; "labels" only for cluster assignments.
- Acronyms keep their capitals in type names (`KMeans`, `DBSCAN`).
- Public enums that may grow are `#[non_exhaustive]`.
- Don't add comments that restate code; comment intent and trade-offs.

## Documentation

- Docs and code change together: a change to behaviour, naming or architecture
  updates the book in the same commit.
- The book describes what's in the repository, not plans.
- `predict` in backticks reads as the method, so in prose call the library
  "this crate" (or "predict" in a heading or title), and write "the `predict`
  method" when you mean the method.
- Code shown in the book lives in `examples/` and is pulled in with
  `{{#include}}`, so `cargo check --all-targets` compiles it.
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
