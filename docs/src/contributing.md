# Development

## Setup

You need a stable Rust toolchain (edition 2024) and, for this book,
[mdBook](https://rust-lang.github.io/mdBook/) and its Mermaid plugin for diagrams:

```bash
cargo install mdbook mdbook-mermaid
```

## Checks

Run these from the repository root. CI runs the same on every push to `main`.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo doc --no-deps --all-features
mdbook build docs
```

`mdbook serve docs --open` gives a live-reloading preview of the book.

## Commits

Each commit is one idea, compiles and passes the checks on its own, and updates
the book whenever it changes behaviour, naming or architecture. Subjects are
imperative, under 72 characters, and prefixed with the area they touch, for
example `dataset: add Dataset`.
