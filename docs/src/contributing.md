# Development

## Setup

You need Rust 1.88 or newer (edition 2024) and, for this book,
[mdBook](https://rust-lang.github.io/mdBook/) with its Mermaid plugin for
diagrams and its KaTeX plugin for maths:

```bash
cargo install mdbook mdbook-mermaid mdbook-katex
```

`cargo binstall` installs the same three as prebuilt binaries, if you have it.

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
