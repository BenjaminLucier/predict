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

## Pull requests

Every change lands through a pull request, squash-merged into one commit on
`main`. Each is one idea, compiles and passes the checks on its own, and
updates the book whenever it changes behaviour, naming or architecture.

The PR title becomes the commit subject, so it's a [Conventional
Commit](https://www.conventionalcommits.org/) scoped by the area it touches,
for example `feat(dataset): add Dataset` or `docs(book): explain R²`. The
description becomes the commit body: say why, then what.
