# Contributing

Thanks for helping out! Bug reports, ideas and pull requests are welcome.

## Before you start

- **Bugs and small fixes:** a PR is welcome, ideally with a failing test.
- **Features and larger changes:** please open an issue first so we can agree
  on the approach before you spend time on it.
- Keep PRs small and focused, and justify every new dependency.
- By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Building

You need Rust 1.88 or newer. For the book, also
[mdBook](https://rust-lang.github.io/mdBook/) with its Mermaid and KaTeX
plugins:

```sh
cargo install mdbook mdbook-mermaid mdbook-katex
```

`cargo test --all-features` builds and tests everything;
`mdbook serve docs --open` previews the book with live reload.

## Checks

CI runs the following, and all must pass:

```sh
cargo fmt --all --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-features --locked
cargo +1.88 check --all-features --locked
RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo +nightly doc --no-deps --all-features --locked
mdbook build docs
cargo semver-checks check-release
cargo publish --dry-run --locked
cargo deny --locked check
```

Tests run on Linux, macOS and Windows. A workflow audit
([zizmor](https://docs.zizmor.sh/)) checks `.github/workflows/`.

## How the code is organised

- [`AGENTS.md`](AGENTS.md) summarises the architecture and the project's
  rules; the [book](https://predict.rs) explains them in full. If you use a
  coding agent, point it at `AGENTS.md`.
- Tests live in `tests/` and use only the public API.
- Code and documentation change together: a change to behaviour or naming
  updates the rustdoc and the book in the same PR. Code shown in the book
  lives in `examples/`, so CI compiles it.
- New dependencies need discussion in an issue first. The minimum Rust
  version is raised only in a minor release, with a reason.

## Commit messages and PR titles

Commits follow [Conventional Commits](https://www.conventionalcommits.org/),
since the changelog is generated from them: `feat:` for new behaviour, `fix:`
for bugs, and `docs`, `refactor`, `test`, `perf`, `ci` or `chore` for the
rest. Scope it with the module it touches, as in `feat(linear): add ridge
regression` or `docs(book): explain R²`. Mark breaking changes with `!`, as in
`feat(dataset)!: ...`.

PRs are squash-merged, so the **PR title** becomes the commit message, and CI
checks that it is a Conventional Commit.

## Releases

[release-plz](https://release-plz.dev/) keeps a release PR open that bumps
the version and adds the `CHANGELOG.md` entries generated from `feat`, `fix`
and `perf` commits. Merging it publishes to crates.io through trusted
publishing (no stored token), tags `vX.Y.Z` and creates the GitHub release.

## Who decides

This project is maintained by one person, who reviews every change and makes
the final call on direction. That's a matter of scale, not a closed door: if
you contribute regularly and want a bigger role, such as reviewing, owning a
part of the code or helping steer it, say so in an issue. This section will
change as the project gains maintainers.

## License

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this project by you shall be licensed under the
project's license (`MIT OR Apache-2.0`), as defined in the Apache-2.0
license, without any additional terms or conditions.
There is no CLA and no sign-off requirement.
