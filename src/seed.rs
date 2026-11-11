//! Where randomness comes from: [`Seed`].

use rand::SeedableRng;
use rand::rngs::Xoshiro256PlusPlus;

/// Where a generator, splitter or estimator gets its random numbers.
///
/// Configs that use randomness store a `Seed` and set it with `with_seed(42)`,
/// so callers rarely name this type. The default, [`Seed::Entropy`], draws a
/// fresh seed from the operating system, so every run differs, as with
/// scikit-learn's `random_state=None`.
///
/// With [`Seed::Fixed`], the same number gives the same random draws on any
/// machine, for the same versions of this crate and of `rand` (pin them with
/// a `Cargo.lock`): `rand` may change what its shuffles and distributions draw
/// in a minor release. Floating-point results computed from the draws are a
/// separate matter, and may differ in the last bits between machines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Seed {
    /// A fresh seed from the operating system: different on every run.
    #[default]
    Entropy,
    /// A fixed seed: the same number always gives the same draws.
    Fixed(u64),
}

impl Seed {
    /// The random number generator for this seed.
    ///
    /// Xoshiro256++: fast, statistically strong for simulation, and the same
    /// on every platform. It is not cryptographically secure, which nothing
    /// here needs. The generator stays private so it can be swapped without
    /// breaking the API.
    ///
    /// # Panics
    ///
    /// With [`Seed::Entropy`], if the operating system can't supply a seed.
    pub(crate) fn rng(self) -> Xoshiro256PlusPlus {
        match self {
            Seed::Entropy => rand::make_rng(),
            Seed::Fixed(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
}

// `rng` is crate-private, so its behaviour is tested here rather than in
// `tests/`; move these once a public generator can show it.
#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    fn first_draws(seed: Seed) -> [u64; 4] {
        let mut rng = seed.rng();
        std::array::from_fn(|_| rng.next_u64())
    }

    #[test]
    fn a_fixed_seed_repeats_its_draws() {
        assert_eq!(first_draws(Seed::Fixed(42)), first_draws(Seed::Fixed(42)));
    }

    #[test]
    fn different_fixed_seeds_draw_differently() {
        assert_ne!(first_draws(Seed::Fixed(1)), first_draws(Seed::Fixed(2)));
    }

    #[test]
    fn entropy_draws_differently_each_time() {
        assert_ne!(first_draws(Seed::Entropy), first_draws(Seed::Entropy));
    }
}
