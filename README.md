
[![CI](https://github.com/lovesh/lovesh/generalized_paillier/actions/workflows/test.yml/badge.svg)](https://github.com/lovesh/lovesh/generalized_paillier/actions/workflows/test.yml)
[![Apache-2](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/lovesh/lovesh/generalized_paillier/blob/master/LICENSE)
[![Dependencies](https://deps.rs/repo/github/lovesh/lovesh/generalized_paillier/status.svg)](https://deps.rs/repo/github/lovesh/lovesh/generalized_paillier)
[![crates.io](https://img.shields.io/crates/v/generalized-paillier)](https://crates.io/crates/generalized-paillier)
[![docs.rs](https://img.shields.io/docsrs/generalized-paillier)](https://docs.rs/generalized-paillier/latest/generalized_paillier/)

<!-- cargo-rdme start -->

# Paillier encryption and Damgard-Jurik encryption

- Paillier encryption from the paper [Public-Key Cryptosystems Based on Composite Degree Residuosity Classes](https://link.springer.com/content/pdf/10.1007/3-540-48910-X_16.pdf). Check [the module](./src/paillier_original.rs) for more docs.
- Generalization of Paillier encryption, called Damgard-Jurik scheme from the paper [A Generalization of Paillier’s Public-Key System with Applications to Electronic Voting](https://people.csail.mit.edu/rivest/voting/papers/DamgardJurikNielsen-AGeneralizationOfPailliersPublicKeySystemWithApplicationsToElectronicVoting.pdf).
Check [the module](./src/damgard_jurik.rs) for more docs

The code is generic over the prime size and expansion factor `S` (for Damgard-Jurik)

By default, it uses standard library and [rayon](https://github.com/rayon-rs/rayon) for parallelization.

For `no_std` support, build as

`cargo build --no-default-features`

and for wasm-32, build as

`cargo build --no-default-features --target wasm32-unknown-unknown`

<!-- cargo-rdme end -->
