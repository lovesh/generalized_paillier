//! Paillier encryption as described in Scheme 1 of the [Public-Key Cryptosystems Based on Composite Degree Residuosity Classes](https://link.springer.com/content/pdf/10.1007/3-540-48910-X_16.pdf)
//!
//! The generator `g` is chosen to be `1+n` and the decryption and ciphertext randomness extraction is performed using Chinese Remainder Theorem.
//!
//! `DecryptionKey`, `EncryptionKey` and `Ciphertext` are generic over the limbs required to represent different values,
//!  eg. `PRIME_LIMBS` are the limbs required to represent the prime numbers used in decryption key so for a 1024 bit prime
//! on a 64-bit platform, `PRIME_LIMBS` would be 16.
//!
//! For `DecryptionKey` and `EncryptionKey`, there exist `PreparedDecryptionKey` and `PreparedEncryptionKey`
//! which contain precomputations to speed up the operations.
//!
//! Implements addition and multiplication of encrypted messages
//!
//! TODO: Find a better way to enforce the relationship between sizes of prime, modulus and modulus square. Maybe a trait with associated constants

use crate::{
    error::PaillierError,
    util::{blum_prime, crt_combine, l},
};
use core::ops::{Mul, Sub};
use crypto_bigint::{
    modular::{MontyForm, MontyParams, SafeGcdInverter},
    subtle::ConstantTimeLess,
    Concat, Odd, PrecomputeInverter, RandomMod, Split, Uint,
};
use crypto_primes::{generate_prime_with_rng, generate_safe_prime_with_rng, is_prime_with_rng};
use rand_core::CryptoRngCore;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Key used to decrypt and extract randomness from ciphertext.
#[derive(Debug, Clone, PartialEq, Zeroize, ZeroizeOnDrop)]
pub struct DecryptionKey<const PRIME_LIMBS: usize> {
    pub p: Odd<Uint<PRIME_LIMBS>>,
    pub q: Odd<Uint<PRIME_LIMBS>>,
}

/// Key used to encrypt message. `MODULUS_LIMBS` are the limbs required to represent the modulus `n` and
/// `MODULUS_SQR_LIMBS` to represent square of the modulus `n^2`
#[derive(Debug, Clone, PartialEq)]
pub struct EncryptionKey<const MODULUS_LIMBS: usize, const MODULUS_SQR_LIMBS: usize> {
    /// Modulus `n`: product of prime numbers `p` and `q`
    pub n: Odd<Uint<MODULUS_LIMBS>>,
    /// Modulus squared, i.e. `n^2`
    pub n_sqr: Odd<Uint<MODULUS_SQR_LIMBS>>,
}

/// `MODULUS_LIMBS` are the limbs required to represent the modulus `n` and `MODULUS_SQR_LIMBS` to represent square of the modulus `n^2`
#[derive(Debug, Clone, PartialEq)]
pub struct Ciphertext<const MODULUS_LIMBS: usize, const MODULUS_SQR_LIMBS: usize>(
    Uint<MODULUS_SQR_LIMBS>,
);

// TODO: Montgomery params are not zeroized but they should be
/// Decryption key with precomputations to make decryption and extracting randomness faster.
#[derive(Debug, Clone, PartialEq, Zeroize, ZeroizeOnDrop)]
pub struct PreparedDecryptionKey<
    const PRIME_LIMBS: usize,
    const MODULUS_LIMBS: usize,
    const MODULUS_SQR_LIMBS: usize,
> {
    pub p: Odd<Uint<PRIME_LIMBS>>,
    pub q: Odd<Uint<PRIME_LIMBS>>,
    /// Montgomery params for reducing mod `p`
    #[zeroize(skip)]
    pub p_mtg: MontyParams<PRIME_LIMBS>,
    /// Montgomery params for reducing mod `p` when the value to reduce is of size `MODULUS_SQR_LIMBS`
    #[zeroize(skip)]
    pub p_mtg_1: MontyParams<MODULUS_SQR_LIMBS>,
    /// Montgomery params for reducing mod `q`
    #[zeroize(skip)]
    pub q_mtg: MontyParams<PRIME_LIMBS>,
    /// Montgomery params for reducing mod `q` when the value to reduce is of size `MODULUS_SQR_LIMBS`
    #[zeroize(skip)]
    pub q_mtg_1: MontyParams<MODULUS_SQR_LIMBS>,
    /// Montgomery params for reducing mod `p^2`
    #[zeroize(skip)]
    pub p_sqr_mtg: MontyParams<MODULUS_SQR_LIMBS>,
    /// Montgomery params for reducing mod `q^2`
    #[zeroize(skip)]
    pub q_sqr_mtg: MontyParams<MODULUS_SQR_LIMBS>,
    /// `p^-1 mod q`
    pub p_inv: MontyForm<PRIME_LIMBS>,
    /// `-q^-1 mod p`
    pub h_p: MontyForm<PRIME_LIMBS>,
    /// `-p^-1 mod q`
    pub h_q: MontyForm<PRIME_LIMBS>,
    /// `n^-1 mod p-1`
    pub n_inv_p: Uint<PRIME_LIMBS>,
    /// `n^-1 mod q-1`
    pub n_inv_q: Uint<PRIME_LIMBS>,
}

/// Encryption key with precomputations to make encryption faster
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedEncryptionKey<const MODULUS_LIMBS: usize, const MODULUS_SQR_LIMBS: usize> {
    /// Modulus `n=p*q`
    pub n: Odd<Uint<MODULUS_LIMBS>>,
    /// `n^2`
    pub n_sqr: Odd<Uint<MODULUS_SQR_LIMBS>>,
    /// Montgomery params for reducing mod `n`
    pub n_mtg_params: MontyParams<MODULUS_SQR_LIMBS>,
    /// Montgomery params for reducing mod `n^2`
    pub n_sqr_mtg_params: MontyParams<MODULUS_SQR_LIMBS>,
    /// `n mod n^2` in Montgomery form
    pub n_mtg: MontyForm<MODULUS_SQR_LIMBS>,
}

impl<const MODULUS_LIMBS: usize, const MODULUS_SQR_LIMBS: usize>
    From<EncryptionKey<MODULUS_LIMBS, MODULUS_SQR_LIMBS>>
    for PreparedEncryptionKey<MODULUS_LIMBS, MODULUS_SQR_LIMBS>
{
    fn from(key: EncryptionKey<MODULUS_LIMBS, MODULUS_SQR_LIMBS>) -> Self {
        let n_mtg = MontyParams::new_vartime(key.n.resize().to_odd().unwrap());
        let n_sqr_mtg = MontyParams::new_vartime(key.n_sqr);
        Self {
            n: key.n,
            n_sqr: key.n_sqr,
            n_mtg_params: n_mtg,
            n_sqr_mtg_params: n_sqr_mtg,
            n_mtg: MontyForm::new(&key.n.resize::<MODULUS_SQR_LIMBS>(), n_sqr_mtg),
        }
    }
}

impl<
        const PRIME_LIMBS: usize,
        const MODULUS_LIMBS: usize,
        const MODULUS_SQR_LIMBS: usize,
        const MODULO_QUAD_LIMBS: usize,
        const PRIME_UNSAT_LIMBS: usize,
        const MODULO_UNSAT_LIMBS: usize,
    > PreparedDecryptionKey<PRIME_LIMBS, MODULUS_LIMBS, MODULUS_SQR_LIMBS>
where
    Uint<PRIME_LIMBS>: Concat<Output = Uint<MODULUS_LIMBS>>,
    Uint<MODULUS_LIMBS>: Split<Output = Uint<PRIME_LIMBS>>,
    Uint<MODULUS_LIMBS>: Concat<Output = Uint<MODULUS_SQR_LIMBS>>,
    Uint<MODULUS_SQR_LIMBS>: Split<Output = Uint<MODULUS_LIMBS>>,
    Uint<MODULUS_SQR_LIMBS>: Concat<Output = Uint<MODULO_QUAD_LIMBS>>,
    Uint<MODULO_QUAD_LIMBS>: Split<Output = Uint<MODULUS_SQR_LIMBS>>,
    Odd<Uint<PRIME_LIMBS>>:
        PrecomputeInverter<Inverter = SafeGcdInverter<PRIME_LIMBS, PRIME_UNSAT_LIMBS>>,
    Odd<Uint<MODULUS_LIMBS>>:
        PrecomputeInverter<Inverter = SafeGcdInverter<MODULUS_LIMBS, MODULO_UNSAT_LIMBS>>,
{
    fn new(
        dk: DecryptionKey<PRIME_LIMBS>,
        ek: &EncryptionKey<MODULUS_LIMBS, MODULUS_SQR_LIMBS>,
    ) -> Self {
        let p_mtg = MontyParams::new(dk.p);
        let q_mtg = MontyParams::new(dk.q);
        let p_sqr = dk.p.square();
        let q_sqr = dk.q.square();
        // `unwrap` is fine since p and q are prime
        let p_inv = MontyForm::new(&dk.p.inv_odd_mod(&dk.q).unwrap(), q_mtg);

        // g = 1+n and as per the paper, h_p = L_p(g^(p-1) mod p^2)^-1 mod p where L_p(x) = (x-1)/p
        // g^(p-1) mod p^2 = (1+n)^(p-1) mod p^2 (from binomial expansion) = 1 + (p-1)n mod p^2 = 1 - n mod p^2
        // L_p(g^(p-1) mod p^2) mod p = ((1 - n) - 1)/p mod p = -q mod p.
        // Thus, h_p = -q^-1 mod p. Similarly, h_q = -p^-1 mod q
        let h_p = dk.q.inv_odd_mod(&dk.p).unwrap().neg_mod(&dk.p);
        let h_q = p_inv.neg();

        // n_inv_p = n^-1 mod p-1 and n_inv_q = n^-1 mod q-1.
        // unwrap is fine since gcd(n, p-1) = 1 and gcd(n, q-1) = 1
        let n_inv_p =
            ek.n.inv_mod(&dk.p.wrapping_sub(&Uint::ONE).resize().to_nz().unwrap())
                .unwrap()
                .resize();
        let n_inv_q =
            ek.n.inv_mod(&dk.q.wrapping_sub(&Uint::ONE).resize().to_nz().unwrap())
                .unwrap()
                .resize();

        Self {
            p: dk.p,
            q: dk.q,
            p_mtg,
            p_mtg_1: MontyParams::new(dk.p.resize().to_odd().unwrap()),
            q_mtg,
            q_mtg_1: MontyParams::new(dk.q.resize().to_odd().unwrap()),
            p_sqr_mtg: MontyParams::new(p_sqr.resize::<MODULUS_SQR_LIMBS>().to_odd().unwrap()),
            q_sqr_mtg: MontyParams::new(q_sqr.resize::<MODULUS_SQR_LIMBS>().to_odd().unwrap()),
            p_inv,
            h_p: MontyForm::new(&h_p, p_mtg),
            h_q,
            n_inv_p,
            n_inv_q,
        }
    }
}

impl<const PRIME_LIMBS: usize> DecryptionKey<PRIME_LIMBS> {
    pub fn new<R: CryptoRngCore>(rng: &mut R) -> Self {
        let p: Uint<PRIME_LIMBS> = generate_prime_with_rng(rng, Uint::<PRIME_LIMBS>::BITS);
        let q: Uint<PRIME_LIMBS> = generate_prime_with_rng(rng, Uint::<PRIME_LIMBS>::BITS);
        Self {
            p: p.to_odd().unwrap(),
            q: q.to_odd().unwrap(),
        }
    }

    pub fn new_with_paillier_blum_primes<R: CryptoRngCore>(rng: &mut R) -> Self {
        let p = blum_prime::<R, PRIME_LIMBS>(rng);
        let q = blum_prime::<R, PRIME_LIMBS>(rng);
        Self {
            p: p.to_odd().unwrap(),
            q: q.to_odd().unwrap(),
        }
    }

    pub fn new_with_safe_primes<R: CryptoRngCore>(rng: &mut R) -> Self {
        let p: Uint<PRIME_LIMBS> = generate_safe_prime_with_rng(rng, Uint::<PRIME_LIMBS>::BITS);
        let q: Uint<PRIME_LIMBS> = generate_safe_prime_with_rng(rng, Uint::<PRIME_LIMBS>::BITS);
        Self {
            p: p.to_odd().unwrap(),
            q: q.to_odd().unwrap(),
        }
    }

    pub fn from_primes(p: Odd<Uint<PRIME_LIMBS>>, q: Odd<Uint<PRIME_LIMBS>>) -> Self {
        Self { p, q }
    }

    pub fn is_valid<R: CryptoRngCore>(&self, rng: &mut R) -> bool {
        is_prime_with_rng(rng, self.p.as_ref()) && is_prime_with_rng(rng, self.q.as_ref())
    }
}

impl<const MODULUS_LIMBS: usize, const MODULUS_SQR_LIMBS: usize>
    EncryptionKey<MODULUS_LIMBS, MODULUS_SQR_LIMBS>
where
    Uint<MODULUS_LIMBS>: Concat<Output = Uint<MODULUS_SQR_LIMBS>>,
{
    const CHECK_MOD_SQR_LIMBS: () = assert!((2 * MODULUS_LIMBS) == MODULUS_SQR_LIMBS);

    pub fn new<const PRIME_LIMBS: usize>(dk: &DecryptionKey<PRIME_LIMBS>) -> Self
    where
        Uint<PRIME_LIMBS>: Concat<Output = Uint<MODULUS_LIMBS>>,
    {
        let _ = Self::CHECK_MOD_SQR_LIMBS;
        const { assert!(2 * PRIME_LIMBS == MODULUS_LIMBS) };

        let n: Uint<MODULUS_LIMBS> = dk.p.widening_mul(&dk.q).into();
        let n_sqr = n.square();
        Self {
            n: n.to_odd().unwrap(), // unwrap is fine since n is odd since its product of 2 odds
            n_sqr: n_sqr.to_odd().unwrap(),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.n_sqr == self.n.square().to_odd().unwrap()
    }
}

impl<const MODULUS_LIMBS: usize, const MODULUS_SQR_LIMBS: usize>
    Ciphertext<MODULUS_LIMBS, MODULUS_SQR_LIMBS>
{
    const CHECK_MOD_SQR_LIMBS: () = assert!((2 * MODULUS_LIMBS) == MODULUS_SQR_LIMBS);

    /// Encrypt the given message. Generates randomness internally
    pub fn new<R: CryptoRngCore>(
        rng: &mut R,
        msg: &Uint<MODULUS_LIMBS>,
        ek: impl Into<PreparedEncryptionKey<MODULUS_LIMBS, MODULUS_SQR_LIMBS>>,
    ) -> Result<Self, PaillierError>
    where
        Uint<MODULUS_LIMBS>: Concat<Output = Uint<MODULUS_SQR_LIMBS>>,
    {
        let ek = ek.into();
        let r = Uint::random_mod(rng, ek.n.as_nz_ref());
        Self::new_given_randomness_and_prepared_key(msg, r, ek)
    }

    /// Encrypt the given message with given randomness
    pub fn new_given_randomness(
        msg: &Uint<MODULUS_LIMBS>,
        r: Uint<MODULUS_LIMBS>,
        ek: impl Into<PreparedEncryptionKey<MODULUS_LIMBS, MODULUS_SQR_LIMBS>>,
    ) -> Result<Self, PaillierError>
    where
        Uint<MODULUS_LIMBS>: Concat<Output = Uint<MODULUS_SQR_LIMBS>>,
    {
        let ek = ek.into();
        Self::new_given_randomness_and_prepared_key(msg, r, ek)
    }

    pub fn new_given_randomness_and_prepared_key(
        msg: &Uint<MODULUS_LIMBS>,
        r: Uint<MODULUS_LIMBS>,
        ek: PreparedEncryptionKey<MODULUS_LIMBS, MODULUS_SQR_LIMBS>,
    ) -> Result<Self, PaillierError>
    where
        Uint<MODULUS_LIMBS>: Concat<Output = Uint<MODULUS_SQR_LIMBS>>,
    {
        let _ = Self::CHECK_MOD_SQR_LIMBS;

        if !bool::from(msg.ct_lt(&ek.n)) {
            return Err(PaillierError::MessageOutOfBound);
        }
        // `r` should also be coprime to `n` but that's going to be true except with a negligible probability and when it
        // happens, `n` has been factored.
        if !bool::from(r.ct_lt(&ek.n)) {
            return Err(PaillierError::RandomnessOutOfBound);
        }

        // g = 1 + n
        // g^m = (1 + n)^m mod n^2 = 1 + n*m mod n^2 from binomial expansion
        // g_m = 1 + n*m mod n^2

        let (g_m, r_n) = join!(
            {
                let n_m = ek.n_mtg * MontyForm::new(&msg.resize(), ek.n_sqr_mtg_params);
                n_m + MontyForm::one(ek.n_sqr_mtg_params)
            },
            {
                let r = r.resize::<MODULUS_SQR_LIMBS>();
                MontyForm::new(&r, ek.n_sqr_mtg_params).pow(&ek.n)
            }
        );

        let c = g_m * r_n;
        Ok(Self(c.retrieve()))
    }

    /// Decrypt the ciphertext and returns the message. Assumes that the ciphertext is valid, i.e. in `[0, n^2)`
    /// Uses CRT as described in section 7 of the paper
    pub fn decrypt<const PRIME_LIMBS: usize>(
        &self,
        dk: impl Into<PreparedDecryptionKey<PRIME_LIMBS, MODULUS_LIMBS, MODULUS_SQR_LIMBS>>,
    ) -> Uint<MODULUS_LIMBS>
    where
        Uint<PRIME_LIMBS>: Concat<Output = Uint<MODULUS_LIMBS>>,
    {
        let _ = Self::CHECK_MOD_SQR_LIMBS;
        const { assert!(2 * PRIME_LIMBS == MODULUS_LIMBS) };

        let dk = dk.into();

        // m_p = m mod p and m_q = m mod q
        let (m_p, m_q) = join!(
            self.message_mod_prime(dk.p, dk.p_mtg, dk.p_sqr_mtg, dk.h_p),
            self.message_mod_prime(dk.q, dk.q_mtg, dk.q_sqr_mtg, dk.h_q)
        );

        crt_combine(&m_p, &m_q, dk.p_inv, &dk.p, dk.q_mtg)
    }

    /// Get randomness used in the ciphertext using CRT
    pub fn get_randomness<const PRIME_LIMBS: usize>(
        &self,
        msg: &Uint<MODULUS_LIMBS>,
        dk: impl Into<PreparedDecryptionKey<PRIME_LIMBS, MODULUS_LIMBS, MODULUS_SQR_LIMBS>>,
        ek: impl Into<PreparedEncryptionKey<MODULUS_LIMBS, MODULUS_SQR_LIMBS>>,
    ) -> Result<Uint<MODULUS_LIMBS>, PaillierError>
    where
        Uint<PRIME_LIMBS>: Concat<Output = Uint<MODULUS_LIMBS>>,
    {
        let _ = Self::CHECK_MOD_SQR_LIMBS;
        const { assert!(2 * PRIME_LIMBS == MODULUS_LIMBS) };

        let dk = dk.into();
        let ek = ek.into();

        if !bool::from(msg.ct_lt(&ek.n)) {
            return Err(PaillierError::MessageOutOfBound);
        }

        // From ciphertext g^m*r^n mod n^2, first get r^n mod n^2
        // g = 1 + n
        // g^{-m} = (1 + n)^-m mod n^2 = 1 - n*m mod n^2 from binomial expansion
        let mn = ek.n_mtg * MontyForm::new(&msg.resize(), ek.n_sqr_mtg_params);
        let g_m_inv = MontyForm::one(ek.n_sqr_mtg_params) - mn;

        // c_z = (1+n)^m.r^n mod n^2 . (1 + n)^-m mod n^2 = r^n mod n^2
        let c_z = MontyForm::new(&self.0, ek.n_sqr_mtg_params) * g_m_inv;

        // As r was chosen in Z*_n, we need r mod n from r^n mod n^2.
        // The following idea of computing r^n mod n^2 using CRT was taken from `open` function in rust-pailler repo https://github.com/mortendahl/rust-paillier/blob/master/src/core.rs
        // r^n mod n^2 mod p = r^n mod p since p is a multiple of n^2. Similarly, r^n mod n^2 mod q = r^n mod q since q is a multiple of n^2
        // To get r mod p from r^n mod p, raise it to n^-1. r^{n * n^-1} mod p = r^{n * n^-1 mod p-1} mod p (from Fermat's theorem).
        // Similarly, r^{n * n^-1} mod q = r^{n * n^-1 mod q-1} mod p

        let (r_p, r_q) = join!(
            {
                let c_p = MontyForm::new(&c_z.retrieve(), dk.p_mtg_1)
                    .retrieve()
                    .resize();
                let c_p = MontyForm::new(&c_p, dk.p_mtg);
                c_p.pow(&dk.n_inv_p).retrieve()
            },
            {
                let c_q = MontyForm::new(&c_z.retrieve(), dk.q_mtg_1)
                    .retrieve()
                    .resize();
                let c_q = MontyForm::new(&c_q, dk.q_mtg);
                c_q.pow(&dk.n_inv_q).retrieve()
            }
        );

        let r = crt_combine(&r_p, &r_q, dk.p_inv, &dk.p, dk.q_mtg);

        Ok(r)
    }

    /// Combine another ciphertext with this ciphertext such that the resulting ciphertext
    /// encrypts the sum of the 2 messages
    pub fn add(
        &self,
        rhs: &Self,
        ek: impl Into<PreparedEncryptionKey<MODULUS_LIMBS, MODULUS_SQR_LIMBS>>,
    ) -> Result<Self, PaillierError> {
        let _ = Self::CHECK_MOD_SQR_LIMBS;

        let ek = ek.into();
        if !rhs.is_valid(&ek) {
            return Err(PaillierError::CiphertextOutOfBound);
        }
        let l = MontyForm::new(&self.0, ek.n_sqr_mtg_params);
        let r = MontyForm::new(&rhs.0, ek.n_sqr_mtg_params);
        Ok(Self((l * r).retrieve()))
    }

    /// Return an updated ciphertext which encrypts the product of current encrypted message and
    /// the given message
    pub fn mul(
        &self,
        msg: &Uint<MODULUS_LIMBS>,
        ek: impl Into<PreparedEncryptionKey<MODULUS_LIMBS, MODULUS_SQR_LIMBS>>,
    ) -> Result<Self, PaillierError> {
        let _ = Self::CHECK_MOD_SQR_LIMBS;

        let ek = ek.into();
        if !bool::from(msg.ct_lt(&ek.n)) {
            return Err(PaillierError::MessageOutOfBound);
        }
        let l = MontyForm::new(&self.0, ek.n_sqr_mtg_params);
        Ok(Self(l.pow(msg).retrieve()))
    }

    /// Returns true if ciphertext is valid, i.e. in `[0, n^2)`
    pub fn is_valid(&self, ek: &PreparedEncryptionKey<MODULUS_LIMBS, MODULUS_SQR_LIMBS>) -> bool {
        self.0.lt(ek.n_sqr.as_ref())
    }

    /// Used in decryption with CRT. Returns `message mod prime`
    fn message_mod_prime<const PRIME_LIMBS: usize>(
        &self,
        prime: Odd<Uint<PRIME_LIMBS>>,
        prime_mtg_params: MontyParams<PRIME_LIMBS>,
        prime_sqr_mtg_params: MontyParams<MODULUS_SQR_LIMBS>,
        h: MontyForm<PRIME_LIMBS>,
    ) -> Uint<PRIME_LIMBS>
    where
        Uint<PRIME_LIMBS>: Concat<Output = Uint<MODULUS_LIMBS>>,
    {
        // works because for ciphertext c, message m and prime p:
        // m_p = L_p(c^{p-1} mod p^2).h_p mod p where L_p(x) = (x-1)/p and h_p = -q^-1 mod p and n = p*q
        // c = (1+n)^m*r^n mod p^2 and c^{p-1} mod p^2 = (1+n)^{m*(p-1)}*r^{n*(p-1)} mod p^2.
        // r^{n*(p-1)} mod p^2 = r^{q*p*(p-1)} mod p^2 = 1 mod p^2 since r^{p*(p-1)} mod p^2 = 1
        // (1+n)^{m*(p-1)} mod p^2 = 1 + n*m*(p-1) mod p^2 from binomial expansion
        // L_p(c^{p-1} mod p^2) = (1 + n*m*(p-1) mod p^2 - 1)/p = (1 - n*m mod p^2 - 1)/p = -m*q
        // m_p = L_p(c^{p-1} mod p^2).h_p mod p = -m*q * -q^-1 mod p = m mod p

        let c = MontyForm::new(&self.0, prime_sqr_mtg_params);
        // c^(p-1) mod p^2
        let m = c.pow(&prime.sub(Uint::ONE)).retrieve();
        // (c^(p-1) mod p^2 - 1)/p
        let m = l::<MODULUS_LIMBS, PRIME_LIMBS, PRIME_LIMBS>(
            &m.resize::<MODULUS_LIMBS>(),
            prime.as_nz_ref(),
        );
        h.mul(MontyForm::new(&m, prime_mtg_params)).retrieve()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::{get_1024_bit_primes, get_2048_bit_primes, timing_info};
    use core::ops::Add;
    use crypto_bigint::{U1024, U128, U2048, U256, U4096, U512, U64, U8192};
    use rand::rngs::OsRng;
    use std::time::Instant;

    macro_rules! check_enc_dec_given_dec_key {
        ( $num_iters: ident, $prime_type:ident, $modulo_type:ident, $modulo_sqr_type: ident, $dk: ident ) => {
            let mut rng = OsRng::default();
            println!("Running {} iterations for {} bits prime", $num_iters, $prime_type::BITS);
            let ek = EncryptionKey::<{ $modulo_type::LIMBS }, { $modulo_sqr_type::LIMBS }>::new(&$dk);
            let pek: PreparedEncryptionKey<{ $modulo_type::LIMBS }, { $modulo_sqr_type::LIMBS }> = ek.clone().into();
            let pdk = PreparedDecryptionKey::<{ $prime_type::LIMBS }, { $modulo_type::LIMBS }, { $modulo_sqr_type::LIMBS }>::new(
                $dk.clone(),
                &ek,
            );

            let mut enc_times = vec![];
            let mut dec_times = vec![];
            let mut rnd_times = vec![];

            for _ in 0..$num_iters {
                let m = $modulo_type::random_mod(&mut rng, ek.n.as_nz_ref());
                let start = Instant::now();
                let ct = Ciphertext::<{ $modulo_type::LIMBS }, { $modulo_sqr_type::LIMBS }>::new(&mut rng, &m, pek.clone()).unwrap();
                enc_times.push(start.elapsed());

                let start = Instant::now();
                let m_ = ct.decrypt(pdk.clone());
                dec_times.push(start.elapsed());
                assert_eq!(m, m_);

                // m_p and m_q correspond to m mod p and m mod q respectively
                let m_p = ct.message_mod_prime(pdk.p, pdk.p_mtg, pdk.p_sqr_mtg, pdk.h_p);
                assert_eq!(m_p, m.rem(&$dk.p.resize().to_nz().unwrap()).resize());

                let m_q = ct.message_mod_prime(pdk.q, pdk.q_mtg, pdk.q_sqr_mtg, pdk.h_q);
                assert_eq!(m_q, m.rem(&$dk.q.resize().to_nz().unwrap()).resize());

                let r = $modulo_type::random_mod(&mut rng, ek.n.as_nz_ref());
                let ct = Ciphertext::<{ $modulo_type::LIMBS }, { $modulo_sqr_type::LIMBS }>::new_given_randomness(
                    &m,
                    r.clone(),
                    pek.clone(),
                ).unwrap();
                let m_ = ct.decrypt(pdk.clone());
                assert_eq!(m, m_);

                let start = Instant::now();
                let r_ = ct.get_randomness::<{ $prime_type::LIMBS }>(&m, pdk.clone(), pek.clone()).unwrap();
                rnd_times.push(start.elapsed());
                assert_eq!(r, r_);
            }

            // Should error on message >= n
            assert!(Ciphertext::<{ $modulo_type::LIMBS }, { $modulo_sqr_type::LIMBS }>::new(&mut rng, ek.n.as_ref(), pek.clone()).is_err());
            assert!(Ciphertext::<{ $modulo_type::LIMBS }, { $modulo_sqr_type::LIMBS }>::new(&mut rng, &ek.n.get().add(Uint::ONE), pek.clone()).is_err());

            // Should error on randomness >= n
            let m = ek.n.sub(Uint::ONE);   // m = n-1
            assert!(Ciphertext::<{ $modulo_type::LIMBS }, { $modulo_sqr_type::LIMBS }>::new_given_randomness_and_prepared_key(&m, ek.n.get(), pek.clone()).is_err());
            assert!(Ciphertext::<{ $modulo_type::LIMBS }, { $modulo_sqr_type::LIMBS }>::new_given_randomness_and_prepared_key(&m, ek.n.get().add(Uint::ONE), pek.clone()).is_err());

            let r = $modulo_type::random_mod(&mut rng, ek.n.as_nz_ref());
            let ct = Ciphertext::<{ $modulo_type::LIMBS }, { $modulo_sqr_type::LIMBS }>::new_given_randomness_and_prepared_key(&m, r, pek.clone()).unwrap();

            // Should error on message >= n
            assert!(ct.get_randomness::<{ $prime_type::LIMBS }>(ek.n.as_nz_ref(), pdk.clone(), pek.clone()).is_err());
            assert!(ct.get_randomness::<{ $prime_type::LIMBS }>(&ek.n.get().add(Uint::ONE), pdk.clone(), pek.clone()).is_err());

            // With valid message, it works
            let r_ = ct.get_randomness::<{ $prime_type::LIMBS }>(&m, pdk.clone(), pek.clone()).unwrap();
            assert_eq!(r, r_);

            println!("Enc time: {:?}", timing_info(enc_times));
            println!("Dec time: {:?}", timing_info(dec_times));
            println!("Time to get encryption randomness: {:?}", timing_info(rnd_times));
        }
    }

    macro_rules! check_ops_given_dec_key {
        ( $num_iters: ident, $prime_type:ident, $modulo_type:ident, $modulo_sqr_type: ident, $dk: ident ) => {
            let mut rng = OsRng::default();
            println!("Running {} iterations for {} bits prime", $num_iters, $prime_type::BITS);
            let ek = EncryptionKey::<{ $modulo_type::LIMBS }, { $modulo_sqr_type::LIMBS }>::new(&$dk);
            let pek: PreparedEncryptionKey<{ $modulo_type::LIMBS }, { $modulo_sqr_type::LIMBS }> = ek.clone().into();
            let pdk = PreparedDecryptionKey::<{$prime_type::LIMBS }, { $modulo_type::LIMBS }, { $modulo_sqr_type::LIMBS }>::new(
                $dk.clone(),
                &ek,
            );
            let n_mtg = MontyParams::new_vartime(ek.n.to_odd().unwrap());

            let mut sum_times = vec![];
            let mut product_times = vec![];

            for _ in 0..$num_iters {
                let m1 = $modulo_type::random_mod(&mut rng, ek.n.as_nz_ref());
                let r1 = $modulo_type::random_mod(&mut rng, ek.n.as_nz_ref());
                let m2 = $modulo_type::random_mod(&mut rng, ek.n.as_nz_ref());
                let r2 = $modulo_type::random_mod(&mut rng, ek.n.as_nz_ref());

                let ct1 = Ciphertext::<{ $modulo_type::LIMBS }, { $modulo_sqr_type::LIMBS }>::new_given_randomness(
                    &m1,
                    r1,
                    pek.clone(),
                ).unwrap();
                let ct2 = Ciphertext::<{ $modulo_type::LIMBS }, { $modulo_sqr_type::LIMBS }>::new_given_randomness(
                    &m2,
                    r2,
                    pek.clone(),
                ).unwrap();

                let start = Instant::now();
                let ct_sum = ct1.add(&ct2, pek.clone()).unwrap();
                sum_times.push(start.elapsed());
                let expected_msg = m1.add_mod(&m2, &ek.n);
                let m_ = ct_sum.decrypt(pdk.clone());
                assert_eq!(expected_msg, m_);

                let expected_r = r1.mul_mod(&r2, ek.n.as_nz_ref());
                let r_ = ct_sum.get_randomness::<{$prime_type::LIMBS }>(&m_, pdk.clone(), pek.clone()).unwrap();
                assert_eq!(expected_r, r_);

                let start = Instant::now();
                let ct_prod = ct1.mul(&m2, pek.clone()).unwrap();
                product_times.push(start.elapsed());
                let expected_msg = m1.mul_mod(&m2, ek.n.as_nz_ref());
                let m_ = ct_prod.decrypt(pdk.clone());
                assert_eq!(expected_msg, m_);

                let expected_r = MontyForm::new(&r1, n_mtg).pow(&m2).retrieve();
                let r_ = ct_prod.get_randomness::<{$prime_type::LIMBS }>(&m_, pdk.clone(), pek.clone()).unwrap();
                assert_eq!(expected_r, r_);
            }

            println!("Sum time: {:?}", timing_info(sum_times));
            println!("Product time: {:?}", timing_info(product_times));
        }
    }

    #[test]
    fn encrypt_decrypt() {
        let num_iters = 100;

        macro_rules! check {
            ( $prime_type:ident, $modulo_type:ident, $modulo_sqr_type: ident ) => {
                let mut rng = OsRng::default();
                let dk = DecryptionKey::<{ $prime_type::LIMBS }>::new(&mut rng);
                check_enc_dec_given_dec_key!(
                    num_iters,
                    $prime_type,
                    $modulo_type,
                    $modulo_sqr_type,
                    dk
                );
            };
        }

        check!(U64, U128, U256);
        check!(U128, U256, U512);
    }

    #[test]
    fn ciphertext_add_mul() {
        let num_iters = 100;

        macro_rules! check {
            ( $prime_type:ident, $modulo_type:ident, $modulo_sqr_type: ident ) => {
                let mut rng = OsRng::default();
                let dk = DecryptionKey::<{ $prime_type::LIMBS }>::new(&mut rng);
                check_ops_given_dec_key!(
                    num_iters,
                    $prime_type,
                    $modulo_type,
                    $modulo_sqr_type,
                    dk
                );
            };
        }

        check!(U64, U128, U256);
        check!(U128, U256, U512);
    }

    #[test]
    fn with_1024_bit_prime() {
        // Check encryption, decryption, encrypted message's addition and multiplication with 1024-bit prime
        let num_iters = 10;
        let (p, q) = get_1024_bit_primes();
        let dk = DecryptionKey::from_primes(p, q);
        check_enc_dec_given_dec_key!(num_iters, U1024, U2048, U4096, dk);
        check_ops_given_dec_key!(num_iters, U1024, U2048, U4096, dk);
    }

    #[test]
    fn with_2048_bit_prime() {
        // Check encryption, decryption, encrypted message's addition and multiplication with 2048-bit prime
        let num_iters = 10;
        let (p, q) = get_2048_bit_primes();
        let dk = DecryptionKey::from_primes(p, q);
        check_enc_dec_given_dec_key!(num_iters, U2048, U4096, U8192, dk);
        check_ops_given_dec_key!(num_iters, U2048, U4096, U8192, dk);
    }
}
