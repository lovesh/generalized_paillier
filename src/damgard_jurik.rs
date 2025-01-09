//! Generalized Paillier encryption or Damgard-Jurik encryption as described in section 3.2 of the paper [A Generalization of Paillier’s Public-Key System with Applications to Electronic Voting](https://people.csail.mit.edu/rivest/voting/papers/DamgardJurikNielsen-AGeneralizationOfPailliersPublicKeySystemWithApplicationsToElectronicVoting.pdf)
//!
//! The generator `g` is chosen to be `1+n` and the decryption and ciphertext randomness extraction is
//! performed using Chinese Remainder Theorem. Both encryption and decryption use the optimizations described in
//! section 4.2
//!
//! `DecryptionKey` and `EncryptionKey` are independent of `S` and to encrypt, first a `PreparedEncryptionKey` with
//! required `S` should be created. `PreparedEncryptionKey` also contains precomputations to speed up encryption.
//! Similarly for decryption, `PreparedDecryptionKey` with appropriate `S` needs to be created. The goal is to do the
//! key generation once and then derive the keys for different message sizes (`S`) according to the use-case.
//!
//! Implements addition and multiplication of encrypted messages
//!
//! The code uses BoxedUint and BoxedMontyParams at some places as the traits Concat and PrecomputeInverter are not
//! implemented for some Uints. Might need to port the code to use BoxedUint, etc everywhere.

pub use crate::paillier_original::DecryptionKey;
use crate::{
    error::PaillierError,
    util::{crt_combine, crt_combine_using_box, l},
};
use alloc::{vec, vec::Vec};
use core::ops::{Add, Mul, Sub};
use crypto_bigint::{
    modular::{BoxedMontyForm, BoxedMontyParams, MontyForm, MontyParams, SafeGcdInverter},
    subtle::ConstantTimeLess,
    BoxedUint, Concat, Odd, PrecomputeInverter, RandomMod, Split, Uint, Zero,
};
use rand_core::CryptoRngCore;
use zeroize::{Zeroize, ZeroizeOnDrop};

#[derive(Debug, Clone, PartialEq)]
pub struct EncryptionKey<const MODULUS_LIMBS: usize>(
    /// Modulus `n=p*q`
    Odd<Uint<MODULUS_LIMBS>>,
);

/// Encryption key adapted for given `S`.
/// `MODULUS_LIMBS` are the limbs needed to represent the modulus `n`
/// `MSG_LIMBS` and `CT_LIMBS` are the limbs needed to represent the message mod `n^S` and ciphertext mod `n^{S+1}` respectively
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedEncryptionKey<
    const S: usize,
    const MODULUS_LIMBS: usize,
    const MSG_LIMBS: usize,
    const CT_LIMBS: usize,
> {
    /// Modulus `n=p*q`
    pub n: Odd<Uint<MODULUS_LIMBS>>,
    /// `n^S`
    pub n_s: Odd<Uint<MSG_LIMBS>>,
    /// `n^{S+1}`
    pub n_s1: Odd<Uint<CT_LIMBS>>,
    /// Montgomery params for reducing mod `n^{S+1}`
    pub n_s1_mtg_params: MontyParams<CT_LIMBS>,
    /// `n mod n^{S+1}` in Montgomery form
    pub n_mtg: MontyForm<CT_LIMBS>,
    /// `[(2!)^-1*n^2 mod n^{S+1}, (3!)^-1*n^3 mod n^{S+1}, ..., (S!)^-1*n^S mod n^{S+1}]`. The last array item is unused.
    pub precomputed: [MontyForm<CT_LIMBS>; S],
}

// TODO: Montgomery params are not zeroized but they should be
/// Decryption key adapted for given `S`.
/// `PRIME_LIMBS` are the limbs needed to represent prime `p` or `q`
/// `MSG_LIMBS`, `HALF_MSG_LIMBS` and `CT_LIMBS` are the limbs needed to represent the message mod `n^S`, message mod `p^S` (or `q^S`)
/// and ciphertext mod `n^{S+1}` respectively
#[derive(Debug, Clone, PartialEq, Zeroize, ZeroizeOnDrop)]
pub struct PreparedDecryptionKey<
    const S: usize,
    const PRIME_LIMBS: usize,
    const MSG_LIMBS: usize,
    const HALF_MSG_LIMBS: usize,
    const CT_LIMBS: usize,
> {
    pub p: Odd<Uint<PRIME_LIMBS>>,
    pub q: Odd<Uint<PRIME_LIMBS>>,
    /// Montgomery params for reducing mod `p`
    #[zeroize(skip)]
    pub p_mtg: MontyParams<PRIME_LIMBS>,
    #[zeroize(skip)]
    pub p_mtg_1: MontyParams<CT_LIMBS>,
    /// Montgomery params for reducing mod `q`
    #[zeroize(skip)]
    pub q_mtg: MontyParams<PRIME_LIMBS>,
    #[zeroize(skip)]
    pub q_mtg_1: MontyParams<CT_LIMBS>,
    /// Array of Montgomery params and item at index `i` is for reducing mod `p^{i+1}`
    #[zeroize(skip)]
    pub p_powers_mtg_params: [MontyParams<HALF_MSG_LIMBS>; S],
    /// Array of Montgomery params and item at index `i` is for reducing mod `q^{i+1}`
    #[zeroize(skip)]
    pub q_powers_mtg_params: [MontyParams<HALF_MSG_LIMBS>; S],
    /// Montgomery params for reducing mod `p^{S+1}`
    #[zeroize(skip)]
    pub p_s1_mtg: MontyParams<CT_LIMBS>,
    /// Montgomery params for reducing mod `q^{S+1}`
    #[zeroize(skip)]
    pub q_s1_mtg: MontyParams<CT_LIMBS>,
    /// `p^-1 mod q`
    pub p_inv: MontyForm<PRIME_LIMBS>,
    /// `p^-1 mod q^S`
    pub p_inv_qs: MontyForm<HALF_MSG_LIMBS>,
    /// `q^-1 mod p^S`
    pub q_inv_ps: MontyForm<HALF_MSG_LIMBS>,
    /// `(p-1)^-1 mod p^S`
    pub p_minus_1_inv_ps: MontyForm<HALF_MSG_LIMBS>,
    /// `(q-1)^-1 mod q^S`
    pub q_minus_1_inv_qs: MontyForm<HALF_MSG_LIMBS>,
    /// `p^-S mod q^S`
    pub ps_inv_qs: MontyForm<HALF_MSG_LIMBS>,
    /// `n^-s mod p-1`
    pub ns_inv_p: Uint<PRIME_LIMBS>,
    /// `n^-s mod q-1`
    pub ns_inv_q: Uint<PRIME_LIMBS>,
    /// `[2!^-1 * n  mod p^2, 3!^-1 * n^2  mod p^3, ..., S!^-1 * n^{S-1}  mod p^S]`
    pub facs_p: Vec<MontyForm<HALF_MSG_LIMBS>>,
    /// `[2!^-1 * n  mod q^2, 3!^-1 * n^2  mod q^3, ..., S!^-1 * n^{S-1}  mod q^S]`
    pub facs_q: Vec<MontyForm<HALF_MSG_LIMBS>>,
}

/// `MSG_LIMBS`, and `CT_LIMBS` are the limbs needed to represent the message mod `n^S` and ciphertext mod `n^{S+1}` respectively
#[derive(Debug, Clone, PartialEq)]
pub struct Ciphertext<
    const S: usize,
    const MODULUS_LIMBS: usize,
    const MSG_LIMBS: usize,
    const CT_LIMBS: usize,
>(Uint<CT_LIMBS>);

impl<const MODULUS_LIMBS: usize> EncryptionKey<MODULUS_LIMBS> {
    pub fn new<const PRIME_LIMBS: usize>(dk: &DecryptionKey<PRIME_LIMBS>) -> Self
    where
        Uint<PRIME_LIMBS>: Concat<Output = Uint<MODULUS_LIMBS>>,
    {
        const { assert!(2 * PRIME_LIMBS == MODULUS_LIMBS) };

        let n: Uint<MODULUS_LIMBS> = dk.p.widening_mul(&dk.q).into();
        Self(n.to_odd().unwrap())
    }
}

impl<const S: usize, const MODULUS_LIMBS: usize, const MSG_LIMBS: usize, const CT_LIMBS: usize>
    PreparedEncryptionKey<S, MODULUS_LIMBS, MSG_LIMBS, CT_LIMBS>
{
    const CHECK_S: () = assert!(S > 0);
    const CHECK_MSG_LIMBS: () = assert!((S * MODULUS_LIMBS) == MSG_LIMBS);
    const CHECK_CT_LIMBS: () = assert!(((S + 1) * MODULUS_LIMBS) == CT_LIMBS);

    pub fn new(ek: EncryptionKey<MODULUS_LIMBS>) -> Self {
        let _ = Self::CHECK_S;
        let _ = Self::CHECK_MSG_LIMBS;
        let _ = Self::CHECK_CT_LIMBS;

        let n_wide = ek.0.as_ref();
        let mut n_s: Uint<MSG_LIMBS> = ek.0.resize();
        // n_js = [n^2, n^3, ... n^S]
        let mut n_js = [Uint::<CT_LIMBS>::ZERO; S]; // last item will be unused since Rust doesn't allow [Uint::ZERO; S-1]
        for i in 1..S {
            n_s = n_s * n_wide;
            n_js[i - 1] = n_s.resize();
        }
        // n_s1 = n^{S+1}
        let n_s1: Uint<CT_LIMBS> = n_s.resize() * n_wide;
        let n_s1_odd = n_s1.to_odd().unwrap();
        let n_s1_mtg_params = MontyParams::new_vartime(n_s1_odd);
        let n_s1_bx_mtg =
            BoxedMontyParams::new_vartime(BoxedUint::from(n_s1_odd).to_odd().unwrap());
        let mut i_fac = Uint::<CT_LIMBS>::ONE;
        let mut precomputed = [MontyForm::zero(n_s1_mtg_params); S]; // last item will be unused since Rust doesn't allow [Uint::ZERO; S-1]
        for i in 2..=S {
            // In practice, S won't be large enough that factorial of i doesn't fit in a single word. 32 bit word will work for S=12
            i_fac = i_fac.mul(Uint::<CT_LIMBS>::from(i as u128));
            let i_fac_inv = BoxedMontyForm::new(BoxedUint::from(i_fac), n_s1_bx_mtg.clone())
                .invert()
                .unwrap()
                .retrieve();
            let i_fac_inv = Uint::<CT_LIMBS>::from_le_slice(i_fac_inv.to_le_bytes().as_ref());
            let i_fac_inv = MontyForm::new(&i_fac_inv, n_s1_mtg_params);
            // i!^-1 mod n^{S+1} * n^{i-2}
            precomputed[i - 2] = i_fac_inv * MontyForm::new(&n_js[i - 2], n_s1_mtg_params);
        }
        Self {
            n: ek.0,
            n_s: n_s.to_odd().unwrap(),
            n_s1: n_s1.to_odd().unwrap(),
            n_mtg: MontyForm::new(&n_wide.resize(), n_s1_mtg_params),
            n_s1_mtg_params,
            precomputed,
        }
    }
}

impl<
        const S: usize,
        const PRIME_LIMBS: usize,
        const MSG_LIMBS: usize,
        const HALF_MSG_LIMBS: usize,
        const CT_LIMBS: usize,
    > PreparedDecryptionKey<S, PRIME_LIMBS, MSG_LIMBS, HALF_MSG_LIMBS, CT_LIMBS>
{
    const CHECK_S: () = assert!(S > 0);
    const CHECK_HALF_MSG_LIMBS: () = assert!((S * PRIME_LIMBS) == HALF_MSG_LIMBS);
    const CHECK_MSG_LIMBS: () = assert!((S * 2 * PRIME_LIMBS) == MSG_LIMBS);
    const CHECK_CT_LIMBS: () = assert!(((S + 1) * 2 * PRIME_LIMBS) == CT_LIMBS);

    /// `PRIME_UNSAT_LIMBS` and `HALF_MSG_UNSAT_LIMBS` are the number of limbs need to invert values with
    /// limbs `PRIME_LIMBS` and `HALF_MSG_LIMBS` respectively using `SafeGcdInverter`
    fn new<
        const MODULUS_LIMBS: usize,
        const CT_WIDE_LIMBS: usize,
        const PRIME_UNSAT_LIMBS: usize,
        // const MSG_UNSAT_LIMBS: usize,
        const HALF_MSG_UNSAT_LIMBS: usize,
    >(
        dk: DecryptionKey<PRIME_LIMBS>,
        ek: PreparedEncryptionKey<S, MODULUS_LIMBS, MSG_LIMBS, CT_LIMBS>,
    ) -> Self
    where
        Uint<PRIME_LIMBS>: Concat<Output = Uint<MODULUS_LIMBS>>,
        Uint<MODULUS_LIMBS>: Split<Output = Uint<PRIME_LIMBS>>,
        // Uint<HALF_MSG_LIMBS>: Concat<Output = Uint<MSG_LIMBS>>,
        // Uint<MSG_LIMBS>: Split<Output = Uint<HALF_MSG_LIMBS>>,
        // Uint<CT_LIMBS>: Concat<Output = Uint<CT_WIDE_LIMBS>>,
        // Uint<CT_WIDE_LIMBS>: Split<Output = Uint<CT_LIMBS>>,
        Odd<Uint<PRIME_LIMBS>>:
            PrecomputeInverter<Inverter = SafeGcdInverter<PRIME_LIMBS, PRIME_UNSAT_LIMBS>>,
        // Odd<Uint<MSG_LIMBS>>:
        //     PrecomputeInverter<Inverter = SafeGcdInverter<MSG_LIMBS, MSG_UNSAT_LIMBS>>,
        Odd<Uint<HALF_MSG_LIMBS>>: PrecomputeInverter<
            Inverter = SafeGcdInverter<HALF_MSG_LIMBS, HALF_MSG_UNSAT_LIMBS>,
            Output = Uint<HALF_MSG_LIMBS>,
        >,
    {
        let _ = Self::CHECK_S;
        let _ = Self::CHECK_HALF_MSG_LIMBS;
        let _ = Self::CHECK_MSG_LIMBS;
        let _ = Self::CHECK_CT_LIMBS;

        let p_mtg = MontyParams::new(dk.p);
        let q_mtg = MontyParams::new(dk.q);
        // TODO: Use const time in the following. But Concat and Split don't exists for certain Uint types so using BoxedMontyParams
        // is the only solution for now.
        let p_mtg_1: MontyParams<CT_LIMBS> =
            MontyParams::new_vartime(dk.p.resize().to_odd().unwrap());
        let q_mtg_1: MontyParams<CT_LIMBS> =
            MontyParams::new_vartime(dk.q.resize().to_odd().unwrap());
        let p_minus_1 = dk.p.wrapping_sub(&Uint::ONE);
        let q_minus_1 = dk.q.wrapping_sub(&Uint::ONE);

        // Using BoxedUint because PrecomputeInverter isn't defined for certain Uints.
        let n_s_bx = BoxedUint::from(ek.n_s).to_odd().unwrap();
        let p_minus_1_bx = BoxedUint::from(p_minus_1.resize::<MSG_LIMBS>());
        let q_minus_1_bx = BoxedUint::from(q_minus_1.resize::<MSG_LIMBS>());
        let ns_inv_p = n_s_bx.inv_mod(&p_minus_1_bx).unwrap();
        let ns_inv_p = Uint::<MSG_LIMBS>::from_le_slice(ns_inv_p.to_le_bytes().as_ref()).resize();
        let ns_inv_q = n_s_bx.inv_mod(&q_minus_1_bx).unwrap();
        let ns_inv_q = Uint::<MSG_LIMBS>::from_le_slice(ns_inv_q.to_le_bytes().as_ref()).resize();

        // p^-1 mod q
        let p_inv = MontyForm::new(&dk.p.inv_odd_mod(&dk.q).unwrap(), q_mtg);

        // TODO: Use const time in the following
        let mut cur_p = dk.p.resize::<HALF_MSG_LIMBS>();
        let mut cur_q = dk.q.resize::<HALF_MSG_LIMBS>();
        let mut p_params = [MontyParams::new_vartime(cur_p.to_odd().unwrap()); S];
        let mut q_params = [MontyParams::new_vartime(cur_q.to_odd().unwrap()); S];
        for i in 2..=S {
            cur_p = cur_p.mul(&dk.p.resize::<HALF_MSG_LIMBS>());
            cur_q = cur_q.mul(&dk.q.resize::<HALF_MSG_LIMBS>());
            p_params[i - 1] = MontyParams::new_vartime(cur_p.resize().to_odd().unwrap());
            q_params[i - 1] = MontyParams::new_vartime(cur_q.resize().to_odd().unwrap());
        }

        // TODO: Use const time in the following
        let p_s1_mtg = MontyParams::new_vartime(
            cur_p
                .resize::<CT_LIMBS>()
                .mul(&dk.p.resize::<CT_LIMBS>())
                .to_odd()
                .unwrap(),
        );
        let q_s1_mtg = MontyParams::new_vartime(
            cur_q
                .resize::<CT_LIMBS>()
                .mul(&dk.q.resize::<CT_LIMBS>())
                .to_odd()
                .unwrap(),
        );

        let p_s_params = p_params[S - 1];
        let q_s_params = q_params[S - 1];

        // The unwraps below are fine as both p and q are primes and p-1 and q-1 are coprime to their powers
        // p^-1 mod q^S
        let p_inv_qs = MontyForm::new(&dk.p.resize(), q_s_params).inv().unwrap();
        // q^-1 mod p^S
        let q_inv_ps = MontyForm::new(&dk.q.resize(), p_s_params).inv().unwrap();
        // (p-1)^-1 mod p^S
        let p_minus_1_inv_ps = MontyForm::new(&p_minus_1.resize(), p_s_params)
            .inv()
            .unwrap();
        // (q-1)^-1 mod p^S
        let q_minus_1_inv_qs = MontyForm::new(&q_minus_1.resize(), q_s_params)
            .inv()
            .unwrap();
        // p^-S mod q^S
        let ps_inv_qs = if S > 1 {
            p_inv_qs.pow(&Uint::<HALF_MSG_LIMBS>::from(S as u128))
        } else {
            p_inv_qs
        };

        // NOTE: The following could be array if Rust allowed generic const-exprs
        let mut facs_p = vec![MontyForm::one(p_s_params); S * (S - 1) / 2];
        let mut facs_q = vec![MontyForm::one(q_s_params); S * (S - 1) / 2];
        let mut ctr = 0;
        for j in 2..=S {
            let mut k_fac = Uint::ONE;
            for k in 2..=j {
                k_fac = k_fac.wrapping_mul(&Uint::<HALF_MSG_LIMBS>::from(k as u128));
                // unwraps are fine as both p and q are primes
                // k_fac_inv_p = k!^-1 mod p^j
                let k_fac_inv_p = MontyForm::new(&k_fac, p_params[j - 1]).inv().unwrap();
                // k_fac_inv_q = k!^-1 mod q^j
                let k_fac_inv_q = MontyForm::new(&k_fac, q_params[j - 1]).inv().unwrap();
                // n_k_p = n^{k-1} mod p^j
                let mut n_k_p = MontyForm::new(p_params[k - 2].modulus(), p_params[j - 1]);
                n_k_p = n_k_p * MontyForm::new(q_params[k - 2].modulus(), p_params[j - 1]);

                // n_k_q = n^{k-1} mod q^j
                let mut n_k_q = MontyForm::new(p_params[k - 2].modulus(), q_params[j - 1]);
                n_k_q = n_k_q * MontyForm::new(q_params[k - 2].modulus(), q_params[j - 1]);

                // k!^-1 * n^{k-1}  mod p^j
                facs_p[ctr] = k_fac_inv_p * n_k_p;
                // k!^-1 * n^{k-1}  mod q^j
                facs_q[ctr] = k_fac_inv_q * n_k_q;
                ctr += 1;
            }
        }
        debug_assert_eq!(ctr, S * (S - 1) / 2);

        Self {
            p: dk.p,
            q: dk.q,
            p_mtg,
            q_mtg,
            p_mtg_1,
            q_mtg_1,
            ns_inv_p,
            ns_inv_q,
            p_inv,
            p_inv_qs,
            q_inv_ps,
            p_minus_1_inv_ps,
            q_minus_1_inv_qs,
            ps_inv_qs,
            p_powers_mtg_params: p_params,
            q_powers_mtg_params: q_params,
            p_s1_mtg,
            q_s1_mtg,
            facs_p,
            facs_q,
        }
    }
}

impl<const S: usize, const MODULUS_LIMBS: usize, const MSG_LIMBS: usize, const CT_LIMBS: usize>
    Ciphertext<S, MODULUS_LIMBS, MSG_LIMBS, CT_LIMBS>
{
    const CHECK_S: () = assert!(S > 0);
    const CHECK_MSG_LIMBS: () = assert!((S * MODULUS_LIMBS) == MSG_LIMBS);
    const CHECK_CT_LIMBS: () = assert!(((S + 1) * MODULUS_LIMBS) == CT_LIMBS);

    /// Encrypt the given message. Generates randomness internally
    pub fn new<R: CryptoRngCore>(
        rng: &mut R,
        msg: &Uint<MSG_LIMBS>,
        ek: PreparedEncryptionKey<S, MODULUS_LIMBS, MSG_LIMBS, CT_LIMBS>,
    ) -> Result<Self, PaillierError> {
        let r = Uint::random_mod(rng, ek.n.as_nz_ref());
        Self::new_given_randomness(msg, r, ek)
    }

    /// Encrypt the given message with given randomness
    pub fn new_given_randomness(
        msg: &Uint<MSG_LIMBS>,
        r: Uint<MODULUS_LIMBS>,
        ek: PreparedEncryptionKey<S, MODULUS_LIMBS, MSG_LIMBS, CT_LIMBS>,
    ) -> Result<Self, PaillierError> {
        let _ = Self::CHECK_S;
        let _ = Self::CHECK_MSG_LIMBS;
        let _ = Self::CHECK_CT_LIMBS;

        if !bool::from(msg.ct_lt(&ek.n_s)) {
            return Err(PaillierError::MessageOutOfBound);
        }
        // `r` should also be coprime to `n` but that's going to be true except with a negligible probability and when it
        // happens, `n` has been factored.
        if !bool::from(r.ct_lt(&ek.n)) {
            return Err(PaillierError::RandomnessOutOfBound);
        }

        let (g_m, r_n_s) = join!(Self::g_m(msg, &ek), {
            let r = r.resize::<CT_LIMBS>();
            MontyForm::new(&r, ek.n_s1_mtg_params).pow(ek.n_s.as_ref())
        });
        Ok(Self((r_n_s * g_m).retrieve()))
    }

    /// Decrypt the ciphertext and returns the message. Assumes that the ciphertext is valid, i.e. in `[0, n^S)`
    pub fn decrypt<const PRIME_LIMBS: usize, const HALF_MSG_LIMBS: usize>(
        &self,
        dk: PreparedDecryptionKey<S, PRIME_LIMBS, MSG_LIMBS, HALF_MSG_LIMBS, CT_LIMBS>,
    ) -> Uint<MSG_LIMBS>
where
        // Uint<HALF_MSG_LIMBS>: Concat<Output = Uint<MSG_LIMBS>>,
    {
        // m_p = m mod p^S and m_q = m mod q^S
        let (m_p, m_q) = join!(
            {
                self.message_mod_prime_power(
                    dk.p,
                    dk.p_s1_mtg,
                    dk.p_powers_mtg_params,
                    dk.facs_p.clone(),
                    dk.q_inv_ps,
                    dk.p_minus_1_inv_ps,
                )
            },
            {
                self.message_mod_prime_power(
                    dk.q,
                    dk.q_s1_mtg,
                    dk.q_powers_mtg_params,
                    dk.facs_q.clone(),
                    dk.p_inv_qs,
                    dk.q_minus_1_inv_qs,
                )
            }
        );

        crt_combine_using_box(
            &m_p,
            &m_q,
            dk.ps_inv_qs,
            dk.p_powers_mtg_params[S - 1].modulus().as_ref(),
            dk.q_powers_mtg_params[S - 1],
        )

        // Following doesn't work as trait Concat is not defined for some Uints
        // crt_combine(
        //     &m_p,
        //     &m_q,
        //     dk.ps_inv_qs,
        //     dk.p_params[S - 1].modulus().as_ref(),
        //     dk.q_params[S - 1],
        // )
    }

    /// Get randomness used in the ciphertext using CRT
    pub fn get_randomness<const PRIME_LIMBS: usize, const HALF_MSG_LIMBS: usize>(
        &self,
        msg: &Uint<MSG_LIMBS>,
        dk: PreparedDecryptionKey<S, PRIME_LIMBS, MSG_LIMBS, HALF_MSG_LIMBS, CT_LIMBS>,
        ek: PreparedEncryptionKey<S, MODULUS_LIMBS, MSG_LIMBS, CT_LIMBS>,
    ) -> Result<Uint<MODULUS_LIMBS>, PaillierError>
    where
        Uint<PRIME_LIMBS>: Concat<Output = Uint<MODULUS_LIMBS>>,
        Uint<MODULUS_LIMBS>: Split<Output = Uint<PRIME_LIMBS>>,
    {
        if !bool::from(msg.ct_lt(&ek.n_s)) {
            return Err(PaillierError::MessageOutOfBound);
        }
        let neg_msg = msg.neg_mod(ek.n_s.as_ref());
        let c = Self::g_m(&neg_msg, &ek);
        // c_z = r^{n^S} mod n^{S+1}
        let c_z = MontyForm::new(&self.0, ek.n_s1_mtg_params) * c;

        // r_p = r^{n^S - n^S mod p-1} mod p = r mod p
        // r_p = r^{n^S - n^S mod q-1} mod q = r mod q
        let (r_p, r_q) = join!(
            {
                // c_p = r^{n^S} mod p
                let c_p = MontyForm::new(&c_z.retrieve(), dk.p_mtg_1)
                    .retrieve()
                    .resize();
                let c_p = MontyForm::new(&c_p, dk.p_mtg);
                c_p.pow(&dk.ns_inv_p).retrieve()
            },
            {
                // c_q = r^{n^S} mod q
                let c_q = MontyForm::new(&c_z.retrieve(), dk.q_mtg_1)
                    .retrieve()
                    .resize();
                let c_q = MontyForm::new(&c_q, dk.q_mtg);
                c_q.pow(&dk.ns_inv_q).retrieve()
            }
        );
        Ok(crt_combine(&r_p, &r_q, dk.p_inv, &dk.p, dk.q_mtg))
    }

    /// Combine another ciphertext with this ciphertext such that the resulting ciphertext
    /// encrypts the sum of the 2 messages
    pub fn add(
        &self,
        rhs: &Self,
        ek: PreparedEncryptionKey<S, MODULUS_LIMBS, MSG_LIMBS, CT_LIMBS>,
    ) -> Result<Self, PaillierError> {
        if !rhs.is_valid(&ek) {
            return Err(PaillierError::CiphertextOutOfBound);
        }
        let l = MontyForm::new(&self.0, ek.n_s1_mtg_params);
        let r = MontyForm::new(&rhs.0, ek.n_s1_mtg_params);
        Ok(Self((l * r).retrieve()))
    }

    /// Return an updated ciphertext which encrypts the product of current encrypted message and
    /// the given message
    pub fn mul(
        &self,
        msg: &Uint<MSG_LIMBS>,
        ek: PreparedEncryptionKey<S, MODULUS_LIMBS, MSG_LIMBS, CT_LIMBS>,
    ) -> Result<Self, PaillierError> {
        if !bool::from(msg.ct_lt(ek.n_s.as_ref())) {
            return Err(PaillierError::MessageOutOfBound);
        }
        let l = MontyForm::new(&self.0, ek.n_s1_mtg_params);
        Ok(Self(l.pow(msg).retrieve()))
    }

    /// Returns true if ciphertext is valid, i.e. in `[0, n^2)`
    pub fn is_valid(
        &self,
        ek: &PreparedEncryptionKey<S, MODULUS_LIMBS, MSG_LIMBS, CT_LIMBS>,
    ) -> bool {
        self.0.lt(ek.n_s1.as_ref())
    }

    /// Used in decryption with CRT. Returns `message mod prime^S`
    fn message_mod_prime_power<const PRIME_LIMBS: usize, const HALF_MSG_LIMBS: usize>(
        &self,
        prime: Odd<Uint<PRIME_LIMBS>>,
        prime_s1_mtg: MontyParams<CT_LIMBS>,
        prime_powers_mtg_params: [MontyParams<HALF_MSG_LIMBS>; S],
        precomputed_fac_terms: Vec<MontyForm<HALF_MSG_LIMBS>>,
        other_prime_inv: MontyForm<HALF_MSG_LIMBS>,
        prime_minus_1_inv: MontyForm<HALF_MSG_LIMBS>,
    ) -> Uint<HALF_MSG_LIMBS> {
        // works because for ciphertext c, message m and prime power p^S:
        // m_p = L_p(c^{p-1} mod p^{S+1}) * (p-1)^-1 mod p^S where L_p(x) = ((x-1)/p * q^-1 mod p^S) and n = p*q
        // c = (1+n)^m*r^{n^S} mod p^{S+1} and c^{p-1} mod p^{S+1} = (1+n)^{m*(p-1)}*r^{n^S*(p-1)} mod p^{S+1}.
        // r^{n^S*(p-1)} mod p^{S+1} = r^{q^S*p^S*(p-1)} mod p^{S+1} = 1 mod p^{S+1} since r^{p^S*(p-1)} mod p^{S+1} = 1
        // Now m_p = L_p((1+n)^{m*(p-1)} mod p^{S+1}) * (p-1)^-1 mod p^S
        // m*(p-1) mod p^S can be extracted from L_p((1+n)^{m*(p-1)} mod p^{S+1}) using the iterative algorithm from Theorem 1 in the paper.
        // m_p = m * (p-1) * (p-1)^-1 mod p^S = m mod p^S

        // g_m = ciphertext^{p-1} mod p^{S+1} = (1+n)^{m*(p-1)} mod p^{S+1}
        let g_m = MontyForm::new(&self.0, prime_s1_mtg)
            .pow(&prime.sub(Uint::ONE))
            .retrieve();

        // L_p((1+n)^{m*(p-1)} mod p^{S+1}) = l_mod_p_s = ((g_m - 1) / p) * q^-1 mod p^S
        let l_mod_p_s = MontyForm::new(&l(&g_m, prime.as_nz_ref()), prime_powers_mtg_params[S - 1])
            * other_prime_inv;

        // i will eventually be m*(p-1) mod p^S
        let mut i = Uint::<HALF_MSG_LIMBS>::zero();
        let mut ctr = 0;
        for j in 1..=S {
            // i mod p^j
            let mut i_mtg = MontyForm::new(&i, prime_powers_mtg_params[j - 1]);
            // L_p((1+n)^i mod p^{j+1}) = L_p((1+n)^i mod p^{S+1}) mod p^j
            let mut t1 = MontyForm::new(&l_mod_p_s.retrieve(), prime_powers_mtg_params[j - 1]);
            let mut t2 = i_mtg.clone();
            for _ in 2..=j as u128 {
                i_mtg = i_mtg - MontyForm::one(prime_powers_mtg_params[j - 1]);
                t2 = t2 * i_mtg;
                t1 = t1 - (t2 * precomputed_fac_terms[ctr]);
                ctr += 1;
            }
            // i = m*(p-1) mod p^j
            i = t1.retrieve();
        }
        debug_assert_eq!(ctr, precomputed_fac_terms.len());
        let msg = MontyForm::new(&i.resize(), prime_powers_mtg_params[S - 1]) * prime_minus_1_inv;
        msg.retrieve().resize()
    }

    /// Calculate c = (1+n)^msg mod n^{S+1} using the optimization described in section 4.2
    fn g_m(
        msg: &Uint<MSG_LIMBS>,
        ek: &PreparedEncryptionKey<S, MODULUS_LIMBS, MSG_LIMBS, CT_LIMBS>,
    ) -> MontyForm<CT_LIMBS> {
        let m = MontyForm::new(&msg.resize(), ek.n_s1_mtg_params);
        let mn = m * ek.n_mtg;
        // c = 1 + m*n
        let mut c = mn.add(&MontyForm::one(ek.n_s1_mtg_params));
        if S > 1 {
            let mut numerator = m;
            for i in 2..=S {
                // TODO: This has to be done only mod n^{S+1-i} so the following is inefficient
                // numerator = numerator * m-(i-1)
                numerator = numerator
                    * (m - MontyForm::new(&Uint::from(i as u128 - 1), ek.n_s1_mtg_params));
                c = c + numerator * ek.precomputed[i - 2];
            }
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::{get_1024_bit_primes, get_2048_bit_primes, timing_info};
    use crypto_bigint::{U1024, U128, U2048, U256, U4096, U64};
    use rand::rngs::OsRng;
    use std::time::Instant;

    // Taken from safegcd.rs in crypto-bigint. Used to find out the limbs needed to perform modular
    // inversion using SafeGcdInverter
    macro_rules! safegcd_nlimbs {
        ($bits:expr) => {
            ($bits + 64).div_ceil(62)
        };
    }

    macro_rules! check_ops_given_dec_key {
        ( $num_iters: ident, $S: expr, $prime_type:ident, $modulo_type:ident, $dk: ident ) => {
            let mut rng = OsRng::default();
            const S: usize = $S;
            const PRIME_LIMBS: usize = $prime_type::LIMBS;
            const MODULUS_LIMBS: usize = $modulo_type::LIMBS;
            const MSG_LIMBS: usize = $modulo_type::LIMBS * S;
            const HALF_MSG_LIMBS: usize = MSG_LIMBS / 2;
            const CT_LIMBS: usize = $modulo_type::LIMBS * (S + 1);
            const CT_WIDE_LIMBS: usize = CT_LIMBS * 2;
            const PRIME_UNSAT_LIMBS: usize = safegcd_nlimbs!(Uint::<PRIME_LIMBS>::BITS as usize);
            // const MSG_UNSAT_LIMBS: usize = safegcd_nlimbs!(Uint::<MSG_LIMBS>::BITS as usize);
            const HALF_MSG_UNSAT_LIMBS: usize =
                safegcd_nlimbs!(Uint::<HALF_MSG_LIMBS>::BITS as usize);
            // const CT_UNSAT_LIMBS: usize = safegcd_nlimbs!(Uint::<CT_LIMBS>::BITS as usize);

            let ek = EncryptionKey::<MODULUS_LIMBS>::new(&$dk);
            let pek = PreparedEncryptionKey::<S, MODULUS_LIMBS, MSG_LIMBS, CT_LIMBS>::new(ek);
            let pdk =
                PreparedDecryptionKey::<S, PRIME_LIMBS, MSG_LIMBS, HALF_MSG_LIMBS, CT_LIMBS>::new::<
                    MODULUS_LIMBS,
                    CT_WIDE_LIMBS,
                    PRIME_UNSAT_LIMBS,
                    HALF_MSG_UNSAT_LIMBS,
                >($dk.clone(), pek.clone());

            let mut sum_times = vec![];
            let mut product_times = vec![];

            let n_mtg = MontyParams::new_vartime(pek.n.to_odd().unwrap());

            println!(
                "Running {} iterations for {} bits prime and S={}",
                $num_iters,
                $prime_type::BITS,
                S
            );
            for _ in 0..$num_iters {
                let m1 = Uint::<MSG_LIMBS>::random_mod(&mut rng, pek.n_s.as_nz_ref());
                let r1 = $modulo_type::random_mod(&mut rng, pek.n.as_nz_ref());
                let m2 = Uint::<MSG_LIMBS>::random_mod(&mut rng, pek.n_s.as_nz_ref());
                let r2 = $modulo_type::random_mod(&mut rng, pek.n.as_nz_ref());

                let ct1 = Ciphertext::new_given_randomness(&m1, r1, pek.clone()).unwrap();
                let ct2 = Ciphertext::new_given_randomness(&m2, r2, pek.clone()).unwrap();

                let start = Instant::now();
                let ct_sum = ct1.add(&ct2, pek.clone()).unwrap();
                sum_times.push(start.elapsed());
                let expected_msg = m1.add_mod(&m2, &pek.n_s);
                let m_ = ct_sum.decrypt(pdk.clone());
                assert_eq!(expected_msg, m_);

                let expected_r = r1.mul_mod(&r2, pek.n.as_nz_ref());
                let r_ = ct_sum
                    .get_randomness::<PRIME_LIMBS, HALF_MSG_LIMBS>(&m_, pdk.clone(), pek.clone())
                    .unwrap();
                assert_eq!(expected_r, r_);

                let start = Instant::now();
                let ct_prod = ct1.mul(&m2, pek.clone()).unwrap();
                product_times.push(start.elapsed());
                let expected_msg = m1.mul_mod(&m2, pek.n_s.as_nz_ref());
                let m_ = ct_prod.decrypt(pdk.clone());
                assert_eq!(expected_msg, m_);

                let expected_r = MontyForm::new(&r1, n_mtg).pow(&m2).retrieve();
                let r_ = ct_prod
                    .get_randomness::<PRIME_LIMBS, HALF_MSG_LIMBS>(&m_, pdk.clone(), pek.clone())
                    .unwrap();
                assert_eq!(expected_r, r_);
            }

            println!("Sum time: {:?}", timing_info(sum_times));
            println!("Product time: {:?}", timing_info(product_times));
        };
    }

    macro_rules! check_enc_dec_given_dec_key {
        ( $num_iters: ident, $S: expr, $prime_type:ident, $modulo_type:ident, $dk: ident ) => {
            let mut rng = OsRng::default();
            const S: usize = $S;
            const PRIME_LIMBS: usize = $prime_type::LIMBS;
            const MODULUS_LIMBS: usize = $modulo_type::LIMBS;
            const MSG_LIMBS: usize = $modulo_type::LIMBS * S;
            const HALF_MSG_LIMBS: usize = MSG_LIMBS / 2;
            const CT_LIMBS: usize = $modulo_type::LIMBS * (S + 1);
            const CT_WIDE_LIMBS: usize = CT_LIMBS * 2;
            const PRIME_UNSAT_LIMBS: usize = safegcd_nlimbs!(Uint::<PRIME_LIMBS>::BITS as usize);
            const HALF_MSG_UNSAT_LIMBS: usize =
                safegcd_nlimbs!(Uint::<HALF_MSG_LIMBS>::BITS as usize);

            let ek = EncryptionKey::<MODULUS_LIMBS>::new(&$dk);
            let pek = PreparedEncryptionKey::<S, MODULUS_LIMBS, MSG_LIMBS, CT_LIMBS>::new(ek);
            let pdk =
                PreparedDecryptionKey::<S, PRIME_LIMBS, MSG_LIMBS, HALF_MSG_LIMBS, CT_LIMBS>::new::<
                    MODULUS_LIMBS,
                    CT_WIDE_LIMBS,
                    PRIME_UNSAT_LIMBS,
                    HALF_MSG_UNSAT_LIMBS,
                >($dk.clone(), pek.clone());

            let mut p_s: Uint<MSG_LIMBS> = $dk.p.resize();
            for _ in 1..S {
                p_s = p_s.wrapping_mul(&$dk.p.resize::<MSG_LIMBS>());
            }
            let mut q_s: Uint<MSG_LIMBS> = $dk.q.resize();
            for _ in 1..S {
                q_s = q_s.wrapping_mul(&$dk.q.resize::<MSG_LIMBS>());
            }
            assert_eq!(p_s, pdk.p_powers_mtg_params[S - 1].modulus().resize());
            assert_eq!(q_s, pdk.q_powers_mtg_params[S - 1].modulus().resize());

            assert_eq!(pdk.facs_p.len(), S * (S - 1) / 2);
            assert_eq!(pdk.facs_q.len(), S * (S - 1) / 2);

            let mut enc_times = vec![];
            let mut dec_times = vec![];
            let mut rnd_times = vec![];

            println!(
                "Running {} iterations for {} bits prime and S={}",
                $num_iters,
                $prime_type::BITS,
                S
            );
            for _ in 0..$num_iters {
                let m = Uint::<MSG_LIMBS>::random_mod(&mut rng, pek.n_s.as_nz_ref());
                let start = Instant::now();
                let ct = Ciphertext::new(&mut rng, &m, pek.clone()).unwrap();
                enc_times.push(start.elapsed());

                // m_p and m_q correspond to m mod p^S and m mod q^S respectively
                let m_p = ct.message_mod_prime_power(
                    pdk.p,
                    pdk.p_s1_mtg,
                    pdk.p_powers_mtg_params,
                    pdk.facs_p.clone(),
                    pdk.q_inv_ps,
                    pdk.p_minus_1_inv_ps,
                );
                assert_eq!(m_p, m.rem(&p_s.to_nz().unwrap()).resize());

                let m_q = ct.message_mod_prime_power(
                    pdk.q,
                    pdk.q_s1_mtg,
                    pdk.q_powers_mtg_params,
                    pdk.facs_q.clone(),
                    pdk.p_inv_qs,
                    pdk.q_minus_1_inv_qs,
                );
                assert_eq!(m_q, m.rem(&q_s.to_nz().unwrap()).resize());

                let start = Instant::now();
                let m_ = ct.decrypt(pdk.clone());
                dec_times.push(start.elapsed());
                assert_eq!(m, m_);

                let r = $modulo_type::random_mod(&mut rng, pek.n.as_nz_ref());
                let ct = Ciphertext::new_given_randomness(&m, r, pek.clone()).unwrap();
                let m_ = ct.decrypt(pdk.clone());
                assert_eq!(m, m_);
                let start = Instant::now();
                let r_ = ct
                    .get_randomness::<PRIME_LIMBS, HALF_MSG_LIMBS>(&m, pdk.clone(), pek.clone())
                    .unwrap();
                rnd_times.push(start.elapsed());
                assert_eq!(r, r_);
            }

            println!("Enc time: {:?}", timing_info(enc_times));
            println!("Dec time: {:?}", timing_info(dec_times));
            println!(
                "Time to get encryption randomness: {:?}",
                timing_info(rnd_times)
            );

            // Should error on message >= n^S
            assert!(Ciphertext::new(&mut rng, pek.n_s.as_ref(), pek.clone()).is_err());
            assert!(
                Ciphertext::new(&mut rng, &pek.n_s.get().add(&Uint::ONE), pek.clone()).is_err()
            );

            // Should error on randomness >= n
            let m = pek.n_s.get().sub(Uint::ONE); // m = n^S-1
            assert!(Ciphertext::new_given_randomness(&m, pek.n.get(), pek.clone()).is_err());
            assert!(
                Ciphertext::new_given_randomness(&m, pek.n.get().add(&Uint::ONE), pek.clone())
                    .is_err()
            );

            // Should error on message >= n^S
            let r = $modulo_type::random_mod(&mut rng, pek.n.as_nz_ref());
            let ct = Ciphertext::new_given_randomness(&m, r, pek.clone()).unwrap();
            assert!(ct
                .get_randomness::<PRIME_LIMBS, HALF_MSG_LIMBS>(
                    pek.n_s.as_ref(),
                    pdk.clone(),
                    pek.clone()
                )
                .is_err());
        };
    }

    #[test]
    fn encrypt_decrypt_s_1_64_bit_prime() {
        let mut rng = OsRng::default();
        let num_iters = 20;
        let dk = DecryptionKey::<{ U64::LIMBS }>::new(&mut rng);
        check_enc_dec_given_dec_key!(num_iters, 1, U64, U128, dk);
    }

    #[test]
    fn encrypt_decrypt_s_2_64_bit_prime() {
        let mut rng = OsRng::default();
        let num_iters = 20;
        let dk = DecryptionKey::<{ U64::LIMBS }>::new(&mut rng);
        check_enc_dec_given_dec_key!(num_iters, 2, U64, U128, dk);
    }

    #[test]
    fn encrypt_decrypt_s_3_64_bit_prime() {
        let mut rng = OsRng::default();
        let num_iters = 20;
        let dk = DecryptionKey::<{ U64::LIMBS }>::new(&mut rng);
        check_enc_dec_given_dec_key!(num_iters, 3, U64, U128, dk);
    }

    #[test]
    fn encrypt_decrypt_s_4_64_bit_prime() {
        let mut rng = OsRng::default();
        let num_iters = 20;
        let dk = DecryptionKey::<{ U64::LIMBS }>::new(&mut rng);
        check_enc_dec_given_dec_key!(num_iters, 4, U64, U128, dk);
    }

    #[test]
    fn ciphertext_add_mul_s_1_64_bit_prime() {
        let mut rng = OsRng::default();
        let num_iters = 20;
        let dk = DecryptionKey::<{ U64::LIMBS }>::new(&mut rng);
        check_ops_given_dec_key!(num_iters, 1, U64, U128, dk);
    }

    #[test]
    fn ciphertext_add_mul_s_2_64_bit_prime() {
        let mut rng = OsRng::default();
        let num_iters = 20;
        let dk = DecryptionKey::<{ U64::LIMBS }>::new(&mut rng);
        check_ops_given_dec_key!(num_iters, 2, U64, U128, dk);
    }

    #[test]
    fn encrypt_decrypt_s_1_128_bit_prime() {
        let mut rng = OsRng::default();
        let num_iters = 20;
        let dk = DecryptionKey::<{ U128::LIMBS }>::new(&mut rng);
        check_enc_dec_given_dec_key!(num_iters, 1, U128, U256, dk);
    }

    #[test]
    fn encrypt_decrypt_s_2_128_bit_prime() {
        let mut rng = OsRng::default();
        let num_iters = 20;
        let dk = DecryptionKey::<{ U128::LIMBS }>::new(&mut rng);
        check_enc_dec_given_dec_key!(num_iters, 2, U128, U256, dk);
    }

    #[test]
    fn encrypt_decrypt_s_3_128_bit_prime() {
        let mut rng = OsRng::default();
        let num_iters = 20;
        let dk = DecryptionKey::<{ U128::LIMBS }>::new(&mut rng);
        check_enc_dec_given_dec_key!(num_iters, 3, U128, U256, dk);
    }

    #[test]
    fn encrypt_decrypt_s_4_128_bit_prime() {
        let mut rng = OsRng::default();
        let num_iters = 20;
        let dk = DecryptionKey::<{ U128::LIMBS }>::new(&mut rng);
        check_enc_dec_given_dec_key!(num_iters, 4, U128, U256, dk);
    }

    #[test]
    fn encrypt_decrypt_s_2_1024_bit_prime() {
        let num_iters = 10;
        let (p, q) = get_1024_bit_primes();
        let dk = DecryptionKey::from_primes(p, q);
        check_enc_dec_given_dec_key!(num_iters, 2, U1024, U2048, dk);
    }

    #[test]
    fn encrypt_decrypt_s_3_1024_bit_prime() {
        let num_iters = 10;
        let (p, q) = get_1024_bit_primes();
        let dk = DecryptionKey::from_primes(p, q);
        check_enc_dec_given_dec_key!(num_iters, 3, U1024, U2048, dk);
    }

    #[test]
    fn encrypt_decrypt_s_4_1024_bit_prime() {
        let num_iters = 10;
        let (p, q) = get_1024_bit_primes();
        let dk = DecryptionKey::from_primes(p, q);
        check_enc_dec_given_dec_key!(num_iters, 4, U1024, U2048, dk);
    }

    #[test]
    fn ciphertext_add_mul_s_2_1024_bit_prime() {
        let num_iters = 10;
        let (p, q) = get_1024_bit_primes();
        let dk = DecryptionKey::from_primes(p, q);
        check_ops_given_dec_key!(num_iters, 2, U1024, U2048, dk);
    }

    #[test]
    fn encrypt_decrypt_s_2_2048_bit_prime() {
        let num_iters = 10;
        let (p, q) = get_2048_bit_primes();
        let dk = DecryptionKey::from_primes(p, q);
        check_enc_dec_given_dec_key!(num_iters, 2, U2048, U4096, dk);
    }

    #[test]
    fn encrypt_decrypt_s_3_2048_bit_prime() {
        let num_iters = 10;
        let (p, q) = get_2048_bit_primes();
        let dk = DecryptionKey::from_primes(p, q);
        check_enc_dec_given_dec_key!(num_iters, 3, U2048, U4096, dk);
    }

    #[test]
    fn encrypt_decrypt_s_4_2048_bit_prime() {
        let num_iters = 10;
        let (p, q) = get_2048_bit_primes();
        let dk = DecryptionKey::from_primes(p, q);
        check_enc_dec_given_dec_key!(num_iters, 4, U2048, U4096, dk);
    }
}
