use core::ops::{Div, Sub};
use crypto_bigint::{
    modular::{MontyForm, MontyParams, SafeGcdInverter},
    BoxedUint, Concat, NonZero, Odd, PrecomputeInverter, Uint, WideningMul, Zero,
};
use crypto_primes::generate_prime_with_rng;
use rand_core::CryptoRngCore;

/// Blum primes are congruent to 3 mod 4, and they allow for efficient proof that a composite number can be
/// factored into Blum primes. Safe primes are also Blum primes.
pub fn blum_prime<R: CryptoRngCore, const PRIME_LIMBS: usize>(rng: &mut R) -> Uint<PRIME_LIMBS> {
    let mut p: Uint<PRIME_LIMBS> = generate_prime_with_rng(rng, Uint::<PRIME_LIMBS>::BITS);
    while p.as_limbs()[0].0 & 3 != 3 {
        p = generate_prime_with_rng(rng, Uint::<PRIME_LIMBS>::BITS);
    }
    p
}

/// Returns lower common multiple of given numbers.
/// Relies on the fact that `gcd(a,b) * lcm(a,b) = a*b`
pub fn lcm<const LIMBS: usize, const WIDE_LIMBS: usize, const UNSAT_LIMBS: usize>(
    a: &Uint<LIMBS>,
    b: &Uint<LIMBS>,
) -> Uint<WIDE_LIMBS>
where
    Uint<LIMBS>: Concat<Output = Uint<WIDE_LIMBS>>,
    Odd<Uint<LIMBS>>: PrecomputeInverter<Inverter = SafeGcdInverter<LIMBS, UNSAT_LIMBS>>,
{
    let d = a.gcd(&b).resize::<WIDE_LIMBS>();
    let c = a.widening_mul(b);
    // c always divides d
    c.div(&d.to_nz().unwrap())
}

/// Returns Carmichael's totient of given primes which is `lcm(p-1, q-1)`
pub fn carmichael_totient<
    const PRIME_LIMBS: usize,
    const OUTPUT_LIMBS: usize,
    const PRIME_UNSAT_LIMBS: usize,
>(
    p: &Uint<PRIME_LIMBS>,
    q: &Uint<PRIME_LIMBS>,
) -> Uint<OUTPUT_LIMBS>
where
    Uint<PRIME_LIMBS>: Concat<Output = Uint<OUTPUT_LIMBS>>,
    Odd<Uint<PRIME_LIMBS>>:
        PrecomputeInverter<Inverter = SafeGcdInverter<PRIME_LIMBS, PRIME_UNSAT_LIMBS>>,
{
    lcm::<PRIME_LIMBS, OUTPUT_LIMBS, PRIME_UNSAT_LIMBS>(
        &p.wrapping_sub(&Uint::<PRIME_LIMBS>::ONE),
        &q.wrapping_sub(&Uint::<PRIME_LIMBS>::ONE),
    )
}

/// Returns Euler's totient of given primes which is `(p-1)*(q-1)`
pub fn euler_totient<const PRIME_LIMBS: usize, const OUTPUT_LIMBS: usize>(
    p: &Uint<PRIME_LIMBS>,
    q: &Uint<PRIME_LIMBS>,
) -> Uint<OUTPUT_LIMBS>
where
    Uint<PRIME_LIMBS>: Concat<Output = Uint<OUTPUT_LIMBS>>,
{
    p.wrapping_sub(&Uint::<PRIME_LIMBS>::ONE)
        .widening_mul(&q.wrapping_sub(&Uint::<PRIME_LIMBS>::ONE))
}

/// Ensure the given value can fit in `max_limbs` limbs by ensuring the remaining most significant limbs to be 0
pub fn ensure_upper_limbs_0<const L: usize>(val: &Uint<L>, max_limbs: usize) {
    for (i, l_i) in val.as_limbs().iter().enumerate() {
        if i >= max_limbs {
            assert!(bool::from(l_i.is_zero()));
        }
    }
}

/// Return `(u-1)/n` and assumes `u-1` to be a multiple of `n`. This is the logarithm function
/// for `(1+n)^m mod n^2` and returns the logarithm `m` as `(1+n)^m mod n^2 = 1 + m*n` and `l(1 + m*n) = m`
pub fn l<const U: usize, const N: usize, const O: usize>(
    u: &Uint<U>,
    n: &NonZero<Uint<N>>,
) -> Uint<O> {
    // u-1
    let m = u.sub(Uint::ONE);
    let n_wide = n.resize();
    let (m, r) = m.div_rem(&n_wide.to_nz().unwrap());
    debug_assert!(bool::from(r.is_zero()));
    ensure_upper_limbs_0(&m, O);
    m.resize::<O>()
}

/// Given `r_p` and `r_q` satisfying `r = r_p mod p` and `r = r_q mod q` for primes `p` and `q`,
/// return `r mod p*q` using the Chinese Remainder Theorem. Works as follows:
/// `r = r_p mod p` => `r = r_p + t*p` and substitute for `r` in `r = r_q mod q` to get `r_p + t*p = r_q mod q`.
/// Calculate `t` as `t = (r_q - r_p)/p mod q` and substitute `t` in `r = r_p + t*p` to get `r = r_p + ((r_q - r_p)/p mod q)*p`
pub fn crt_combine<const LIMBS: usize, const WIDE_LIMBS: usize>(
    r_p: &Uint<LIMBS>,
    r_q: &Uint<LIMBS>,
    p_inv: MontyForm<LIMBS>,
    p: &Uint<LIMBS>,
    q_mtg: MontyParams<LIMBS>,
) -> Uint<WIDE_LIMBS>
where
    Uint<LIMBS>: Concat<Output = Uint<WIDE_LIMBS>>,
{
    // r_q - r_p mod q
    let diff = MontyForm::new(r_q, q_mtg) - MontyForm::new(r_p, q_mtg);
    // (r_q - r_p)/p mod q
    let t = (diff * p_inv).retrieve();
    // ((r_q - r_p)/p mod q) * p + r_p
    t.widening_mul(p) + r_p.resize()
}

/// Same as `crt_combine` above but doesn't require the trait Concat to be implemented for `Uint<LIMBS>` and
/// thus uses BoxedUint
pub fn crt_combine_using_box<const LIMBS: usize, const WIDE_LIMBS: usize>(
    r_p: &Uint<LIMBS>,
    r_q: &Uint<LIMBS>,
    p_inv: MontyForm<LIMBS>,
    p: &Uint<LIMBS>,
    q_mtg: MontyParams<LIMBS>,
) -> Uint<WIDE_LIMBS> {
    // r_q - r_p mod q
    let diff = MontyForm::new(r_q, q_mtg) - MontyForm::new(r_p, q_mtg);
    // (r_q - r_p)/p mod q
    let t = (diff * p_inv).retrieve();
    let t_box = BoxedUint::from(t);
    let m_box = t_box.widening_mul(BoxedUint::from(p)) + BoxedUint::from(r_p);
    Uint::<WIDE_LIMBS>::from_le_slice(m_box.to_le_bytes().as_ref())
}

#[cfg(feature = "parallel")]
#[macro_export]
macro_rules! join {
    ($a: expr, $b: expr) => {
        rayon::join(|| $a, || $b)
    };
}

#[cfg(not(feature = "parallel"))]
#[macro_export]
macro_rules! join {
    ($a: expr, $b: expr) => {
        ($a, $b)
    };
}

#[cfg(test)]
pub fn timing_info(mut times: Vec<std::time::Duration>) -> String {
    // Given timings of an operation repeated several times, prints the total time takes, least time,
    // median time and the highest time
    times.sort();
    let median = {
        let mid = times.len() / 2;
        if times.len() % 2 == 0 {
            (times[mid - 1] + times[mid]) / 2
        } else {
            times[mid]
        }
    };
    let total = times.iter().sum::<std::time::Duration>();
    format!(
        "{:.2?} | [{:.2?}, {:.2?}, {:.2?}]",
        total,
        times[0],
        median,
        times[times.len() - 1]
    )
}

#[cfg(test)]
pub fn get_1024_bit_primes() -> (Odd<crypto_bigint::U1024>, Odd<crypto_bigint::U1024>) {
    let p = crypto_bigint::U1024::from_str_radix_vartime("148677972634832330983979593310074301486537017973460461278300587514468301043894574906886127642530475786889672304776052879927627556769456140664043088700743909632312483413393134504352834240399191134336344285483935856491230340093391784574980688823380828143810804684752914935441384845195613674104960646037368551517", 10).unwrap();
    let q = crypto_bigint::U1024::from_str_radix_vartime("158741574437007245654463598139927898730476924736461654463975966787719309357536545869203069369466212089132653564188443272208127277664424448947476335413293018778018615899291704693105620242763173357203898195318179150836424196645745308205164116144020613415407736216097185962171301808761138424668335445923774195463", 10).unwrap();
    (p.to_odd().unwrap(), q.to_odd().unwrap())
}

#[cfg(test)]
pub fn get_2048_bit_primes() -> (Odd<crypto_bigint::U2048>, Odd<crypto_bigint::U2048>) {
    let p = crypto_bigint::U2048::from_str_radix_vartime("29714581929123975538113401757096867247503888049897126155282036684655427098443105525014011037627595171636270743123002658539126362781975975175765337944068032414914877908601576682891727277414354084913151212699556099504403364557952921342801004492280996715668400103640816843970991636313372745470315455035628601408170417079028041375322988613489555126184463766534396540607235696364068780046050136089443239241198755363075399416619880240793665666130686930042641834471008631848126179567943667666801104241898884410812817279169595932728564045398540809381698710218625876508851295613368971979430951746728583910413116439939078434559", 10).unwrap();
    let q = crypto_bigint::U2048::from_str_radix_vartime("26092039125439665744416238260398697435648406017098864449978544271624805738059383134259926966553183020513772201496445138041100372380433951022528474017361803675300903527015075913474169512090459118347512405005520042799270078794768712536842118407057375282490800716584000679340618387331368881454328913585366779623070416709172563900009042884661367457056955039492864910532308631507979947887262253149026114965531208152102534718129699718880396068707567121640888946634505008511577162806565588378302758525657914103598229285420198323100812024493357088882840233483389168400067067993178810813818498522088103582183754940421621446417", 10).unwrap();
    (p.to_odd().unwrap(), q.to_odd().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::{
        modular::{MontyForm, MontyParams},
        Random, RandomMod, U128, U256, U64,
    };
    use crypto_primes::is_prime_with_rng;
    use num_traits::identities::One;
    use rand_core::OsRng;
    use std::time::Instant;

    #[test]
    fn math_properties() {
        let mut rng = OsRng::default();
        let p = generate_prime_with_rng::<U64>(&mut rng, U64::BITS);
        let q = generate_prime_with_rng::<U64>(&mut rng, U64::BITS);
        let n = p.widening_mul(&q);
        let n2 = n.square();
        let p2 = p.square();

        let params_n = MontyParams::new(n.to_odd().unwrap());
        let params_n2 = MontyParams::new(n2.to_odd().unwrap());
        let params_p2 = MontyParams::new(p2.to_odd().unwrap());
        let phi = euler_totient(&p, &q);
        let phi_n2 = n.widening_mul(&phi);
        let phi_p2 = p.square().wrapping_sub(&Uint::ONE);

        for _ in 0..100 {
            let r = U128::random_mod(&mut rng, &n.to_nz().unwrap());
            let r_mon = MontyForm::new(&r, params_n);
            let r1 = r_mon.pow(&phi).retrieve();
            println!("r^phi mod n {}", r1);
            assert!(r1.is_one());

            let r_big = r.resize();
            let r_big_mon = MontyForm::new(&r_big, params_n2);
            let r2 = r_big_mon.pow(&phi).retrieve();
            println!("r^phi mod n^2 {}", r2);

            let r3 = r_big_mon
                .pow(&phi)
                .retrieve()
                .rem(&n.resize().to_nz().unwrap());
            println!("r^phi mod n^2 mod n {}", r3);
            assert!(r3.is_one());

            let start = Instant::now();
            let a = U256::random_mod(&mut rng, &n2.to_nz().unwrap());
            let x = U256::random_mod(&mut rng, &n2.to_nz().unwrap());
            let a_x = MontyForm::new(&a, params_n2).pow(&x).retrieve();
            println!("a^x mod n^2 in {:?}", start.elapsed());

            let start = Instant::now();
            // x mod phi(n^2)
            let x_phi_n2 = x.rem(&phi_n2.to_nz().unwrap());
            let a_x_1 = MontyForm::new(&a, params_n2).pow(&x_phi_n2).retrieve();
            println!("a^(x mod phi(n^2)) mod n^2 in {:?}", start.elapsed());
            assert_eq!(a_x, a_x_1);

            let start = Instant::now();
            let a = U128::random_mod(&mut rng, &p2.to_nz().unwrap());
            let x = U128::random_mod(&mut rng, &p2.to_nz().unwrap());
            let a_x = MontyForm::new(&a, params_p2).pow(&x).retrieve();
            println!("a^x mod p^2 in {:?}", start.elapsed());

            let start = Instant::now();
            let x_phi_p2 = x.rem(&phi_p2.to_nz().unwrap());
            let a_x_1 = MontyForm::new(&a, params_p2).pow(&x_phi_p2).retrieve();
            println!("a^(x mod phi(p^2)) mod p^2 in {:?}", start.elapsed());
            assert_eq!(a_x, a_x_1);

            let x_p_minus_1 = x.rem(&p.wrapping_sub(&Uint::ONE).resize().to_nz().unwrap());
            let a_x_2 = MontyForm::new(&a, params_p2).pow(&x_p_minus_1).retrieve();
            println!("a^(x mod p-1) mod p^2 {}", a_x_2);
            assert_ne!(a_x, a_x_2);

            // a^(x mod p-1) mod p^2 mod p
            let a_x_3 = a_x_2.rem(&p.resize().to_nz().unwrap());
            // a^x mod p^2 mod p
            let a_x_4 = a_x.rem(&p.resize().to_nz().unwrap());
            assert_eq!(a_x_4, a_x_3);
        }
    }

    #[test]
    fn prime() {
        let mut rng = OsRng::default();
        for _ in 0..10 {
            let p = blum_prime::<_, { U64::LIMBS }>(&mut rng);
            let words = p.as_words();
            assert!(is_prime_with_rng::<U64>(&mut rng, &p));
            assert_eq!(words[0] % 4, 3);
        }
    }

    #[test]
    fn crt() {
        let mut rng = OsRng::default();

        macro_rules! check {
            ( $prime_type:ident, $product_type:ident ) => {
                let p = generate_prime_with_rng::<$prime_type>(&mut rng, $prime_type::BITS);
                let q = generate_prime_with_rng::<$prime_type>(&mut rng, $prime_type::BITS);
                let n = p.widening_mul(&q);
                let q_mtg = MontyParams::new(q.to_odd().unwrap());
                // p^-1 mod q
                let p_inv = MontyForm::new(&p, q_mtg).inv().unwrap();
                for _ in 0..100 {
                    let a = $product_type::random_mod(&mut rng, &n.to_nz().unwrap());
                    // a_p = a mod p
                    let a_p: $prime_type = a.rem(&p.resize().to_nz().unwrap()).resize();
                    // a_q = a mod q
                    let a_q: $prime_type = a.rem(&q.resize().to_nz().unwrap()).resize();
                    assert_eq!(
                        a,
                        crt_combine::<{ $prime_type::LIMBS }, { $product_type::LIMBS }>(
                            &a_p, &a_q, p_inv, &p, q_mtg
                        )
                    );
                    assert_eq!(a, crt_combine_using_box(&a_p, &a_q, p_inv, &p, q_mtg));
                }
            };
        }
        check!(U64, U128);
        check!(U128, U256);
    }

    #[test]
    fn lcm_check() {
        let mut rng = OsRng::default();
        for _ in 0..10 {
            let a = U64::random(&mut rng);
            let b = U64::random(&mut rng);
            let g: U64 = a.gcd(&b);
            let l: U128 = lcm(&a, &b);
            assert_eq!(
                g.widening_mul(&l),
                a.widening_mul(&b).resize(),
                "failed on {} {}",
                a,
                b
            );
        }
    }
}
