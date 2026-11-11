//! Тесты без GPU: матрицы самотеста (как в src/pearl.h и ref/selftest_gen.py),
//! удачи/промахи тайлов взяты из эталона ref/pearlhash_ref.py.
use super::*;
use std::io::{BufRead, BufReader};
use std::net::TcpListener;

const SEED_A: u32 = 0xA11CE;
const SEED_B: u32 = 0xB0B;
const SEED_N: u64 = 0x5EED;
const M: usize = 256;
const N: usize = 128;
const K: usize = 4096;
const NBITS: u32 = 0x1e10_0000;

fn gen_val(seed: u32, idx: u32) -> i8 {
    let mut x = idx.wrapping_mul(0x9E37_79B1).wrapping_add(seed);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2_AE35);
    x ^= x >> 16;
    ((((x >> 8) * 129) >> 24) as i32 - 64) as i8
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn matrices() -> (Vec<i8>, Vec<i8>) {
    let mut a: Vec<i8> = (0..M * K).map(|i| gen_val(SEED_A, i as u32)).collect();
    for (i, v) in a.iter_mut().take(1024).enumerate() {
        *v = ((splitmix(SEED_N + i as u64) % 129) as i32 - 64) as i8; // nonce 0
    }
    let bt: Vec<i8> = (0..N * K).map(|i| gen_val(SEED_B, i as u32)).collect();
    (a, bt)
}

fn header() -> [u8; 76] {
    let mut h = [0u8; 76];
    h[..4].copy_from_slice(&1u32.to_le_bytes());
    for i in 0..32 {
        h[4 + i] = (31 - i) as u8; // prev_block[i] = i, в заголовке — в обратном порядке
        h[36 + i] = 0x80 + (31 - i) as u8;
    }
    h[68..72].copy_from_slice(&0x6a00_0000u32.to_le_bytes());
    h[72..76].copy_from_slice(&NBITS.to_le_bytes());
    h
}

fn tile(r0: u32, c0: u32) -> (Vec<u32>, Vec<u32>) {
    let rows = (0..8).map(|i| r0 + 8 * i).collect();
    let cols = (0..8).flat_map(|j| [c0 + 8 * j, c0 + 8 * j + 1]).collect();
    (rows, cols)
}

#[test]
fn matrices_match_reference() {
    let (a, bt) = matrices();
    assert_eq!(&a[..8], &[-15, -55, 14, 38, 4, -64, -21, -36]);
    assert_eq!(&a[1030..1034], &[44, -14, -27, 61]);
    assert_eq!(&bt[..6], &[-44, 38, 48, 38, 15, 15]);
}

#[test]
fn proof_hit_accepted_miss_rejected() {
    let (a, bt) = matrices();
    for (r0, c0, hit) in [(0, 2, true), (0, 64, true), (0, 0, false), (1, 2, false)] {
        let (rows, cols) = tile(r0, c0);
        let (h, pp) = proof::build(&header(), &a, M, &bt, N, K, 128, &rows, &cols).unwrap();
        let res = proof::verify(&h, &pp, 3, None);
        assert_eq!(res.is_ok(), hit, "тайл ({r0},{c0}): {res:?}");
        // сериализация туда-обратно, с gzip и без
        for gz in [false, true] {
            let b64 = proof::encode(&pp, gz).unwrap();
            let back = proof::decode(&b64, gz).unwrap();
            assert_eq!(proof::verify(&h, &back, 3, None).is_ok(), hit);
        }
    }
}

#[test]
fn nbits_ceil_not_stricter() {
    use zk_pow::api::proof_utils::nbits_to_difficulty;
    for nb in [0x1e10_0000u32, 0x1c02_0000, 0x1d00_ffff, 0x2000_013b] {
        let t = nbits_to_difficulty(nb);
        assert_eq!(nbits_to_difficulty(target_to_nbits_ceil(t)), t, "{nb:08x}");
        let t2 = t + U256::from(12345u32);
        assert!(nbits_to_difficulty(target_to_nbits_ceil(t2)) >= t2);
    }
}

#[test]
fn url_parsing() {
    assert_eq!(parse_url("stratum+tcp://a.b:3333").unwrap(), ("a.b".into(), 3333, false));
    assert_eq!(parse_url("stratum+ssl://x.y:443").unwrap(), ("x.y".into(), 443, true));
    assert_eq!(parse_url("prl.kryptex.network:7048").unwrap(), ("prl.kryptex.network".into(), 7048, false));
}

/// Поддельный пул на localhost: задание самотеста → шар от клиента → проверка → «принят».
#[test]
fn stratum_end_to_end_with_fake_pool() {
    let srv = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = srv.local_addr().unwrap().port();
    let target = zk_pow::api::proof_utils::nbits_to_difficulty(NBITS);
    let pool = std::thread::spawn(move || {
        let (sock, _) = srv.accept().unwrap();
        let mut w = sock.try_clone().unwrap();
        let mut r = BufReader::new(sock);
        let mut line = String::new();
        r.read_line(&mut line).unwrap();
        let auth: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(auth["method"], "mining.authorize");
        assert_eq!(auth["params"]["wallet"], "prl1test.rig");
        writeln!(w, r#"{{"id":1,"result":true,"error":null,"type":"v2"}}"#).unwrap();
        let mut tb = [0u8; 32];
        target.to_big_endian(&mut tb);
        let hex: String = tb.iter().map(|b| format!("{b:02x}")).collect();
        let hh: String = header().iter().map(|b| format!("{b:02x}")).collect();
        writeln!(w, r#"{{"method":"mining.notify","params":{{"job_id":"j1","header":"{hh}","height":7,"target":"{hex}"}}}}"#).unwrap();
        line.clear();
        r.read_line(&mut line).unwrap();
        let sub: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(sub["method"], "mining.submit");
        assert_eq!(sub["params"]["job_id"], "j1");
        let pp = proof::decode(sub["params"]["plain_proof"].as_str().unwrap(), true).unwrap();
        let h = zk_pow::api::proof::IncompleteBlockHeader::from_bytes(&header()).unwrap();
        let ok = proof::verify(&h, &pp, 3, None).is_ok();
        writeln!(w, r#"{{"id":{},"result":{ok},"error":null}}"#, sub["id"]).unwrap();
        std::thread::sleep(Duration::from_millis(500));
        ok
    });
    let url = std::ffi::CString::new(format!("127.0.0.1:{port}")).unwrap();
    let user = std::ffi::CString::new("prl1test.rig").unwrap();
    let x = std::ffi::CString::new("x").unwrap();
    unsafe { assert_eq!(pm_pool_start(url.as_ptr(), user.as_ptr(), x.as_ptr(), x.as_ptr()), 0) };
    let mut job = PmJob { serial: 0, header: [0; 76], target_le: [0; 32], height: 0, cert_version: 0, job_id: [0; 64] };
    let mut serial = 0;
    for _ in 0..100 {
        serial = unsafe { pm_pool_job(&mut job) };
        if serial != 0 { break }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(serial, 1);
    assert_eq!(job.header, header());
    assert_eq!(job.cert_version, 3);
    assert_eq!(U256::from_little_endian(&job.target_le), target);
    let (a, bt) = matrices();
    let (rows, cols) = tile(0, 2);
    unsafe {
        pm_pool_submit(&job, a.as_ptr(), M, bt.as_ptr(), N, K, 128, rows.as_ptr(), 8, cols.as_ptr(), 16, 0);
    }
    assert!(pool.join().unwrap(), "поддельный пул отклонил доказательство");
    let (mut acc, mut rej, mut st, mut c) = (0, 0, 0, 0);
    unsafe { pm_pool_stats(&mut acc, &mut rej, &mut st, &mut c) };
    assert_eq!((acc, rej), (1, 0));
}
