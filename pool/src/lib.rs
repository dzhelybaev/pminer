//! pminer-pool: построение PlainProof официальным кодом Pearl (zk-pow, pearl-blake3),
//! локальная проверка официальным верификатором и stratum-клиент пула
//! (диалект «объектов»: Suprnova / Herominers / Kryptex, вариант v2 с gzip).
//! Вызывается из CUDA-части через C ABI (см. src/pool.h).

use std::ffi::{CStr, c_char};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use primitive_types::U256;

macro_rules! plog {
    ($($a:tt)*) => {{ println!("[пул] {}", format!($($a)*)); let _ = std::io::stdout().flush(); }};
}

// ======================= доказательство =======================
pub mod proof {
    use super::*;
    use pearl_blake3::{MerkleTree, blake3_digest, pad_to_chunk_boundary};
    use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern};
    use zk_pow::api::verify::verify_plain_proof;
    use zk_pow::ffi::plain_proof::{MatrixMerkleProof, PlainProof, check_cert_version_eligible};

    fn rel_pattern(idx: &[u32]) -> Result<PeriodicPattern> {
        ensure!(!idx.is_empty(), "пустой список индексов");
        let o = idx[0];
        PeriodicPattern::from_list(&idx.iter().map(|&x| x - o).collect::<Vec<_>>())
    }

    fn matrix_proof(mat: &[i8], rows_n: usize, k: usize, jk: &[u8; 32], idx: &[u32]) -> MatrixMerkleProof {
        let bytes: Vec<u8> = mat.iter().map(|&x| x as u8).collect();
        let padded = pad_to_chunk_boundary(&bytes);
        let tree = MerkleTree::new(&padded, *jk);
        let rows: Vec<usize> = idx.iter().map(|&x| x as usize).collect();
        let leaves = MerkleTree::compute_leaf_indices_from_rows(&rows, (rows_n, k));
        MatrixMerkleProof { proof: tree.get_multileaf_proof(&leaves), row_indices: rows }
    }

    /// PlainProof для тайла: строки rows матрицы A (m×k), строки cols матрицы B^T (n×k).
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        header76: &[u8], a: &[i8], m: usize, bt: &[i8], n: usize, k: usize, rank: usize, rows: &[u32],
        cols: &[u32],
    ) -> Result<(IncompleteBlockHeader, PlainProof)> {
        ensure!(a.len() == m * k && bt.len() == n * k, "размеры матриц не совпадают");
        let header = IncompleteBlockHeader::from_bytes(header76)?;
        let cfg = MiningConfiguration {
            common_dim: k as u32,
            rank: rank as u16,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: rel_pattern(rows)?,
            cols_pattern: rel_pattern(cols)?,
            moe: None,
        };
        let mut d = header.to_bytes().to_vec();
        d.extend_from_slice(&cfg.to_bytes());
        let jk = blake3_digest(&d, None);
        let pp = PlainProof {
            m,
            n,
            k,
            noise_rank: rank,
            a: matrix_proof(a, m, k, &jk, rows),
            bt: matrix_proof(bt, n, k, &jk, cols),
            moe: None,
        };
        Ok((header, pp))
    }

    /// Официальная проверка (blake3, без plonky2). nbits_override — сложность шара.
    pub fn verify(header: &IncompleteBlockHeader, pp: &PlainProof, cert_version: u32, nbits: Option<u32>) -> Result<()> {
        let v = check_cert_version_eligible(cert_version, pp)?;
        verify_plain_proof(header, pp, nbits, v.seed_derivation())
    }

    /// base64(bincode) или base64(gzip(bincode)) — как PlainProof.to_base64 в pearl_mining.
    pub fn encode(pp: &PlainProof, gzip: bool) -> Result<String> {
        use base64::Engine;
        let mut bytes = bincode::serialize(pp)?;
        if gzip {
            let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
            e.write_all(&bytes)?;
            bytes = e.finish()?;
        }
        Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
    }

    pub fn decode(b64: &str, gzip: bool) -> Result<PlainProof> {
        use base64::Engine;
        let mut bytes = base64::engine::general_purpose::STANDARD.decode(b64.trim())?;
        if gzip {
            let mut d = flate2::read::GzDecoder::new(&bytes[..]);
            let mut out = Vec::new();
            d.read_to_end(&mut out)?;
            bytes = out;
        }
        PlainProof::deserialize_compat(&bytes)
    }
}

/// Компактная форма (nbits) для цели, округлённая вверх: проверка с ней не строже цели пула.
pub fn target_to_nbits_ceil(t: U256) -> u32 {
    if t.is_zero() {
        return 0x0300_0001;
    }
    let mut size = (t.bits() as u32).div_ceil(8);
    let mut mant: U256;
    let mut lost = false;
    if size <= 3 {
        mant = t << (8 * (3 - size) as usize);
    } else {
        let sh = 8 * (size - 3) as usize;
        mant = t >> sh;
        lost = (mant << sh) != t;
    }
    if lost {
        mant += U256::one();
    }
    // бит знака 0x800000 недопустим, переполнение 24 бит — сдвигаем ещё на байт
    while mant.low_u32() & 0x80_0000 != 0 || mant > U256::from(0xff_ffffu32) {
        let low = mant.low_u32() & 0xff != 0;
        mant >>= 8;
        if low {
            mant += U256::one();
        }
        size += 1;
    }
    (size << 24) | mant.low_u32()
}

// ======================= stratum =======================
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PmJob {
    pub serial: u64,
    pub header: [u8; 76],
    pub target_le: [u8; 32],
    pub height: u64,
    pub cert_version: u32,
    pub job_id: [c_char; 64],
}

#[derive(Default)]
struct Shared {
    job: Option<PmJob>,
    serial: u64,
    accepted: u64,
    rejected: u64,
    stale: u64,
    connected: bool,
    gzip: bool,
}

struct Submit {
    job_id: String,
    proof_b64: String,
    label: String,
}

struct Pool {
    shared: Arc<Mutex<Shared>>,
    tx: Mutex<mpsc::Sender<Submit>>,
}

static POOL: OnceLock<Pool> = OnceLock::new();

trait Rw: Read + Write + Send {}
impl<T: Read + Write + Send> Rw for T {}

pub fn parse_url(url: &str) -> Result<(String, u16, bool)> {
    let mut u = url.trim();
    let mut tls = false;
    for (p, t) in [("stratum+tcp://", false), ("stratum+ssl://", true), ("stratum+tls://", true), ("ssl://", true),
                   ("tls://", true), ("tcp://", false), ("stratum://", false)] {
        if let Some(rest) = u.strip_prefix(p) {
            u = rest;
            tls = t;
            break;
        }
    }
    let (h, p) = u.rsplit_once(':').context("в адресе пула нет порта")?;
    Ok((h.to_string(), p.trim_end_matches('/').parse()?, tls))
}

fn connect(host: &str, port: u16, tls: bool) -> Result<Box<dyn Rw>> {
    let addr = (host, port).to_socket_addrs()?.next().context("адрес пула не найден")?;
    let tcp = TcpStream::connect_timeout(&addr, Duration::from_secs(10))?;
    tcp.set_nodelay(true)?;
    tcp.set_read_timeout(Some(Duration::from_millis(100)))?;
    if !tls {
        return Ok(Box::new(tcp));
    }
    let roots = rustls::RootCertStore { roots: webpki_roots::TLS_SERVER_ROOTS.into() };
    let cfg = rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions()?
        .with_root_certificates(roots)
        .with_no_client_auth();
    let name = rustls::pki_types::ServerName::try_from(host.to_string())?;
    let conn = rustls::ClientConnection::new(Arc::new(cfg), name)?;
    Ok(Box::new(rustls::StreamOwned::new(conn, tcp)))
}

fn json_id(v: &serde_json::Value) -> Option<u64> {
    v.get("id").and_then(|x| x.as_u64().or_else(|| x.as_str().and_then(|s| s.parse().ok())))
}

fn err_text(e: &serde_json::Value) -> (i64, String) {
    match e {
        serde_json::Value::Object(o) => (
            o.get("code").and_then(|c| c.as_i64()).unwrap_or(0),
            o.get("msg").or(o.get("message")).map(|m| m.to_string()).unwrap_or_default(),
        ),
        serde_json::Value::Array(a) => (
            a.first().and_then(|c| c.as_i64()).unwrap_or(0),
            a.get(1).map(|m| m.to_string()).unwrap_or_default(),
        ),
        other => (0, other.to_string()),
    }
}

/// notify в диалекте объектов: {job_id, header(hex 76 Б), target(hex U256, big-endian), height[, cert_version]}
fn parse_notify(p: &serde_json::Value) -> Result<PmJob> {
    let o = p.as_object().context("params не объект (позиционный диалект не поддерживается)")?;
    let job_id = match o.get("job_id").context("нет job_id")? {
        serde_json::Value::String(s) => s.clone(),
        v => v.to_string(),
    };
    let hh = o.get("header").and_then(|v| v.as_str()).context("нет header")?;
    let header = hex_bytes(hh)?;
    ensure!(header.len() == 76, "header: {} байт вместо 76", header.len());
    let th = o.get("target").and_then(|v| v.as_str()).context("нет target")?;
    let tb = hex_bytes(th)?;
    ensure!(tb.len() <= 32, "target длиннее 32 байт");
    let t = U256::from_big_endian(&tb);
    let mut job = PmJob { serial: 0, header: [0; 76], target_le: [0; 32], height: 0, cert_version: 3, job_id: [0; 64] };
    job.header.copy_from_slice(&header);
    t.to_little_endian(&mut job.target_le);
    job.height = o.get("height").and_then(|v| v.as_u64()).unwrap_or(0);
    if let Some(cv) = o.get("cert_version").and_then(|v| v.as_u64()) {
        job.cert_version = cv as u32;
    }
    for (i, b) in job_id.bytes().take(63).enumerate() {
        job.job_id[i] = b as c_char;
    }
    Ok(job)
}

fn cstr_arr(a: &[c_char]) -> String {
    a.iter().take_while(|&&c| c != 0).map(|&c| c as u8 as char).collect()
}

fn hex_bytes(s: &str) -> Result<Vec<u8>> {
    let s = s.trim().trim_start_matches("0x");
    let s = if s.len() % 2 == 1 { format!("0{s}") } else { s.to_string() };
    (0..s.len()).step_by(2).map(|i| Ok(u8::from_str_radix(&s[i..i + 2], 16)?)).collect()
}

fn session(url: &str, user: &str, pass: &str, agent: &str, sh: &Arc<Mutex<Shared>>, rx: &mpsc::Receiver<Submit>) -> Result<()> {
    let (host, port, tls) = parse_url(url)?;
    let mut s = connect(&host, port, tls)?;
    plog!("подключено к {host}:{port}{}", if tls { " (TLS)" } else { "" });
    let worker = user.split_once('.').map(|(_, w)| w).filter(|w| !w.is_empty()).unwrap_or("default");
    let auth = serde_json::json!({"id": 1, "method": "mining.authorize",
        "params": {"wallet": user, "worker": worker, "pass": pass, "agent": agent, "type": "v2"}});
    s.write_all(format!("{auth}\n").as_bytes())?;
    s.flush()?;
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 65536];
    let mut next_id = 100u64;
    let mut pending: std::collections::HashMap<u64, String> = Default::default();
    let mut warned_positional = false;
    loop {
        // отправка шаров
        while let Ok(sub) = rx.try_recv() {
            let id = next_id;
            next_id += 1;
            let msg = serde_json::json!({"id": id, "method": "mining.submit",
                "params": {"job_id": sub.job_id, "plain_proof": sub.proof_b64}});
            s.write_all(format!("{msg}\n").as_bytes())?;
            s.flush()?;
            pending.insert(id, sub.label);
        }
        // приём
        match s.read(&mut tmp) {
            Ok(0) => bail!("пул закрыл соединение"),
            Ok(nr) => buf.extend_from_slice(&tmp[..nr]),
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => continue,
            Err(e) => return Err(e.into()),
        }
        ensure!(buf.len() <= 8 << 20, "слишком длинная строка от пула");
        while let Some(pos) = buf.iter().position(|&c| c == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let Ok(v) = serde_json::from_slice::<serde_json::Value>(&line) else { continue };
            let method = v.get("method").and_then(|m| m.as_str()).unwrap_or("");
            if method == "mining.notify" {
                match parse_notify(&v["params"]) {
                    Ok(mut job) => {
                        let mut g = sh.lock().unwrap();
                        g.serial += 1;
                        job.serial = g.serial;
                        let id = cstr_arr(&job.job_id);
                        plog!("задание {id}, высота {}, cert v{}", job.height, job.cert_version);
                        g.job = Some(job);
                    }
                    Err(e) => {
                        if !warned_positional {
                            plog!("не разобрал задание: {e}");
                            warned_positional = true;
                        }
                    }
                }
                continue;
            }
            if !method.is_empty() {
                continue; // прочие уведомления (set_difficulty и т.п.) в этом диалекте не нужны
            }
            let Some(id) = json_id(&v) else { continue };
            let err = v.get("error").filter(|e| !e.is_null());
            if id == 1 {
                if let Some(e) = err {
                    let (c, m) = err_text(e);
                    bail!("авторизация отклонена: {c} {m}");
                }
                let v2 = v.get("type").and_then(|t| t.as_str()) == Some("v2")
                    || v.get("result").and_then(|r| r.get("type")).and_then(|t| t.as_str()) == Some("v2");
                let mut g = sh.lock().unwrap();
                g.connected = true;
                g.gzip = v2;
                plog!("авторизован{}", if v2 { ", протокол v2 (gzip)" } else { "" });
            } else if let Some(label) = pending.remove(&id) {
                let mut g = sh.lock().unwrap();
                match err {
                    None if v.get("result").map(|r| r.as_bool() != Some(false)).unwrap_or(true) => {
                        g.accepted += 1;
                        plog!("{label}: ПРИНЯТ (всего {})", g.accepted);
                    }
                    e => {
                        let (c, m) = e.map(err_text).unwrap_or((0, "result=false".into()));
                        if c == 21 { g.stale += 1 } else { g.rejected += 1 }
                        plog!("{label}: отклонён {c} {m}");
                    }
                }
            }
        }
    }
}

fn pool_thread(url: String, user: String, pass: String, agent: String, sh: Arc<Mutex<Shared>>, rx: mpsc::Receiver<Submit>) {
    let mut backoff = 2u64;
    loop {
        if let Err(e) = session(&url, &user, &pass, &agent, &sh, &rx) {
            plog!("соединение: {e:#}; повтор через {backoff} с");
        }
        let was = std::mem::replace(&mut sh.lock().unwrap().connected, false);
        if was {
            backoff = 2;
        }
        std::thread::sleep(Duration::from_secs(backoff));
        backoff = (backoff * 2).min(60);
    }
}

unsafe fn cstr(p: *const c_char) -> String {
    if p.is_null() { String::new() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() }
}

/// Запуск клиента пула в фоне. 0 — ок.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pm_pool_start(url: *const c_char, user: *const c_char, pass: *const c_char, agent: *const c_char) -> i32 {
    let (url, user, pass, agent) = unsafe { (cstr(url), cstr(user), cstr(pass), cstr(agent)) };
    if let Err(e) = parse_url(&url) {
        plog!("неверный адрес пула «{url}»: {e}");
        return 1;
    }
    let (tx, rx) = mpsc::channel();
    let shared = Arc::new(Mutex::new(Shared::default()));
    if POOL.set(Pool { shared: shared.clone(), tx: Mutex::new(tx) }).is_err() {
        return 2;
    }
    std::thread::spawn(move || pool_thread(url, user, pass, agent, shared, rx));
    0
}

/// Текущее задание; возвращает его serial (0 — заданий ещё нет).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pm_pool_job(out: *mut PmJob) -> u64 {
    let Some(p) = POOL.get() else { return 0 };
    match p.shared.lock().unwrap().job {
        Some(j) => {
            unsafe { *out = j };
            j.serial
        }
        None => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pm_pool_stats(acc: *mut u64, rej: *mut u64, stale: *mut u64, connected: *mut i32) {
    let Some(p) = POOL.get() else { return };
    let g = p.shared.lock().unwrap();
    unsafe {
        *acc = g.accepted;
        *rej = g.rejected;
        *stale = g.stale;
        *connected = g.connected as i32;
    }
}

/// Шар: данные копируются, доказательство строится, проверяется и отправляется в фоне.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn pm_pool_submit(
    job: *const PmJob, a: *const i8, m: usize, bt: *const i8, n: usize, k: usize, rank: usize,
    rows: *const u32, nrows: usize, cols: *const u32, ncols: usize, gpu: i32,
) -> i32 {
    let Some(p) = POOL.get() else { return 1 };
    let (job, a, bt, rows, cols) = unsafe {
        (
            *job,
            std::slice::from_raw_parts(a, m * k).to_vec(),
            std::slice::from_raw_parts(bt, n * k).to_vec(),
            std::slice::from_raw_parts(rows, nrows).to_vec(),
            std::slice::from_raw_parts(cols, ncols).to_vec(),
        )
    };
    let tx = p.tx.lock().unwrap().clone();
    let shared = p.shared.clone();
    std::thread::spawn(move || {
        let label = format!("GPU{gpu} шар (строка {}, столбец {})", rows[0], cols[0]);
        let res = std::panic::catch_unwind(|| -> Result<String> {
            let (header, pp) = proof::build(&job.header, &a, m, &bt, n, k, rank, &rows, &cols)?;
            let nbits = target_to_nbits_ceil(U256::from_little_endian(&job.target_le));
            proof::verify(&header, &pp, job.cert_version, Some(nbits)).context("локальная проверка")?;
            let gz = shared.lock().unwrap().gzip;
            proof::encode(&pp, gz)
        });
        match res {
            Ok(Ok(b64)) => {
                let id = cstr_arr(&job.job_id);
                plog!("{label}: доказательство {} КБ, отправляю", b64.len() / 1024);
                let _ = tx.send(Submit { job_id: id, proof_b64: b64, label });
            }
            Ok(Err(e)) => plog!("{label}: НЕ отправлен — {e:#}"),
            Err(_) => plog!("{label}: НЕ отправлен — паника при построении доказательства"),
        }
    });
    0
}

/// Синхронная проверка тайла (для самотеста координат удач с GPU). 0 — доказательство верно.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn pm_verify_tile(
    header76: *const u8, a: *const i8, m: usize, bt: *const i8, n: usize, k: usize, rank: usize,
    rows: *const u32, nrows: usize, cols: *const u32, ncols: usize, cert_version: u32, nbits: u32,
) -> i32 {
    let r = std::panic::catch_unwind(|| -> Result<()> {
        let (hdr, a, bt, rows, cols) = unsafe {
            (
                std::slice::from_raw_parts(header76, 76),
                std::slice::from_raw_parts(a, m * k),
                std::slice::from_raw_parts(bt, n * k),
                std::slice::from_raw_parts(rows, nrows),
                std::slice::from_raw_parts(cols, ncols),
            )
        };
        let (h, pp) = proof::build(hdr, a, m, bt, n, k, rank, rows, cols)?;
        proof::verify(&h, &pp, cert_version, Some(nbits))
    });
    match r {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            plog!("проверка тайла: {e:#}");
            1
        }
        Err(_) => 2,
    }
}

#[cfg(test)]
mod tests;
