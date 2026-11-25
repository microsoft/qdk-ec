//! Bounded, locality-owned workers. Only requests and results cross worker boundaries.

use super::thread_pooling::{DecodeError, SharedDecodingHypergraph};
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak, mpsc};
#[cfg(feature = "cli")]
use structdoc::StructDoc;
use tokio::sync::{Notify, Semaphore, oneshot};
use tonic::Status;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct DecoderPoolConfig {
    /// Optional unique pool name.
    #[serde(default)]
    pub name: String,
    /// Comma-separated CPU numbers or inclusive ranges, all on one NUMA node.
    pub cpus: String,
    /// Maximum concurrent cold initializations in this pool.
    #[serde(default = "default_initializations")]
    pub max_initializations: usize,
}

fn default_initializations() -> usize {
    1
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub(super) fn next_identity() -> u64 {
    NEXT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| value.checked_add(1))
        .expect("decoder cache identities exhausted")
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct CacheKey {
    pub decoder: u64,
    pub graph: u64,
}

pub(super) struct CachedValue {
    pub owner: Weak<SharedDecodingHypergraph>,
    pub value: Box<dyn Any + Send>,
}

pub(super) type WorkerCache = HashMap<CacheKey, CachedValue>;
type PreparedValue = Weak<dyn Any + Send + Sync>;
type Work = Box<dyn FnOnce(&mut WorkerCache) + Send>;

enum Message {
    Run {
        key: CacheKey,
        work: Work,
    },
    Evict {
        decoder: u64,
        graphs: Option<Arc<HashSet<u64>>>,
        done: oneshot::Sender<()>,
    },
}

struct Domain {
    id: usize,
    name: String,
    workers: Vec<usize>,
    busy: AtomicUsize,
    initializing: AtomicUsize,
    max_initializations: usize,
    prepared: Mutex<HashMap<CacheKey, PreparedValue>>,
}

#[repr(align(128))]
struct WorkerState {
    id: usize,
    cpu: usize,
    domain: Arc<Domain>,
    busy: AtomicBool,
    cached: RwLock<HashSet<CacheKey>>,
    notify: Arc<Notify>,
}

struct Worker {
    state: Arc<WorkerState>,
    sender: mpsc::Sender<Message>,
    control: Mutex<WorkerControl>,
}

#[derive(Default)]
struct WorkerControl {
    active: Option<CacheKey>,
    evictions: Vec<Message>,
}

pub(crate) struct ExecutionPools {
    domains: Vec<Arc<Domain>>,
    workers: Vec<Arc<Worker>>,
    notify: Arc<Notify>,
    pub(super) admission: Arc<Semaphore>,
}

pub(super) struct WorkerLease {
    worker: Arc<Worker>,
    key: CacheKey,
    initializing: bool,
}

impl Drop for WorkerLease {
    fn drop(&mut self) {
        self.initialized();
        let mut control = self.worker.control.lock().unwrap();
        for eviction in control.evictions.drain(..) {
            if let Message::Evict { decoder, graphs, .. } = &eviction {
                remove_cached(&self.worker.state, *decoder, graphs.as_deref());
            }
            if self.worker.sender.send(eviction).is_err() {
                eprintln!("decoder worker stopped before cache eviction");
            }
        }
        control.active = None;
        self.worker.state.domain.busy.fetch_sub(1, Ordering::Relaxed);
        self.worker.state.busy.store(false, Ordering::Release);
        self.worker.state.notify.notify_waiters();
    }
}

impl WorkerLease {
    pub(super) fn initialized(&mut self) {
        if self.initializing {
            self.initializing = false;
            self.worker.state.domain.initializing.fetch_sub(1, Ordering::Release);
            self.worker.state.notify.notify_waiters();
        }
    }

    pub(super) fn identity(&self) -> (usize, usize, usize) {
        (self.worker.state.domain.id, self.worker.state.id, self.worker.state.cpu)
    }

    pub(super) fn cache_inserted(&self, key: CacheKey) {
        self.worker.state.cached.write().unwrap().insert(key);
    }

    pub(super) fn cache_removed(&self, key: CacheKey) {
        self.worker.state.cached.write().unwrap().remove(&key);
    }

    pub(super) fn prepared<P: Any + Send + Sync>(
        &self,
        key: CacheKey,
        build: impl FnOnce() -> Result<P, DecodeError>,
    ) -> Result<Arc<P>, DecodeError> {
        let mut prepared = self.worker.state.domain.prepared.lock().unwrap();
        if let Some(value) = prepared.get(&key).and_then(Weak::upgrade) {
            return Ok(value.downcast::<P>().expect("a decoder cache key has one preparation type"));
        }
        prepared.retain(|_, value| value.strong_count() > 0);
        let value = Arc::new(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(build))
                .map_err(|_| DecodeError::Backend("decoder panicked while preparing graph".into()))??,
        );
        let erased: Arc<dyn Any + Send + Sync> = value.clone();
        prepared.insert(key, Arc::downgrade(&erased));
        Ok(value)
    }

    pub(super) fn submit(self, run: impl FnOnce(&mut WorkerCache, Self) + Send + 'static) -> Result<(), Status> {
        let sender = self.worker.sender.clone();
        sender
            .send(Message::Run {
                key: self.key,
                work: Box::new(move |cache| run(cache, self)),
            })
            .map_err(|_| Status::unavailable("decoder worker stopped"))
    }
}

impl ExecutionPools {
    pub(crate) fn new(configurations: &[DecoderPoolConfig]) -> io::Result<Arc<Self>> {
        let parsed = validate_pools(configurations)?;
        let capacity = parsed.iter().map(|(_, cpus)| cpus.len()).sum();
        let notify = Arc::new(Notify::new());
        let mut domains = Vec::new();
        let mut workers = Vec::with_capacity(capacity);
        for (domain_id, (configuration, cpus)) in parsed.into_iter().enumerate() {
            let domain = Arc::new(Domain {
                id: domain_id,
                name: if configuration.name.is_empty() {
                    format!("pool-{domain_id}")
                } else {
                    configuration.name.clone()
                },
                workers: (workers.len()..workers.len() + cpus.len()).collect(),
                busy: AtomicUsize::new(0),
                initializing: AtomicUsize::new(0),
                max_initializations: configuration.max_initializations,
                prepared: Mutex::new(HashMap::new()),
            });
            for (index, cpu) in cpus.into_iter().enumerate() {
                let state = Arc::new(WorkerState {
                    id: index,
                    cpu,
                    domain: Arc::clone(&domain),
                    busy: AtomicBool::new(false),
                    cached: RwLock::new(HashSet::new()),
                    notify: Arc::clone(&notify),
                });
                let (sender, receiver) = mpsc::channel();
                let (started, ready) = mpsc::sync_channel(0);
                let owner = Arc::clone(&state);
                std::thread::Builder::new()
                    .name(format!("deq-p{domain_id}-w{index}"))
                    .spawn(move || {
                        let affinity = bind_cpu(cpu);
                        let success = affinity.is_ok();
                        if started.send(affinity).is_err() || !success {
                            return;
                        }
                        worker_loop(&owner, &receiver);
                    })?;
                ready
                    .recv()
                    .map_err(|_| io::Error::other("decoder worker exited during startup"))??;
                workers.push(Arc::new(Worker {
                    state,
                    sender,
                    control: Mutex::new(WorkerControl::default()),
                }));
            }
            domains.push(domain);
        }
        Ok(Arc::new(Self {
            domains,
            workers,
            notify,
            admission: Arc::new(Semaphore::new(capacity)),
        }))
    }

    pub(crate) fn capacity(&self) -> usize {
        self.workers.len()
    }

    pub(super) async fn reserve(&self, key: CacheKey) -> WorkerLease {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(lease) = self.try_reserve(key) {
                return lease;
            }
            notified.await;
        }
    }

    fn try_reserve(&self, key: CacheKey) -> Option<WorkerLease> {
        let mut domains: Vec<_> = self.domains.iter().collect();
        domains.sort_by_key(|domain| domain.busy.load(Ordering::Relaxed) * self.workers.len() / domain.workers.len());
        for cached in [true, false] {
            for domain in &domains {
                for &index in &domain.workers {
                    let worker = &self.workers[index];
                    let mut control = worker.control.lock().unwrap();
                    if worker.state.busy.load(Ordering::Acquire)
                        || worker.state.cached.read().unwrap().contains(&key) != cached
                    {
                        continue;
                    }
                    let initializing = !cached;
                    if initializing
                        && domain
                            .initializing
                            .fetch_update(Ordering::Acquire, Ordering::Relaxed, |count| {
                                (count < domain.max_initializations).then_some(count + 1)
                            })
                            .is_err()
                    {
                        continue;
                    }
                    if worker
                        .state
                        .busy
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                        .is_err()
                    {
                        if initializing {
                            domain.initializing.fetch_sub(1, Ordering::Release);
                            self.notify.notify_waiters();
                        }
                        continue;
                    }
                    domain.busy.fetch_add(1, Ordering::Relaxed);
                    control.active = Some(key);
                    return Some(WorkerLease {
                        worker: Arc::clone(worker),
                        key,
                        initializing,
                    });
                }
            }
        }
        None
    }

    pub(super) fn enqueue_eviction(
        &self,
        decoder: u64,
        graphs: Option<&Arc<HashSet<u64>>>,
    ) -> Result<Vec<oneshot::Receiver<()>>, Status> {
        let mut completions = Vec::with_capacity(self.workers.len());
        for worker in &self.workers {
            let mut control = worker.control.lock().unwrap();
            let selected =
                |key: &CacheKey| key.decoder == decoder && graphs.as_ref().is_none_or(|ids| ids.contains(&key.graph));
            if !control.active.as_ref().is_some_and(selected) && !worker.state.cached.read().unwrap().iter().any(selected) {
                continue;
            }
            remove_cached(&worker.state, decoder, graphs.map(Arc::as_ref));
            let (done, completion) = oneshot::channel();
            let message = Message::Evict {
                decoder,
                graphs: graphs.cloned(),
                done,
            };
            if control.active.is_some() {
                control.evictions.push(message);
            } else {
                worker
                    .sender
                    .send(message)
                    .map_err(|_| Status::unavailable("decoder worker stopped during eviction"))?;
            }
            completions.push(completion);
        }
        Ok(completions)
    }

    pub(super) async fn evict(&self, decoder: u64, graphs: Option<Arc<HashSet<u64>>>) -> Result<(), Status> {
        for completion in self.enqueue_eviction(decoder, graphs.as_ref())? {
            completion
                .await
                .map_err(|_| Status::unavailable("decoder worker stopped during eviction"))?;
        }
        Ok(())
    }
}

fn remove_cached(state: &WorkerState, decoder: u64, graphs: Option<&HashSet<u64>>) {
    state
        .cached
        .write()
        .unwrap()
        .retain(|key| key.decoder != decoder || graphs.is_some_and(|ids| !ids.contains(&key.graph)));
}

fn worker_loop(state: &WorkerState, receiver: &mpsc::Receiver<Message>) {
    let mut cache = WorkerCache::new();
    while let Ok(message) = receiver.recv() {
        match message {
            Message::Run { key, work } => {
                if !cache.contains_key(&key) {
                    cache.retain(|key, entry| {
                        if entry.owner.strong_count() == 0 {
                            state.cached.write().unwrap().remove(key);
                            false
                        } else {
                            true
                        }
                    });
                }
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(&mut cache))).is_err() {
                    eprintln!("decoder task panicked in pool {}", state.domain.name);
                }
            }
            Message::Evict { decoder, graphs, done } => {
                let keep =
                    |key: &CacheKey| key.decoder != decoder || graphs.as_ref().is_some_and(|ids| !ids.contains(&key.graph));
                cache.retain(|key, _| keep(key));
                state.cached.write().unwrap().retain(keep);
                state.domain.prepared.lock().unwrap().retain(|key, _| keep(key));
                // A cancelled caller does not cancel worker-local cleanup.
                let _ = done.send(());
            }
        }
    }
}

pub(super) fn parse_cpus(value: &str) -> io::Result<Vec<usize>> {
    let mut cpus = Vec::new();
    let mut seen = HashSet::new();
    for item in value.split(',') {
        let item = item.trim();
        let (start, end) = item.split_once('-').unwrap_or((item, item));
        let start = start.parse::<usize>().map_err(|_| invalid_cpu_list(value))?;
        let end = end.parse::<usize>().map_err(|_| invalid_cpu_list(value))?;
        if start > end || end >= 1024 {
            return Err(invalid_cpu_list(value));
        }
        for cpu in start..=end {
            if !seen.insert(cpu) {
                return Err(invalid_cpu_list(value));
            }
            cpus.push(cpu);
        }
    }
    if cpus.is_empty() {
        return Err(invalid_cpu_list(value));
    }
    Ok(cpus)
}

fn invalid_cpu_list(value: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("invalid or repeated CPU in list {value:?}"),
    )
}

fn validate_pools(configurations: &[DecoderPoolConfig]) -> io::Result<Vec<(&DecoderPoolConfig, Vec<usize>)>> {
    if configurations.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "at least one decoder pool is required",
        ));
    }
    let mut names = HashSet::new();
    let mut assigned = HashSet::new();
    let mut parsed = Vec::new();
    for (index, configuration) in configurations.iter().enumerate() {
        let name = if configuration.name.is_empty() {
            format!("pool-{index}")
        } else {
            configuration.name.clone()
        };
        if !names.insert(name) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "decoder pool names must be unique",
            ));
        }
        let cpus = parse_cpus(&configuration.cpus)?;
        if configuration.max_initializations == 0 || configuration.max_initializations > cpus.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "initialization capacity must be between one and the pool size",
            ));
        }
        let mut node = None;
        for &cpu in &cpus {
            if !assigned.insert(cpu) {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "decoder pool CPU lists overlap"));
            }
            let current = numa_node(cpu)?;
            if let Some(previous) = node {
                if previous != current {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "each decoder pool must stay within one NUMA node",
                    ));
                }
            } else {
                node = Some(current);
            }
        }
        parsed.push((configuration, cpus));
    }
    Ok(parsed)
}

#[cfg(target_os = "linux")]
pub(super) fn numa_node(cpu: usize) -> io::Result<Option<usize>> {
    for entry in std::fs::read_dir(format!("/sys/devices/system/cpu/cpu{cpu}"))? {
        let entry = entry?;
        let name = entry.file_name();
        if let Some(node) = name.to_str().and_then(|name| name.strip_prefix("node")) {
            return node
                .parse()
                .map(Some)
                .map_err(|_| io::Error::other("invalid NUMA node identifier"));
        }
    }
    Ok(None)
}

#[cfg(not(target_os = "linux"))]
fn numa_node(_cpu: usize) -> io::Result<Option<usize>> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "CPU-list decoder pools currently require Linux",
    ))
}

#[cfg(target_os = "linux")]
fn bind_cpu(cpu: usize) -> io::Result<()> {
    // cpu_set_t is an integer bitset; CPU_SET is only called with an in-range index.
    let mut mask: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    if cpu >= usize::try_from(libc::CPU_SETSIZE).unwrap() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "CPU exceeds the supported affinity mask",
        ));
    }
    let result = unsafe {
        libc::CPU_ZERO(&mut mask);
        libc::CPU_SET(cpu, &mut mask);
        libc::sched_setaffinity(0, std::mem::size_of_val(&mask), &raw const mask)
    };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(not(target_os = "linux"))]
fn bind_cpu(_cpu: usize) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "CPU-list decoder pools currently require Linux",
    ))
}

#[cfg(test)]
#[path = "../../tests/unit/execution_pool_test.rs"]
pub(super) mod tests;
