use super::*;

#[test]
fn cpu_lists_reject_ambiguous_or_unbounded_ranges() {
    assert_eq!(parse_cpus("2, 4-6,9").unwrap(), vec![2, 4, 5, 6, 9]);
    for value in ["", "1,", "-1", "3-1", "1-3,2", "1-2-3", "1024", "0-18446744073709551615"] {
        assert!(parse_cpus(value).is_err(), "{value}");
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn allowed_cpus() -> Vec<usize> {
    // The test reads its inherited mask and never changes the test-runner affinity.
    let mut mask: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&mask), &raw mut mask) },
        0,
    );
    (0..usize::try_from(libc::CPU_SETSIZE).unwrap())
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &mask) })
        .collect()
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn worker_leases_preserve_capacity_affinity_and_cached_routing() {
    use futures_util::FutureExt as _;
    let cpus = allowed_cpus();
    let configurations: Vec<_> = cpus
        .iter()
        .take(2)
        .enumerate()
        .map(|(index, cpu)| DecoderPoolConfig {
            name: format!("test-{index}"),
            cpus: cpu.to_string(),
            max_initializations: 1,
        })
        .collect();
    let pools = ExecutionPools::new(&configurations).unwrap();
    assert_eq!(pools.capacity(), configurations.len());
    let key = CacheKey { decoder: 1, graph: 1 };
    let mut leases = Vec::new();
    for _ in 0..pools.capacity() {
        leases.push(pools.reserve(key).await);
    }
    let mut blocked = Box::pin(pools.reserve(key));
    assert!(blocked.as_mut().now_or_never().is_none());
    let mut released = leases.pop().unwrap();
    let identity = released.identity();
    released.cache_inserted(key);
    released.initialized();
    drop(released);
    let lease = blocked.await;
    assert_eq!(lease.identity(), identity);
    let (done, completed) = oneshot::channel();
    lease
        .submit(move |_, _lease| {
            let cpu = unsafe { libc::sched_getcpu() };
            done.send(usize::try_from(cpu).unwrap()).unwrap();
        })
        .unwrap();
    assert_eq!(completed.await.unwrap(), identity.2);
}

#[cfg(target_os = "linux")]
#[test]
fn repeated_pool_cpus_and_names_are_rejected() {
    let cpu = allowed_cpus()[0].to_string();
    let first = DecoderPoolConfig {
        name: "first".into(),
        cpus: cpu.clone(),
        max_initializations: 1,
    };
    let second = DecoderPoolConfig {
        name: "second".into(),
        cpus: cpu,
        max_initializations: 1,
    };
    assert!(
        ExecutionPools::new(&[first.clone(), second])
            .err()
            .unwrap()
            .to_string()
            .contains("overlap")
    );
    assert!(
        ExecutionPools::new(&[first.clone(), first])
            .err()
            .unwrap()
            .to_string()
            .contains("unique")
    );
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn a_cold_waiter_cannot_consume_a_cached_requests_wakeup() {
    use futures_util::FutureExt as _;

    let cpus = allowed_cpus();
    let node = numa_node(cpus[0]).unwrap();
    let cpus: Vec<_> = cpus
        .into_iter()
        .filter(|&cpu| numa_node(cpu).unwrap() == node)
        .take(2)
        .collect();
    if cpus.len() < 2 {
        eprintln!("cached-wakeup regression requires two available CPUs on one NUMA node");
        return;
    }
    let pools = ExecutionPools::new(&[DecoderPoolConfig {
        name: "shared-domain".into(),
        cpus: format!("{},{}", cpus[0], cpus[1]),
        max_initializations: 1,
    }])
    .unwrap();
    let hot_key = CacheKey { decoder: 1, graph: 1 };
    let mut hot = pools.reserve(hot_key).await;
    hot.cache_inserted(hot_key);
    hot.initialized();
    let constructing = pools.reserve(CacheKey { decoder: 1, graph: 2 }).await;
    let mut cold_waiter = Box::pin(pools.reserve(CacheKey { decoder: 1, graph: 3 }));
    let mut hot_waiter = Box::pin(pools.reserve(hot_key));
    assert!(cold_waiter.as_mut().now_or_never().is_none());
    assert!(hot_waiter.as_mut().now_or_never().is_none());
    drop(hot);
    assert!(cold_waiter.as_mut().now_or_never().is_none());
    let available = tokio::time::timeout(std::time::Duration::from_secs(1), hot_waiter).await;
    drop(constructing);
    assert!(
        available.is_ok(),
        "an unrelated constructor must not block a ready cached worker"
    );
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn eviction_waits_for_a_reserved_cache_hit_before_removing_state() {
    use futures_util::FutureExt as _;
    let cpu = allowed_cpus()[0];
    let pools = ExecutionPools::new(&[DecoderPoolConfig {
        name: "test".into(),
        cpus: cpu.to_string(),
        max_initializations: 1,
    }])
    .unwrap();
    let graph = Arc::new(SharedDecodingHypergraph::new(
        crate::decoder::blackbox_decoder::DecodingHypergraph::default(),
    ));
    let key = CacheKey {
        decoder: 7,
        graph: graph.identity(),
    };
    let owner = Arc::downgrade(&graph);
    let lease = pools.reserve(key).await;
    let (done, ready) = oneshot::channel();
    lease
        .submit(move |cache, mut lease| {
            cache.insert(
                key,
                CachedValue {
                    owner,
                    value: Box::new(42_u32),
                },
            );
            lease.cache_inserted(key);
            lease.initialized();
            done.send(()).unwrap();
        })
        .unwrap();
    ready.await.unwrap();
    let lease = pools.reserve(key).await;
    let mut evicting = Box::pin(pools.evict(key.decoder, Some(Arc::new(HashSet::from([key.graph])))));
    assert!(evicting.as_mut().now_or_never().is_none());
    let (done, completed) = oneshot::channel();
    lease
        .submit(move |cache, _lease| {
            assert_eq!(*cache[&key].value.downcast_ref::<u32>().unwrap(), 42);
            done.send(()).unwrap();
        })
        .unwrap();
    completed.await.unwrap();
    evicting.await.unwrap();
    let lease = pools.reserve(key).await;
    let (done, completed) = oneshot::channel();
    lease
        .submit(move |cache, _lease| {
            assert!(!cache.contains_key(&key));
            done.send(()).unwrap();
        })
        .unwrap();
    completed.await.unwrap();
}
