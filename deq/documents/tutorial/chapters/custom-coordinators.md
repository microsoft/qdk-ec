# Plug in your own coordinator in Rust

The coordinator is the service that sits between the controller (static or
JIT) and the decoder: it tracks gadget instances, accumulates measurement
outcomes, decides when enough context has arrived to commit a logical
readout, and calls the decoder when it's time to actually decode. deq ships
several built-in coordinators (`naive`, `monolithic`, `window`), but
sometimes none of them is the right shape for what you're building — maybe
you want to forward decoding decisions to an external control system, keep
your own notion of "pending" and "committed" for a non-standard scheduling
policy, or bridge the coordinator interface onto hardware you already run
outside of deq.

This chapter shows how to write that coordinator as a **Rust crate that
depends on `deq_runtime`**, without touching `deq_runtime`'s own source.

We will walk through:

1. What a coordinator actually has to implement.
2. A minimal worked example.
3. Making cancellation real.
4. Wiring your coordinator into a controller and starting it.
5. Exposing it over gRPC, if you need to.

---

## What a coordinator has to implement

Every coordinator — built-in or custom — implements the tonic-generated
`coordinator_server::Coordinator` trait. This is the same RPC surface the
`deq.coordinator` proto service describes:

```rust
#[tonic::async_trait]
pub trait Coordinator: Send + Sync + 'static {
    async fn load_library(&self, request: Request<bin::Library>)
        -> Result<Response<()>, Status>;
    async fn unload(&self, request: Request<UnloadLibrary>)
        -> Result<Response<()>, Status>;
    async fn execute(&self, request: Request<bin::Instruction>)
        -> Result<Response<ExecuteResponse>, Status>;
    async fn decode(&self, request: Request<Outcomes>)
        -> Result<Response<Readouts>, Status>;
    async fn reset(&self, request: Request<ResetRequest>)
        -> Result<Response<()>, Status>;
}
```

`decode` is the interesting one: it is *not* guaranteed to return as soon as
it's called. The caller is handing you one gadget's measurement outcomes;
you hand back its logical readout whenever your coordinator has gathered
enough surrounding context to be sure of it. This is the same streaming
contract the built-in `window` and `monolithic` coordinators follow (see
[Driving the runtime from Python](python-runtime.md#decoding-is-streaming-not-call-and-return)
for the caller-side view of this).

On top of that gRPC surface, deq_runtime defines a second, smaller trait
every custom coordinator must also implement:

```rust
#[tonic::async_trait]
pub trait CustomCoordinator: coordinator_server::Coordinator {
    async fn cancel_pending(&self);
}
```

`cancel_pending` exists for shutdown. When a `deq` runtime shuts down — a
`deq server` process exits, or a Python `Runtime` leaves its `async with`
block — it needs any `decode()` call that is still sitting pending (waiting
for more context that will now never arrive) to resolve immediately,
instead of hanging forever. The built-in coordinators do this with an
internal `CancellationToken` that every in-flight `decode()` selects against;
your implementation is free to use whatever mechanism fits its internal
bookkeeping, as long as calling `cancel_pending()` actually unblocks pending
work. There is no default implementation — a coordinator that doesn't wire
this up correctly will hang on shutdown.

---

## A minimal worked example

Let's write a coordinator that does the simplest possible real thing:
decode each gadget's outcomes independently, with no cross-gadget context,
but track in-flight work so shutdown is clean. (This is intentionally
close to the built-in `naive` coordinator — the point of this section is
the wiring, not a sophisticated decoding policy.)

```rust
use deq_runtime::bin;
use deq_runtime::coordinator::{
    self, coordinator_server, CustomCoordinator, ExecuteResponse, Outcomes,
    Readouts, ResetRequest, UnloadLibrary,
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

pub struct MyCoordinator {
    cancellation: CancellationToken,
}

impl MyCoordinator {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            cancellation: CancellationToken::new(),
        })
    }
}

#[tonic::async_trait]
impl coordinator_server::Coordinator for MyCoordinator {
    async fn load_library(&self, request: Request<bin::Library>) -> Result<Response<()>, Status> {
        // Register gadget/port/check-model types however your backend needs to.
        Ok(Response::new(()))
    }

    async fn unload(&self, _request: Request<UnloadLibrary>) -> Result<Response<()>, Status> {
        Ok(Response::new(()))
    }

    async fn execute(&self, request: Request<bin::Instruction>) -> Result<Response<ExecuteResponse>, Status> {
        // Instantiate the gadget/check model/error model described by the
        // instruction and return the id assigned to it.
        todo!()
    }

    async fn decode(&self, request: Request<Outcomes>) -> Result<Response<Readouts>, Status> {
        tokio::select! {
            () = self.cancellation.cancelled() => Err(Status::cancelled("coordinator is shutting down")),
            result = self.decode_one(request.into_inner()) => result.map(Response::new),
        }
    }

    async fn reset(&self, _request: Request<ResetRequest>) -> Result<Response<()>, Status> {
        Ok(Response::new(()))
    }
}

#[tonic::async_trait]
impl CustomCoordinator for MyCoordinator {
    async fn cancel_pending(&self) {
        self.cancellation.cancel();
    }
}
```

(`decode_one` is a stand-in for whatever per-gadget decoding logic your
coordinator actually needs — the shape above is just to show how
cancellation plugs into the `tokio::select!` that every pending `decode`
call should be racing against.)

---

## Making cancellation real

The one thing worth getting right here is: `cancel_pending()` must
*actually* unblock every call that's pending *right now*, not just future
ones. A `CancellationToken` works well because `cancelled()` resolves
immediately for any call racing against it the moment `cancel()` fires,
regardless of when that call started waiting — there's no need to track a
separate list of in-flight futures.

If your coordinator wraps an external system (a socket, a subprocess, a
remote queue), make sure cancellation actually reaches that boundary too —
for example, close the queue, or forward the cancellation signal to your
backend's own control channel.

---

## Wiring your coordinator into a controller

Wrap your coordinator in `DynCoordinator::Custom`, build a
`CoordinatorClient` from it, and hand that client to any controller. None
of this requires going through `deq server`'s CLI config — `JitController`
and `StaticController` are ordinary public types you construct and start
yourself:

```rust
use deq_runtime::controller::JitController;
use deq_runtime::coordinator::{CoordinatorClient, DynCoordinator};
use serde_json::json;

let coordinator = DynCoordinator::Custom(MyCoordinator::new());
let coordinator_client = CoordinatorClient::Local(coordinator.clone());

let controller = JitController::new(json!({}));
controller.start(coordinator_client).await;
```

From here on, `controller` behaves exactly like one built against a
built-in coordinator: stream `JitInstruction`s via `controller.execute(...)`
/ `controller.batch_execute(...)`, submit outcomes via
`controller.decode(...)` / `controller.batch_decode(...)`.

When you're done, call `cancel_pending()` on both the controller and the
coordinator to unwind any pending decodes before dropping them:

```rust
controller.cancel_pending().await;
coordinator.cancel_pending().await;
```

---

## Exposing your coordinator over gRPC

If you also want external processes to be able to talk to your custom
coordinator, `DynCoordinator::add_service` (behind the `cli` feature) knows
how to register a `Custom` coordinator on a `tonic` router, the same way it
registers the built-in ones:

```rust
let mut server = tonic::transport::Server::builder();
let router = server.add_service(/* whatever other services you need */);
let router = coordinator.add_service(router);
let router = deq_runtime::controller::DynController::Jit(controller.clone()).add_service(router);
router.serve(addr).await?;
```

Internally, `add_service` wraps your `Arc<dyn CustomCoordinator>` in a thin
adapter so it can be registered as a concrete gRPC service type — you don't
need to do anything special to make this work, it falls out of implementing
`coordinator_server::Coordinator` as shown above.

---

## Summary

| Step                                          | What you do                                                                 |
| ---------------------------------------------- | ---------------------------------------------------------------------------- |
| Implement the gRPC surface                     | `impl coordinator_server::Coordinator for MyCoordinator`                      |
| Implement real cancellation                    | `impl CustomCoordinator for MyCoordinator { async fn cancel_pending(&self) }` |
| Wrap it                                        | `DynCoordinator::Custom(Arc::new(MyCoordinator::new()))`                      |
| Give it to a controller                        | `CoordinatorClient::Local(coordinator)`, then `controller.start(client)`      |
| Expose it remotely (optional)                  | `coordinator.add_service(router)` behind the `cli` feature                   |
| Shut down cleanly                              | `controller.cancel_pending().await; coordinator.cancel_pending().await;`     |
