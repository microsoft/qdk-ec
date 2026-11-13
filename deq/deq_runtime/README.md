# deq-runtime

The runtime engine for **deq**, a dynamic and generic quantum error correction decoding
system. This crate hosts the decoders, coordinators, and gRPC services that
execute QEC decoding workloads, and also ships as a Python extension module
(`deq-runtime`) for use from Python.

The high-level Python frontend and tooling live in the companion
[`deq`](https://pypi.org/project/deq/) package.

See the [qdk-ec repository](https://github.com/microsoft/qdk-ec) for
documentation and examples.
