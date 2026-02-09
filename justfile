# Run all CI checks
ci: fmt build clippy test bindings

# Check code formatting
fmt:
    cargo fmt --all -- --check

# Build all targets
build:
    cargo build --workspace --exclude binar-python --exclude paulimer-bindings --all-features --all-targets --release

# Run clippy lints
clippy:
    cargo clippy --all --all-features --all-targets --release

# Run tests
test:
    cargo test --workspace --exclude binar-python --exclude paulimer-bindings --all-features --all-targets --release

# Build and test Python bindings
bindings: binar-bindings paulimer-bindings

# Build and test binar Python bindings
binar-bindings:
    cd binar/bindings/python && maturin develop --release --extras dev && pytest tests/ -v

# Build and test paulimer Python bindings
paulimer-bindings:
    cd paulimer/bindings/python && maturin develop --release --extras dev && pytest tests/ -v
