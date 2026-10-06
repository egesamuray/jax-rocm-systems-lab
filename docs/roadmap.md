# Ordered roadmap

Work is gated by evidence and hardware availability, with no delivery dates.
Each stage starts with a small correctness experiment and a reproducible record.

1. **RCCL:** on an available AMD system, capture the stack and topology, run one
   small collective with a correctness oracle, then record synchronized samples.
   Exit: reproducible command, raw evidence, and a documented bandwidth definition.
   Evidence: [2x MI210 AllReduce baseline](rccl-mi210-baseline.md) (2026-10-06).
2. **HIP/XLA FFI:** after the RCCL record, implement one bounded kernel and
   compare it with equivalent JAX math. Check shape/dtype, ABI, stream use, buffer
   ownership, numerical correctness, and error handling before timing.
   Exit: one correct integration and an honest baseline comparison.
3. **MaxText:** after FFI correctness, pin an upstream revision and choose one
   small configuration that fits available hardware. Record sharding and runtime
   settings, finite loss, and synchronized step timings with warmup separated.
   Exit: one reproducible smoke run; scaling or optimization claims require
   separate measurements.

Upstream starting points:

- [RCCL documentation](https://rocm.docs.amd.com/projects/rccl/en/latest/)
- [HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [JAX FFI guide](https://docs.jax.dev/en/latest/ffi.html)
- [MaxText](https://github.com/AI-Hypercomputer/maxtext)
