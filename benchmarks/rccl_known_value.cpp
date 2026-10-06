// Known-value RCCL AllReduce check: one MPI process per GPU, FP32 sum.
//
// Rank r fills its send buffer with the constant r + 1, so every output element
// must equal n(n + 1) / 2 exactly (3.0 for two ranks; exact in FP32). The receive
// buffer is pre-filled with NaN so a collective that writes nothing is caught.
// Each count is checked out-of-place and in-place. Rank 0 prints one JSON object;
// the exit status is non-zero if any element on any rank is wrong.
//
// Build (see docs/rccl-mi210-baseline.md):
//   mpicxx -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include \
//     benchmarks/rccl_known_value.cpp -L/opt/rocm/lib -lrccl -lamdhip64 \
//     -Wl,-rpath,/opt/rocm/lib -o rccl_known_value

#include <hip/hip_runtime_api.h>
#include <mpi.h>
#include <rccl/rccl.h>

#include <cstdio>
#include <vector>

#define HIP_OK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
  std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
  MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)
#define RCCL_OK(call) do { ncclResult_t r = (call); if (r != ncclSuccess) { \
  std::fprintf(stderr, "RCCL error %s at %s:%d\n", ncclGetErrorString(r), __FILE__, __LINE__); \
  MPI_Abort(MPI_COMM_WORLD, 3); } } while (0)

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 0, local_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm local;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
  MPI_Comm_rank(local, &local_rank);
  HIP_OK(hipSetDevice(local_rank));

  hipDeviceProp_t prop;
  HIP_OK(hipGetDeviceProperties(&prop, local_rank));
  char bus_id[32] = {0};
  HIP_OK(hipDeviceGetPCIBusId(bus_id, sizeof(bus_id), local_rank));
  std::vector<char> all_bus_ids(32 * nranks);
  MPI_Gather(bus_id, 32, MPI_CHAR, all_bus_ids.data(), 32, MPI_CHAR, 0, MPI_COMM_WORLD);

  ncclUniqueId id;
  if (rank == 0) RCCL_OK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm;
  RCCL_OK(ncclCommInitRank(&comm, nranks, id, rank));
  hipStream_t stream;
  HIP_OK(hipStreamCreate(&stream));

  const float contribution = static_cast<float>(rank + 1);
  const float expected = static_cast<float>(nranks * (nranks + 1) / 2);
  const size_t counts[] = {1, 1000, size_t(1) << 20, (size_t(1) << 24) + 3};
  long long total_wrong = 0;
  if (rank == 0) {
    std::printf("{\"collective\": \"AllReduce\", \"dtype\": \"float32\", \"op\": \"sum\", "
                "\"ranks\": %d, \"expected\": %.1f, \"devices\": [", nranks, expected);
    for (int r = 0; r < nranks; ++r)
      std::printf("%s{\"rank\": %d, \"pci_bus_id\": \"%s\"}", r ? ", " : "", r, &all_bus_ids[32 * r]);
    std::printf("], \"gcn_arch\": \"%s\", \"cases\": [", prop.gcnArchName);
  }
  bool first = true;
  for (size_t count : counts) {
    const size_t bytes = count * sizeof(float);
    std::vector<float> host(count, contribution);
    float *send = nullptr, *recv = nullptr;
    HIP_OK(hipMalloc(&send, bytes));
    HIP_OK(hipMalloc(&recv, bytes));
    for (int in_place = 0; in_place < 2; ++in_place) {
      float* out = in_place ? send : recv;
      HIP_OK(hipMemcpy(send, host.data(), bytes, hipMemcpyHostToDevice));
      if (!in_place) HIP_OK(hipMemset(recv, 0xFF, bytes));  // NaN bit pattern
      RCCL_OK(ncclAllReduce(send, out, count, ncclFloat32, ncclSum, comm, stream));
      HIP_OK(hipStreamSynchronize(stream));
      std::vector<float> result(count);
      HIP_OK(hipMemcpy(result.data(), out, bytes, hipMemcpyDeviceToHost));
      long long wrong = 0;
      for (float value : result) wrong += (value != expected);  // NaN != expected
      long long wrong_all = 0;
      MPI_Reduce(&wrong, &wrong_all, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      if (rank == 0) {
        std::printf("%s{\"count\": %zu, \"bytes\": %zu, \"in_place\": %d, \"wrong\": %lld}",
                    first ? "" : ", ", count, bytes, in_place, wrong_all);
        total_wrong += wrong_all;
      }
      first = false;
    }
    HIP_OK(hipFree(send));
    HIP_OK(hipFree(recv));
  }
  MPI_Bcast(&total_wrong, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
  int version = 0;
  RCCL_OK(ncclGetVersion(&version));
  if (rank == 0)
    std::printf("], \"rccl_version_code\": %d, \"total_wrong\": %lld, \"passed\": %s}\n",
                version, total_wrong, total_wrong == 0 ? "true" : "false");
  RCCL_OK(ncclCommDestroy(comm));
  HIP_OK(hipStreamDestroy(stream));
  MPI_Comm_free(&local);
  MPI_Finalize();
  return total_wrong == 0 ? 0 : 1;
}
