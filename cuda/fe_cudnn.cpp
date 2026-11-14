// Optional cuDNN fused attention for flashembed-gpu (host C++, built with the C++ compiler) (bf16): cuDNN's SDPA (tcgen05 flash attention on Blackwell) on the
// engine's packed layout. Built only when FLASHEMBED_CUDNN_INCLUDE points at the cuDNN 9 and cudnn-frontend (MIT)
// headers; libcudnn is loaded at run time (dlopen), so the binary has no link-time cuDNN dependency and falls back to
// its own attention kernels when the library is missing or a shape is unsupported.
//
// Layout: one qkv buffer [ntok][ld] (q at q_off + h*D, k/v at k_off/v_off + hk*D), packed sequences given by cu[] on
// the device; output [ntok][ld_out]. Mapped to cuDNN as ragged [B, H, S, D] tensors (token stride ld, head stride D,
// per-sequence ragged offsets cu[b]*ld) with a padding mask from the sequence lengths. A symmetric window |i-j| <= w
// is cuDNN's diagonal band with left bound w+1 and right bound w (top-left aligned). Graphs are cached per
// (B, S) bucket (powers of two; unused sequence slots get length 0).
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <string>

static thread_local std::string g_cd_err;

#ifdef FE_CUDNN
#include <dlfcn.h>

#include <chrono>
#include <cstdio>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#define NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#include <cudnn_frontend.h>

namespace cudnn_frontend {
void* cudnn_dlhandle = nullptr;
}
namespace fe = cudnn_frontend;

namespace {
enum : int64_t { U_Q = 1, U_K, U_V, U_O, U_RAG, U_RAGO, U_SL };


struct Key {
  int B, S, Hq, Hkv, D, w, ld, ld_out;
  float scale;
  bool operator<(const Key& o) const {
    return std::tie(B, S, Hq, Hkv, D, w, ld, ld_out, scale) <
           std::tie(o.B, o.S, o.Hq, o.Hkv, o.D, o.w, o.ld, o.ld_out, o.scale);
  }
};

struct Ctx {
  std::map<Key, std::shared_ptr<fe::KernelCache>> kcache;  // per config with B = S = 0: compiled kernels reused
  double build_s = 0;
  bool dyn = true, log = false;
  cudnnHandle_t h = nullptr;
  cudaStream_t st = nullptr;
  std::map<Key, std::shared_ptr<fe::graph::Graph>> graphs;
  void* ws = nullptr;
  int64_t ws_bytes = 0;
  int nbuilt = 0;
};

int pow2(int x) {
  int p = 1;
  while (p < x) p <<= 1;
  return p;
}

std::shared_ptr<fe::graph::Graph> build(Ctx& c, const Key& k, bool dyn) {
  const float scale = k.scale;
  auto g = std::make_shared<fe::graph::Graph>();
  if (dyn) {  // dynamic shapes + a kernel cache shared by all (B, S) buckets of this config
    Key kc = k;
    kc.B = kc.S = 0;
    auto& cache = c.kcache[kc];
    if (!cache) cache = std::make_shared<fe::KernelCache>();
    g->set_dynamic_shape_enabled(true).set_kernel_cache(cache);
  }
  g->set_io_data_type(fe::DataType_t::BFLOAT16)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  using TA = fe::graph::Tensor_attributes;
  auto rag = g->tensor(TA().set_name("rag").set_uid(U_RAG).set_dim({k.B + 1, 1, 1, 1}).set_stride({1, 1, 1, 1})
                           .set_data_type(fe::DataType_t::INT32));
  auto rago = g->tensor(TA().set_name("rago").set_uid(U_RAGO).set_dim({k.B + 1, 1, 1, 1}).set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));
  auto sl = g->tensor(TA().set_name("sl").set_uid(U_SL).set_dim({k.B, 1, 1, 1}).set_stride({1, 1, 1, 1})
                          .set_data_type(fe::DataType_t::INT32));
  const int64_t S = k.S, ld = k.ld, lo = k.ld_out;
  auto q = g->tensor(TA().set_name("q").set_uid(U_Q).set_dim({k.B, k.Hq, S, k.D}).set_stride({S * ld, k.D, ld, 1})
                         .set_ragged_offset(rag));
  auto kk = g->tensor(TA().set_name("k").set_uid(U_K).set_dim({k.B, k.Hkv, S, k.D}).set_stride({S * ld, k.D, ld, 1})
                          .set_ragged_offset(rag));
  auto v = g->tensor(TA().set_name("v").set_uid(U_V).set_dim({k.B, k.Hkv, S, k.D}).set_stride({S * ld, k.D, ld, 1})
                         .set_ragged_offset(rag));
  auto at = fe::graph::SDPA_attributes()
                .set_name("sdpa")
                .set_generate_stats(false)
                .set_attn_scale(scale)
                .set_padding_mask(true)
                .set_seq_len_q(sl)
                .set_seq_len_kv(sl);
  if (k.w >= 0)
    at.set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
        .set_diagonal_band_left_bound(k.w + 1)
        .set_diagonal_band_right_bound(k.w);
  auto [o, stats] = g->sdpa(q, kk, v, at);
  (void)stats;
  o->set_output(true).set_uid(U_O).set_dim({k.B, k.Hq, S, k.D}).set_stride({S * lo, k.D, lo, 1}).set_ragged_offset(rago);
  auto chk = [](fe::error_t e, const char* what) {
    if (!e.is_good()) throw std::runtime_error(std::string(what) + ": " + e.get_message());
  };
  chk(g->validate(), "validate");
  chk(g->build_operation_graph(c.h), "build_operation_graph");
  chk(g->create_execution_plans({fe::HeurMode_t::A}), "create_execution_plans");
  chk(g->check_support(c.h), "check_support");
  chk(g->build_plans(c.h), "build_plans");
  int64_t ws = 0;
  chk(g->get_workspace_size(ws), "workspace");
  if (ws > c.ws_bytes) {
    if (c.ws) cudaFree(c.ws);
    c.ws = nullptr;
    if (cudaMalloc(&c.ws, ws) != cudaSuccess) throw std::runtime_error("cudnn workspace alloc");
    c.ws_bytes = ws;
  }
  return g;
}
}  // namespace
#endif

extern "C" {

const char* fecd_error() { return g_cd_err.c_str(); }

// nullptr when cuDNN is not built in or cannot be loaded (fecd_error says why)
void* fecd_new(void* stream) {
#ifdef FE_CUDNN
  try {
    if (!fe::cudnn_dlhandle) {
      const char* lib = std::getenv("FLASHEMBED_CUDNN_LIB");
      void* hd = dlopen(lib && *lib ? lib : "libcudnn.so.9", RTLD_NOW | RTLD_GLOBAL);
      if (!hd) {
        g_cd_err = std::string("dlopen libcudnn: ") + dlerror();
        return nullptr;
      }
      fe::cudnn_dlhandle = hd;
    }
    if (fe::detail::get_backend_version() < 91000) {
      g_cd_err = "cuDNN >= 9.10 needed for ragged SDPA";
      return nullptr;
    }
    auto* c = new Ctx();
    c->st = (cudaStream_t)stream;
    const char* dy = std::getenv("FLASHEMBED_CUDNN_DYN");
    c->dyn = dy && std::string(dy) == "1";  // opt-in: cuDNN 9.26 has no dynamic-shape engine for ragged banded SDPA
    c->log = std::getenv("FLASHEMBED_CUDNN_LOG") != nullptr;
    if (fe::detail::create_handle(&c->h) != CUDNN_STATUS_SUCCESS) {
      g_cd_err = "cudnnCreate failed";
      delete c;
      return nullptr;
    }
    fe::detail::set_stream(c->h, c->st);
    return c;
  } catch (const std::exception& ex) {
    g_cd_err = ex.what();
    return nullptr;
  }
#else
  (void)stream;
  g_cd_err = "built without cuDNN (set FLASHEMBED_CUDNN_INCLUDE at build time)";
  return nullptr;
#endif
}

// bf16 attention via cuDNN over nb sequence slots (a power of two >= the real count; the caller fills rag = cu*ld,
// rago = cu*ld_out [nb+1] and sl = lengths [nb], 0 for unused slots). 0 = done, nonzero = not done (fecd_error)
int fecd_attn(void* ctx, int nb, int smax, int Hq, int Hkv, int D, int window, float scale, const void* q, const void* k,
              const void* v, void* o, int ld, int ld_out, const int* rag, const int* rago, const int* sl) {
#ifdef FE_CUDNN
  auto* c = (Ctx*)ctx;
  try {
    const Key key{nb, pow2(smax < 64 ? 64 : smax), Hq, Hkv, D, window, ld, ld_out, scale};
    auto it = c->graphs.find(key);
    if (it == c->graphs.end()) {
      const auto t0 = std::chrono::steady_clock::now();
      std::shared_ptr<fe::graph::Graph> g;
      if (c->dyn) {
        try {
          g = build(*c, key, true);
        } catch (const std::exception& ex) {  // no dynamic-shape engine for this config: static graphs from now on
          if (c->log) fprintf(stderr, "cudnn dynamic shapes unavailable (%s); static graphs\n", ex.what());
          c->dyn = false;
        }
      }
      if (!g) g = build(*c, key, false);
      it = c->graphs.emplace(key, g).first;
      const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      c->build_s += dt;
      c->nbuilt++;
      if (c->log)
        fprintf(stderr, "cudnn graph B=%d S=%d H=%d/%d D=%d w=%d dyn=%d: %.1f ms\n", key.B, key.S, key.Hq, key.Hkv, key.D,
                key.w, (int)c->dyn, dt * 1e3);
    }
    std::unordered_map<int64_t, void*> pack{{U_Q, (void*)q},     {U_K, (void*)k},       {U_V, (void*)v}, {U_O, o},
                                            {U_RAG, (void*)rag}, {U_RAGO, (void*)rago}, {U_SL, (void*)sl}};
    fe::error_t st = it->second->execute(c->h, pack, c->ws);
    if (!st.is_good()) {
      g_cd_err = "execute: " + st.get_message();
      return 2;
    }
    return 0;
  } catch (const std::exception& ex) {
    g_cd_err = ex.what();
    return 1;
  }
#else
  (void)ctx, (void)nb, (void)smax, (void)Hq, (void)Hkv, (void)D, (void)window, (void)scale, (void)q, (void)k, (void)v,
      (void)o, (void)ld, (void)ld_out, (void)rag, (void)rago, (void)sl;
  g_cd_err = "built without cuDNN";
  return 1;
#endif
}

int fecd_graphs(void* ctx) {
#ifdef FE_CUDNN
  return ctx ? ((Ctx*)ctx)->nbuilt : 0;
#else
  (void)ctx;
  return 0;
#endif
}

size_t fecd_bytes(void* ctx) {
#ifdef FE_CUDNN
  if (!ctx) return 0;
  auto* c = (Ctx*)ctx;
  return (size_t)c->ws_bytes;
#else
  (void)ctx;
  return 0;
#endif
}

void fecd_free(void* ctx) {
#ifdef FE_CUDNN
  auto* c = (Ctx*)ctx;
  if (!c) return;
  if (c->log) fprintf(stderr, "cudnn attention: %d graphs, %.2f s building\n", c->nbuilt, c->build_s);
  c->graphs.clear();
  c->kcache.clear();
  if (c->h) fe::detail::destroy_handle(c->h);
  if (c->ws) cudaFree(c->ws);
  delete c;
#else
  (void)ctx;
#endif
}
}
