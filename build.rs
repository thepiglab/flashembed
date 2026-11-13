// 1. ONNX protobuf types for the graph rewriter (src/quant.rs), compiled with protox (pure Rust; no protoc needed).
// 2. With feature `cuda`: compile cuda/fegpu.cu with nvcc (CUDA_HOME or nvcc on PATH) and link cudart + cuBLAS.
// FLASHEMBED_CUDA_ARCH: comma-separated SM versions (default "100,120": B200 and RTX PRO 6000 Blackwell; PTX for the
// highest one is embedded too, so newer GPUs JIT it).
fn main() {
    println!("cargo:rerun-if-changed=proto/onnx.proto");
    let fds = protox::compile(["proto/onnx.proto"], ["proto"]).expect("proto/onnx.proto");
    prost_build::Config::new().compile_fds(fds).expect("prost codegen");

    println!("cargo:rerun-if-changed=cuda/fegpu.cu");
    println!("cargo:rerun-if-env-changed=FLASHEMBED_CUDA_ARCH");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let archs = std::env::var("FLASHEMBED_CUDA_ARCH").unwrap_or_else(|_| "100,120".into());
    let mut b = cc::Build::new();
    b.cuda(true).cudart("shared").file("cuda/fegpu.cu").flag("-O3").flag("-std=c++17")
        .flag("-lineinfo").flag("-Xptxas=-v").warnings(false);
    let list: Vec<&str> = archs.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    for a in &list {
        b.flag(format!("-gencode=arch=compute_{a},code=sm_{a}"));
    }
    if let Some(last) = list.last() {
        b.flag(format!("-gencode=arch=compute_{last},code=compute_{last}"));
    }
    b.compile("fegpu");
    if let Some(home) = std::env::var_os("CUDA_HOME").or_else(|| std::env::var_os("CUDA_PATH")) {
        let home = std::path::PathBuf::from(home);
        for d in ["lib64", "lib", "targets/x86_64-linux/lib"] {
            println!("cargo:rustc-link-search=native={}", home.join(d).display());
        }
    }
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
}
