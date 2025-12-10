"""
run.py - Development helper for CUTLASS NVFP4 GEMM kernel

This script reads kernel.cu, injects it into submission.py, and runs the tests.

Usage:
    python run.py test           # Run correctness tests
    python run.py bench          # Run benchmarks  
    python run.py all            # Run both tests and benchmarks
    python run.py compile        # Just compile the kernel (no tests)
    python run.py test --kernel cublaslt   # Use cuBLASLt kernel
"""

import sys
import os
import subprocess
import shutil
import re
import math
from pathlib import Path

DEV_DIR = Path(__file__).parent
PROBLEM_DIR = DEV_DIR.parent
KERNEL_CU = DEV_DIR / "kernel.cu"
KERNEL_CUBLASLT = DEV_DIR / "kernel_cublaslt.cu"
SUBMISSION_PY = PROBLEM_DIR / "submission.py"
SUBMISSION_BACKUP = PROBLEM_DIR / "submission.py.backup"

SUBMISSION_TEMPLATE = r'''import torch
import sys
import io
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

GEMM_CPP = r"""
#include <torch/extension.h>

torch::Tensor cutlass_nvfp4_gemm(
    torch::Tensor A,       // [M, K/2, L] float4_e2m1fn_x2 
    torch::Tensor B,       // [N, K/2, L] float4_e2m1fn_x2
    torch::Tensor SFA,     // Scale factors for A (permuted layout)
    torch::Tensor SFB,     // Scale factors for B (permuted layout)
    torch::Tensor C,       // [M, N, L] float16 output
    float alpha,
    float beta
);
"""

GEMM_CUDA = r"""
{KERNEL_CODE}
"""

_cutlass_gemm = None

def _get_kernel():
    """Lazy load and cache the compiled CUTLASS kernel."""
    global _cutlass_gemm
    if _cutlass_gemm is None:
        # Handle stdout/stderr for multiprocessing compatibility
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if sys.stdout is None:
            sys.stdout = io.StringIO()
        if sys.stderr is None:
            sys.stderr = io.StringIO()
        try:
            # CUTLASS include paths - adjust if needed for your environment
            cutlass_include = "/workspace/cutlass/include"
            cutlass_tools_include = "/workspace/cutlass/tools/util/include"
            
            _cutlass_gemm = load_inline(
                name="cutlass_nvfp4_gemm",
                cpp_sources=[GEMM_CPP],
                cuda_sources=[GEMM_CUDA],
                functions=["cutlass_nvfp4_gemm"],
                extra_cuda_cflags=[
                    "-std=c++17",
                    "-gencode=arch=compute_100a,code=sm_100a",
                    "--ptxas-options=--gpu-name=sm_100a",
                    "-O3",
                    "-w",
                    "--use_fast_math",
                    "-allow-unsupported-compiler",
                    f"-I{cutlass_include}",
                    f"-I{cutlass_tools_include}",
                    "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
                ],
                extra_ldflags=["-lcuda"],
                verbose=False,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    return _cutlass_gemm



def compile_kernel():
    """Pre-compile the kernel to exclude compilation time from benchmarks."""
    _get_kernel()


def custom_kernel(data: input_t) -> output_t:
    """
    Run CUTLASS GEMM on NVFP4 block-scaled inputs.
    
    Args:
        data: Tuple of (a, b, sfa, sfb, sfa_permuted, sfb_permuted, c)
            - a: [M, K/2, L] float4_e2m1fn_x2
            - b: [N, K/2, L] float4_e2m1fn_x2
            - sfa: [M, K//16, L] float8_e4m3fn (CPU layout)
            - sfb: [N, K//16, L] float8_e4m3fn (CPU layout)
            - sfa_permuted: Permuted scale factors for CUTLASS
            - sfb_permuted: Permuted scale factors for CUTLASS
            - c: [M, N, L] float16 output buffer
    
    Returns:
        c: [M, N, L] float16 output tensor
    """
    a, b, sfa_cpu, sfb_cpu, sfa_permuted, sfb_permuted, c = data
    kernel = _get_kernel()
    
    return kernel.cutlass_nvfp4_gemm(
        a, b, 
        sfa_permuted, sfb_permuted,
        c,
        1.0,  # alpha
        0.0   # beta
    )
'''

# Template for cuBLASLt submission - note different function name and extra linker flags
SUBMISSION_TEMPLATE_CUBLASLT = r'''import torch
import sys
import io
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

GEMM_CPP = r"""
#include <torch/extension.h>

torch::Tensor cublaslt_nvfp4_gemm(
    torch::Tensor A,       // [M, K/2, L] float4_e2m1fn_x2 
    torch::Tensor B,       // [N, K/2, L] float4_e2m1fn_x2
    torch::Tensor SFA,     // Scale factors for A
    torch::Tensor SFB,     // Scale factors for B
    torch::Tensor C,       // [M, N, L] float16 output
    float alpha,
    float beta
);
"""

GEMM_CUDA = r"""
{KERNEL_CODE}
"""

_cublaslt_gemm = None

def _get_kernel():
    """Lazy load and cache the compiled cuBLASLt kernel."""
    global _cublaslt_gemm
    if _cublaslt_gemm is None:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if sys.stdout is None:
            sys.stdout = io.StringIO()
        if sys.stderr is None:
            sys.stderr = io.StringIO()
        try:
            _cublaslt_gemm = load_inline(
                name="cublaslt_nvfp4_gemm",
                cpp_sources=[GEMM_CPP],
                cuda_sources=[GEMM_CUDA],
                functions=["cublaslt_nvfp4_gemm"],
                extra_cuda_cflags=[
                    "-std=c++17",
                    "-gencode=arch=compute_100a,code=sm_100a",
                    "--ptxas-options=--gpu-name=sm_100a",
                    "-O3",
                    "-w",
                    "--use_fast_math",
                    "-allow-unsupported-compiler",
                ],
                extra_ldflags=["-lcuda", "-lcublasLt", "-lcublas"],
                verbose=False,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    return _cublaslt_gemm


def compile_kernel():
    """Pre-compile the kernel to exclude compilation time from benchmarks."""
    _get_kernel()


def custom_kernel(data: input_t) -> output_t:
    """Run cuBLASLt GEMM on NVFP4 block-scaled inputs."""
    a, b, sfa_cpu, sfb_cpu, sfa_permuted, sfb_permuted, c = data
    kernel = _get_kernel()
    
    # cuBLASLt uses non-permuted (CPU) scale factors - kernel converts to blocked format
    return kernel.cublaslt_nvfp4_gemm(
        a, b, 
        sfa_cpu, sfb_cpu,  # Pass CPU-layout scales, NOT permuted
        c,
        1.0,  # alpha
        0.0   # beta
    )
'''

def read_kernel(kernel_file=None):
    """Read the kernel file."""
    if kernel_file is None:
        kernel_file = KERNEL_CU
    
    if not kernel_file.exists():
        print(f"Error: {kernel_file} not found!")
        sys.exit(1)
    
    with open(kernel_file, 'r') as f:
        return f.read()


def generate_submission(kernel_code: str, use_cublaslt: bool = False) -> str:
    """Generate submission.py content with the kernel code injected."""
    template = SUBMISSION_TEMPLATE_CUBLASLT if use_cublaslt else SUBMISSION_TEMPLATE
    return template.replace("{KERNEL_CODE}", kernel_code)


def backup_submission():
    """Backup the current submission.py if it exists."""
    if SUBMISSION_PY.exists() and not SUBMISSION_BACKUP.exists():
        shutil.copy(SUBMISSION_PY, SUBMISSION_BACKUP)
        print(f"Backed up submission.py to {SUBMISSION_BACKUP}")


def write_submission(content: str):
    """Write the generated submission.py."""
    with open(SUBMISSION_PY, 'w') as f:
        f.write(content)
    print(f"Generated {SUBMISSION_PY}")


def run_command(cmd: str, capture_output: bool = False):
    """Run a shell command in the problem directory."""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print('='*60 + "\n")
    
    if capture_output:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=PROBLEM_DIR,
            capture_output=True,
            text=True,
        )
        # Print the output as it would normally appear
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode, result.stdout
    else:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=PROBLEM_DIR,
        )
        return result.returncode, None


def extract_benchmark_means(output: str) -> list:
    """Extract mean values from benchmark output."""
    means = []
    for line in output.split('\n'):
        match = re.match(r'benchmark\.\d+\.mean:\s*([0-9.]+)', line)
        if match:
            means.append(float(match.group(1)))
    return means


def compute_geometric_mean(values: list) -> float:
    """Compute geometric mean of a list of values."""
    if not values:
        return 0.0
    log_sum = sum(math.log(v) for v in values if v > 0)
    return math.exp(log_sum / len(values))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    # Check for --kernel flag
    use_cublaslt = False
    kernel_file = KERNEL_CU
    if "--kernel" in sys.argv:
        idx = sys.argv.index("--kernel")
        if idx + 1 < len(sys.argv):
            kernel_name = sys.argv[idx + 1].lower()
            if kernel_name == "cublaslt":
                use_cublaslt = True
                kernel_file = KERNEL_CUBLASLT
                print("Using cuBLASLt kernel...")
            elif kernel_name == "cutlass":
                kernel_file = KERNEL_CU
                print("Using CUTLASS kernel...")
            else:
                print(f"Unknown kernel: {kernel_name}")
                print("Available kernels: cutlass, cublaslt")
                sys.exit(1)
    
    # Read kernel and generate submission
    print(f"Reading {kernel_file.name}...")
    kernel_code = read_kernel(kernel_file)
    
    print("Generating submission.py...")
    submission_content = generate_submission(kernel_code, use_cublaslt)
    
    # backup_submission()
    write_submission(submission_content)
    
    # Execute based on command
    if command == "test":
        ret, _ = run_command("bash run.sh test")
        return ret
    elif command == "bench" or command == "benchmark":
        ret, output = run_command("bash run.sh benchmark", capture_output=True)
        if output:
            means = extract_benchmark_means(output)
            if means:
                geomean = compute_geometric_mean(means)
                print(f"\n{'='*60}")
                print(f"BENCHMARK SUMMARY")
                print('='*60)
                for i, m in enumerate(means):
                    print(f"  Benchmark {i}: {m/1000:.2f} µs")
                print(f"\n  Geometric Mean: {geomean/1000:.2f} µs ({geomean:.2f} ns)")
                print('='*60)
        return ret
    elif command == "all":
        ret, _ = run_command("bash run.sh test")
        if ret == 0:
            ret, output = run_command("bash run.sh benchmark", capture_output=True)
            if output:
                means = extract_benchmark_means(output)
                if means:
                    geomean = compute_geometric_mean(means)
                    print(f"\n{'='*60}")
                    print(f"BENCHMARK SUMMARY")
                    print('='*60)
                    for i, m in enumerate(means):
                        print(f"  Benchmark {i}: {m/1000:.2f} µs")
                    print(f"\n  Geometric Mean: {geomean/1000:.2f} µs ({geomean:.2f} ns)")
                    print('='*60)
        return ret
    elif command == "compile":
        # Just try to import and compile
        print("\nCompiling kernel...")
        os.chdir(PROBLEM_DIR)
        sys.path.insert(0, str(PROBLEM_DIR))
        from submission import compile_kernel
        compile_kernel()
        print("Compilation successful!")
        return 0
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
