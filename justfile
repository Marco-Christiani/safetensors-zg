default: build

build:
  zig build test

bench enable_sort:
  #!/usr/bin/env bash
  echo "========== Benchmark Environment =========="
  echo -n "Zig Version:"
  zig version
 
  echo -e "\nSystem Info:"
  uname -a
 
  echo -e "\nCPU Info:"
  lscpu | grep -E 'Model name|Architecture|CPU\(s\)|Thread|MHz' || sysctl -a | grep machdep.cpu || true
 
  echo -e "\nMemory Info:"
  free -h || vm_stat || true
 
  echo -e "\nCompiling..."
  set -x
  zig build -Doptimize=ReleaseFast -Denable_sort={{enable_sort}}
  set +x

   echo -e "\nRunning benchmark..."
  ./zig-out/bin/benchmark_stz

bench-rs safetensors-clone:
  cd {{ safetensors-clone }}/safetensors && cargo bench
