# Safetensors implementation for [Zigrad](https://github.com/Marco-Christiani/zigrad)

## Benchmarks
```
Rust (official impl)
Serialize 10_MB       time:   [521.39 µs 522.73 µs 524.21 µs]
Deserialize 10_MB      time:   [2.6482 µs 2.6622 µs 2.6763 µs]

Zig (ours)
Serialize 10MB [min=328.709, avg=339.422, max=351.483]
Deserialize 10MB [min=0.902, avg=0.982, max=1.042]
```

[Plots courtesy of Claude](https://claude.site/artifacts/cc6dcc58-3545-4541-b25e-871a6ddc4e6f)
