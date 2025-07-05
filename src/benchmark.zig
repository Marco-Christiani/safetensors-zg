//! Benchmarking. See [benchmark.rs](https://github.com/huggingface/safetensors/blob/7bf65ad7d56be10331dd9c15b67d82d1c5f39cc0/safetensors/benches/benchmark.rs).
//!
//! # ADR
//!
//!   - Only have f32 as benchmark.rs only runs f32 hardcoded.
//!   - It is trivial to use comptime generics in zig but ideally we have something to compare to,
//!   so if more types are ever added to the saftensors benchmarks then lets update this.
const std = @import("std");
const stz = @import("root.zig");

/// Benchmark helper to create a 2MB tensor
fn getSampleData(allocator: std.mem.Allocator) !struct {
    data: []align(8) u8,
    shape: []usize,
    dtype: stz.Dtype,
} {
    // See comments regarding f32
    // 1000 x 500 elements = 500,000 elements; F32 = 4 bytes each => 2,000,000 bytes (~2 MB)
    const shape = try allocator.dupe(usize, &[_]usize{ 1000, 500 });
    const dtype = stz.Dtype.f32;
    const n = shape[0] * shape[1] * dtype.size();
    const data = try allocator.alignedAlloc(u8, 8, n);
    @memset(data, 0);
    return .{ .data = data, .shape = shape, .dtype = dtype };
}

const BmResult = struct {
    const Self = @This();
    total_mb: usize,
    avg: f64,
    min: f64,
    max: f64,

    fn _ns_to_us(x: f64) f64 {
        return x / std.time.ns_per_us;
    }

    pub fn as_us(self: Self) Self {
        return Self{
            .total_mb = self.total_mb,
            .avg = _ns_to_us(self.avg),
            .min = _ns_to_us(self.min),
            .max = _ns_to_us(self.max),
        };
    }

    pub fn print_to_writer(self: Self, writer: anytype) !void {
        try writer.print("{d}_MB [min={d:.3}, avg={d:.3}, max={d:.3}]", .{ self.total_mb, self.min, self.avg, self.max });
    }
};

/// Benchmark serialization: builds 5 tensors (total ~10 MB) and measures the time to serialize.
/// Should match the official [benchmark.rs](https://github.com/huggingface/safetensors/blob/7bf65ad7d56be10331dd9c15b67d82d1c5f39cc0/safetensors/benches/benchmark.rs)
fn benchSerialize(allocator: std.mem.Allocator) !BmResult {
    const sample = try getSampleData(allocator);
    defer allocator.free(sample.data);
    defer allocator.free(sample.shape);

    const n_layers = 5;
    var total_mb: usize = 0;
    var tensorList = std.ArrayList(stz.Tensor).init(allocator);
    defer tensorList.deinit();

    inline for (0..n_layers) |i| {
        const name = std.fmt.comptimePrint("weight{d}", .{i});
        try tensorList.append(stz.Tensor{
            .name = name,
            .dtype = sample.dtype,
            .shape = sample.shape,
            .data = sample.data,
        });
        // See comments regarding f32
        total_mb += sample.data.len / @as(usize, 1e6);
    }
    std.debug.print("total_mb: {d} n_layers*2: {d}\n", .{ total_mb, n_layers * 2 });
    // Each tensor is 2MB
    std.debug.assert(total_mb == n_layers * 2);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    // warmup
    for (0..5) |_| {
        _ = try stz.serialize_tensors(tensorList, arena.allocator());
        _ = arena.reset(.retain_capacity);
    }

    var total: usize = 0;
    var min: usize = std.math.maxInt(u63);
    var max: usize = 0;
    var timer = try std.time.Timer.start();
    const n = 100;
    for (0..n) |_| {
        timer.reset();
        _ = try stz.serialize_tensors(tensorList, arena.allocator());
        const time = timer.lap();
        _ = arena.reset(.retain_capacity);
        total += time;
        if (time < min) min = time;
        if (time > max) max = time;
    }

    return BmResult{
        .total_mb = total_mb,
        .avg = @as(f64, @floatFromInt(total)) / n,
        .min = @floatFromInt(min),
        .max = @floatFromInt(max),
    };
}

/// Benchmark deserialization: takes a serialized buffer (from 5 tensors) and measures the time to load.
/// Should match the official [benchmark.rs](https://github.com/huggingface/safetensors/blob/7bf65ad7d56be10331dd9c15b67d82d1c5f39cc0/safetensors/benches/benchmark.rs)
fn benchDeserialize(allocator: std.mem.Allocator) !BmResult {
    const sample = try getSampleData(allocator);
    defer allocator.free(sample.data);
    defer allocator.free(sample.shape);

    const n_layers = 5;
    var total_mb: usize = 0;
    var tensorList = std.ArrayList(stz.Tensor).init(allocator);
    defer tensorList.deinit();
    inline for (0..n_layers) |i| {
        const name = std.fmt.comptimePrint("weight{d}", .{i});
        try tensorList.append(stz.Tensor{
            .name = name,
            .dtype = sample.dtype,
            .shape = sample.shape,
            .data = sample.data,
        });
        // See comments regarding f32
        total_mb += sample.data.len / @as(usize, 1e6);
    }
    std.debug.print("total_mb: {d} n_layers*2: {d}\n", .{ total_mb, n_layers * 2 });
    // Each tensor is 2MB
    std.debug.assert(total_mb == n_layers * 2);

    const serialized = try stz.serialize_tensors(tensorList, allocator);
    defer allocator.free(serialized);

    // Just need some space to parse the header (can be up to `stz.MAX_HEADER_SIZE`)
    var buf: [1_000_000]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);

    // warmup
    for (0..5) |_| {
        _ = try stz.SafeTensorsFile.deserialize(serialized, fba.allocator());
        defer fba.reset();
    }

    var total: usize = 0;
    var min: usize = std.math.maxInt(u63);
    var max: usize = 0;
    var timer = try std.time.Timer.start();
    const n = 100;
    for (0..n) |_| {
        timer.reset();
        _ = try stz.SafeTensorsFile.deserialize(serialized, fba.allocator());
        const time = timer.lap();
        fba.reset();
        total += time;
        if (time < min) min = time;
        if (time > max) max = time;
    }

    return BmResult{
        .total_mb = total_mb,
        .avg = @as(f64, @floatFromInt(total)) / n,
        .min = @floatFromInt(min),
        .max = @floatFromInt(max),
    };
}

/// Benchmark runner.
pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    const ser_result = try benchSerialize(std.heap.raw_c_allocator);
    const dser_result = try benchDeserialize(std.heap.raw_c_allocator);

    try stdout.print("Serialization: ", .{});
    try ser_result.as_us().print_to_writer(stdout);
    try stdout.print("\nDeserialization: ", .{});
    try dser_result.as_us().print_to_writer(stdout);
    try stdout.print("\n", .{});
}
