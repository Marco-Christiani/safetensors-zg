//! Benchmarking. See [benchmark.rs](https://github.com/huggingface/safetensors/blob/7bf65ad7d56be10331dd9c15b67d82d1c5f39cc0/safetensors/benches/benchmark.rs).
const std = @import("std");
const stz = @import("root.zig");

/// Benchmark helper to create a 2MB tensor
fn getSampleData(allocator: std.mem.Allocator) !struct {
    data: []u8,
    shape: []usize,
    dtype: stz.Dtype,
} {
    // 1000 x 500 elements = 500,000 elements; F32 = 4 bytes each => 2,000,000 bytes (~2 MB)
    const shape = try allocator.dupe(usize, &[_]usize{ 1000, 500 });
    const dtype = stz.Dtype.f32;
    const n = shape[0] * shape[1] * dtype.size();
    const data = try allocator.alloc(u8, n);
    @memset(data, 0);
    return .{ .data = data, .shape = shape, .dtype = dtype };
}

const BmResult = struct {
    const Self = @This();
    avg: f64,
    min: f64,
    max: f64,

    fn _ns_to_us(x: f64) f64 {
        return x / std.time.ns_per_us;
    }

    pub fn as_us(self: Self) Self {
        return Self{
            .avg = _ns_to_us(self.avg),
            .min = _ns_to_us(self.min),
            .max = _ns_to_us(self.max),
        };
    }

    pub fn print_to_writer(self: Self, writer: anytype) !void {
        try writer.print("[min={d:.3}, avg={d:.3}, max={d:.3}]", .{ self.min, self.avg, self.max });
    }
};

/// Benchmark serialization: builds 5 tensors (total ~10 MB) and measures the time to serialize.
/// Should match the official [benchmark.rs](https://github.com/huggingface/safetensors/blob/7bf65ad7d56be10331dd9c15b67d82d1c5f39cc0/safetensors/benches/benchmark.rs)
fn benchSerialize(allocator: std.mem.Allocator) !BmResult {
    const sample = try getSampleData(allocator);
    defer allocator.free(sample.data);
    defer allocator.free(sample.shape);

    const n_layers = 5;
    var tensorList = std.ArrayList(stz.Tensor).init(allocator);
    defer tensorList.deinit();

    for (0..n_layers) |i| {
        const name = try std.fmt.allocPrint(allocator, "weight{}", .{i});
        try tensorList.append(stz.Tensor{
            .name = name,
            .dtype = sample.dtype,
            .shape = sample.shape,
            .data = sample.data,
        });
    }

    defer for (tensorList.items) |tensor| {
        allocator.free(tensor.name);
    };

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    // warmup
    for (0..5) |_| {
        _ = try stz.serialize_tensors(tensorList, arena.allocator());
        defer _ = arena.reset(.retain_capacity);
    }

    var total: usize = 0;
    var min: usize = std.math.maxInt(u63);
    var max: usize = 0;
    var timer = try std.time.Timer.start();
    const n = 100;
    for (0..n) |_| {
        timer.reset();
        _ = try stz.serialize_tensors(tensorList, arena.allocator());
        defer _ = arena.reset(.retain_capacity);
        const time = timer.lap();
        total += time;
        if (time < min) min = time;
        if (time > max) max = time;
    }

    return BmResult{
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
    var tensorList = std.ArrayList(stz.Tensor).init(allocator);
    defer tensorList.deinit();
    for (0..n_layers) |i| {
        const name = try std.fmt.allocPrint(allocator, "weight{}", .{i});
        try tensorList.append(stz.Tensor{
            .name = name,
            .dtype = sample.dtype,
            .shape = sample.shape,
            .data = sample.data,
        });
    }
    defer for (tensorList.items) |tensor| {
        allocator.free(tensor.name);
    };
    const serialized = try stz.serialize_tensors(tensorList, allocator);
    defer allocator.free(serialized);

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
        defer fba.reset();
        const time = timer.lap();
        total += time;
        if (time < min) min = time;
        if (time > max) max = time;
    }

    return BmResult{
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
