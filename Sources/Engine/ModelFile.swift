import Foundation

// The .ill model file: a 16 KiB header (magic, version, format, tensor table of (offset, size)
// pairs) followed by the tensors, each 256-byte aligned, already in the layout the kernels read.
// The whole file is mmapped and wrapped in one MTLBuffer without copying.

public enum WeightFormat: UInt32, CustomStringConvertible {
    case f16 = 0
    case q8 = 1
    case q4 = 2
    /// Affine 4-bit, one group per (row, 28- or 38-wide lane slice), 8-bit E5M3 scale + 6-bit zero
    /// point, rows interleaved in 16-row blocks for lane-sliced kernels (4.5 / 4.37 bits per weight).
    case q4a = 3

    public var description: String {
        switch self {
        case .f16: return "f16"
        case .q8: return "q8"
        case .q4: return "q4"
        case .q4a: return "q4a"
        }
    }

    public init?(name: String) {
        switch name {
        case "f16": self = .f16
        case "q8": self = .q8
        case "q4": self = .q4
        case "q4a": self = .q4a
        default: return nil
        }
    }
}

public enum Dims {
    public static let layers = 24
    public static let hidden = 896
    public static let ff = 4864
    public static let heads = 14
    public static let kvHeads = 2
    public static let headDim = 64
    public static let qkvRows = 1152
    public static let vocab = 151936
    public static let ropeTheta = 1_000_000.0
}

public enum TensorID {
    public static let embed = 0
    public static let embedSB = 1
    public static let finalNorm = 2
    public static let perLayer = 12
    /// 2-bit sketch of the q4 embedding (high bits of every code) for the two-stage lm_head.
    public static let embedHi = 3 + Dims.layers * perLayer
    public static let count = embedHi + 1

    public static let attnNorm = 0, wqkv = 1, wqkvSB = 2, bqkv = 3, wo = 4, woSB = 5
    public static let mlpNorm = 6, wgu = 7, wguSB = 8, wd = 9, wdSB = 10

    @inline(__always) public static func layer(_ l: Int, _ k: Int) -> Int { 3 + l * perLayer + k }
}

let modelMagic: UInt32 = 0x314D_4C49 // "ILM1"
let headerSize = 16384
let pageSize = 16384

public final class ModelFile {
    public let path: String
    public let base: UnsafeMutableRawPointer
    public let length: Int // mapped length (multiple of the page size)
    public let format: WeightFormat
    private let table: UnsafePointer<UInt64>

    public init(path: String) throws {
        self.path = path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw EngineError("cannot open \(path)") }
        defer { close(fd) }
        var sb = stat()
        guard fstat(fd, &sb) == 0 else { throw EngineError("cannot stat \(path)") }
        let size = Int(sb.st_size)
        guard size >= headerSize, size % pageSize == 0 else { throw EngineError("\(path): bad size \(size)") }
        guard let p = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0), p != MAP_FAILED else {
            throw EngineError("mmap failed for \(path)")
        }
        base = p
        length = size
        let h = UnsafeRawPointer(p)
        guard h.load(as: UInt32.self) == modelMagic else { throw EngineError("\(path): not an instantllm model") }
        guard h.load(fromByteOffset: 4, as: UInt32.self) == 1 else { throw EngineError("\(path): unsupported version") }
        guard let f = WeightFormat(rawValue: h.load(fromByteOffset: 8, as: UInt32.self)) else {
            throw EngineError("\(path): unknown weight format")
        }
        format = f
        guard h.load(fromByteOffset: 12, as: UInt32.self) == UInt32(TensorID.count) else {
            throw EngineError("\(path): tensor table mismatch")
        }
        table = h.advanced(by: 16).assumingMemoryBound(to: UInt64.self)
    }

    deinit { munmap(base, length) }

    @inline(__always) public func offset(_ id: Int) -> Int { Int(table[2 * id]) }
    @inline(__always) public func size(_ id: Int) -> Int { Int(table[2 * id + 1]) }
    public func pointer(_ id: Int) -> UnsafeRawPointer { UnsafeRawPointer(base).advanced(by: offset(id)) }
}

public struct EngineError: Error, CustomStringConvertible {
    public let description: String
    public init(_ d: String) { description = d }
}

// Writes tensors in table order with 256-byte alignment.
final class ModelWriter {
    private var blobs: [(Int, Data)] = []

    func add(_ id: Int, _ data: Data) { blobs.append((id, data)) }

    func write(to path: String, format: WeightFormat) throws {
        var table = [UInt64](repeating: 0, count: 2 * TensorID.count)
        var cursor = headerSize
        var layout: [(Int, Data)] = []
        for (id, d) in blobs.sorted(by: { $0.0 < $1.0 }) {
            cursor = (cursor + 255) & ~255
            table[2 * id] = UInt64(cursor)
            table[2 * id + 1] = UInt64(d.count)
            layout.append((cursor, d))
            cursor += d.count
        }
        let total = (cursor + pageSize - 1) / pageSize * pageSize
        var out = Data(count: total)
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            let p = raw.baseAddress!
            p.storeBytes(of: modelMagic, as: UInt32.self)
            p.storeBytes(of: UInt32(1), toByteOffset: 4, as: UInt32.self)
            p.storeBytes(of: format.rawValue, toByteOffset: 8, as: UInt32.self)
            p.storeBytes(of: UInt32(TensorID.count), toByteOffset: 12, as: UInt32.self)
            for (i, v) in table.enumerated() { p.storeBytes(of: v, toByteOffset: 16 + 8 * i, as: UInt64.self) }
            for (off, d) in layout {
                d.withUnsafeBytes { src in p.advanced(by: off).copyMemory(from: src.baseAddress!, byteCount: d.count) }
            }
        }
        try out.write(to: URL(fileURLWithPath: path))
    }
}
