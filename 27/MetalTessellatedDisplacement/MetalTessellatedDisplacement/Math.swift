
import simd
import Accelerate

func align(_ value: Int, upTo alignment: Int) -> Int {
    return ((value + alignment - 1) / alignment) * alignment
}

func gcd(_ m: Int, _ n: Int) -> Int {
    var a = 0
    var b = max(m, n)
    var r = min(m, n)

    while r != 0 {
        a = b
        b = r
        r = a % b
    }
    return b
}

func lcm(_ m: Int, _ n: Int) -> Int {
    return m * n / gcd(m, n)
}

extension UInt16 {
    init(representingAsFloat16 x: Float) {
        var input = x
        var output: UInt16 = 0
        withUnsafeMutablePointer(to: &input) { inputPtr in
            withUnsafeMutablePointer(to: &output) { outputPtr in
                var src = vImage_Buffer(data: inputPtr, height: 1, width: 1, rowBytes: 4)
                var dst = vImage_Buffer(data: outputPtr, height: 1, width: 1, rowBytes: 2)
                vImageConvert_PlanarFtoPlanar16F(&src, &dst, .zero)
            }
        }
        self.init(output)
    }
}

extension SIMD4 {
    var xyz: SIMD3<Scalar> {
        return SIMD3<Scalar>(x, y, z)
    }
}

extension float4x4 {
    init(scale2D s: SIMD2<Float>) {
        self.init(SIMD4<Float>(s.x,   0, 0, 0),
                  SIMD4<Float>(  0, s.y, 0, 0),
                  SIMD4<Float>(  0,   0, 1, 0),
                  SIMD4<Float>(  0,   0, 0, 1))
    }

    init(rotateZ zRadians: Float) {
        let s = sin(zRadians)
        let c = cos(zRadians)
        self.init(SIMD4<Float>( c, s, 0, 0),
                  SIMD4<Float>(-s, c, 0, 0),
                  SIMD4<Float>( 0, 0, 1, 0),
                  SIMD4<Float>( 0, 0, 0, 1))
    }

    init(translate2D t: SIMD2<Float>) {
        self.init(SIMD4<Float>(  1,   0, 0, 0),
                  SIMD4<Float>(  0,   1, 0, 0),
                  SIMD4<Float>(  0,   0, 1, 0),
                  SIMD4<Float>(t.x, t.y, 0, 1))
    }

    init(scale s: SIMD3<Float>) {
        self.init(SIMD4<Float>(s.x,   0,   0, 0),
                  SIMD4<Float>(  0, s.y,   0, 0),
                  SIMD4<Float>(  0,   0, s.z, 0),
                  SIMD4<Float>(  0,   0,   0, 1))
    }

    init(rotateAbout axis: SIMD3<Float>, byAngle radians: Float) {
        let x = axis.x
        let y = axis.y
        let z = axis.z
        let s = sin(radians)
        let c = cos(radians)

        self.init(
            SIMD4<Float>(x * x + (1 - x * x) * c, x * y * (1 - c) - z * s, x * z * (1 - c) + y * s, 0),
            SIMD4<Float>(x * y * (1 - c) + z * s, y * y + (1 - y * y) * c, y * z * (1 - c) - x * s, 0),
            SIMD4<Float>(x * z * (1 - c) - y * s, y * z * (1 - c) + x * s, z * z + (1 - z * z) * c, 0),
            SIMD4<Float>(                      0,                       0,                       0, 1)
        )
    }

    init(translate t: SIMD3<Float>) {
        self.init(SIMD4<Float>(  1,   0,   0, 0),
                  SIMD4<Float>(  0,   1,   0, 0),
                  SIMD4<Float>(  0,   0,   1, 0),
                  SIMD4<Float>(t.x, t.y, t.z, 1))
    }

    init(lookAt at: SIMD3<Float>, from: SIMD3<Float>, up: SIMD3<Float>) {
        let zNeg = normalize(at - from)
        let x = normalize(cross(zNeg, up))
        let y = normalize(cross(x, zNeg))
        self.init(SIMD4<Float>(x, 0),
                  SIMD4<Float>(y, 0),
                  SIMD4<Float>(-zNeg, 0),
                  SIMD4<Float>(from, 1))
    }

    init(orthographicProjectionWithLeft left: Float, top: Float,
         right: Float, bottom: Float, near: Float, far: Float)
    {
        let sx = 2 / (right - left)
        let sy = 2 / (top - bottom)
        let sz = 1 / (near - far)
        let tx = (left + right) / (left - right)
        let ty = (top + bottom) / (bottom - top)
        let tz = near / (near - far)
        self.init(SIMD4<Float>(sx,  0,  0, 0),
                  SIMD4<Float>( 0, sy,  0, 0),
                  SIMD4<Float>( 0,  0, sz, 0),
                  SIMD4<Float>(tx, ty, tz, 1))
    }

    init(perspectiveProjectionFoVY fovYRadians: Float,
         aspectRatio: Float,
         near: Float,
         far: Float)
    {
        let sy = 1 / tan(fovYRadians * 0.5)
        let sx = sy / aspectRatio
        let zRange = far - near
        let sz = -(far + near) / zRange
        let tz = -2 * far * near / zRange
        self.init(SIMD4<Float>(sx, 0,  0,  0),
                  SIMD4<Float>(0, sy,  0,  0),
                  SIMD4<Float>(0,  0, sz, -1),
                  SIMD4<Float>(0,  0, tz,  0))
    }

    var upperLeft3x3: float3x3 {
        return float3x3(columns.0.xyz, columns.1.xyz, columns.2.xyz)
    }
}

extension simd_quatf {
    func rotate(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let u = imag
        let w = real
        let vr = 2.0 * dot(u, v) * u + (w * w - dot(u, u)) * v + 2.0 * w * cross(u, v)
        return vr
    }
}
