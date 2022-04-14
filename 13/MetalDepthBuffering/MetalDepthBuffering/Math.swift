
import simd

func align(_ value: Int, upTo alignment: Int) -> Int {
    return ((value + alignment - 1) / alignment) * alignment
}

extension simd_float4x4 {
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
        let s = sin(radians)
        let c = cos(radians)

        self.init(
            SIMD4<Float>(axis.x * axis.x + (1 - axis.x * axis.x) * c,
                         axis.x * axis.y * (1 - c) - axis.z * s,
                         axis.x * axis.z * (1 - c) + axis.y * s,
                         0),
            SIMD4<Float>(axis.x * axis.y * (1 - c) + axis.z * s,
                         axis.y * axis.y + (1 - axis.y * axis.y) * c,
                         axis.y * axis.z * (1 - c) - axis.x * s,
                         0),
            SIMD4<Float>(axis.x * axis.z * (1 - c) - axis.y * s,
                         axis.y * axis.z * (1 - c) + axis.x * s,
                         axis.z * axis.z + (1 - axis.z * axis.z) * c,
                         0),
            SIMD4<Float>(0, 0, 0, 1)
        )
    }

    init(translate t: SIMD3<Float>) {
        self.init(SIMD4<Float>(  1,   0,   0, 0),
                  SIMD4<Float>(  0,   1,   0, 0),
                  SIMD4<Float>(  0,   0,   1, 0),
                  SIMD4<Float>(t.x, t.y, t.z, 1))
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
}
