
import Foundation
import Metal
import simd

enum LightType : UInt32 {
    case ambient
    case directional
}

class Light {
    var type = LightType.directional
    var color = SIMD3<Float>(1, 1, 1)
    var intensity: Float = 1.0

    var worldTransform: simd_float4x4 = matrix_identity_float4x4

    var position: SIMD3<Float> {
        return worldTransform.columns.3.xyz
    }

    var direction: SIMD3<Float> {
        return -worldTransform.columns.2.xyz
    }

    var castsShadows = false
    var shadowTexture: MTLTexture?

    var projectionMatrix: simd_float4x4 {
        return simd_float4x4(orthographicProjectionWithLeft: -1.5,
                             top: 1.5,
                             right: 1.5,
                             bottom: -1.5,
                             near: 0,
                             far: 10)
    }
}
