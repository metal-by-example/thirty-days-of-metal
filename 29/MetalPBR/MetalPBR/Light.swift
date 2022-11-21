
import Foundation
import Cocoa
import Metal
import simd

enum LightType : UInt32 {
    case directional
    case omni
}

class Light {
    var type = LightType.directional
    var color = SIMD3<Float>(1, 1, 1)
    var intensity: Float = 1.0

    var worldTransform: float4x4 = matrix_identity_float4x4

    var position: SIMD3<Float> {
        return worldTransform.columns.3.xyz
    }

    var direction: SIMD3<Float> {
        return -worldTransform.columns.2.xyz
    }
}
