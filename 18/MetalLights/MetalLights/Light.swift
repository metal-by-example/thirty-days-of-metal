
import Foundation

enum LightType : UInt32 {
    case ambient
    case directional
}

class Light {
    var type = LightType.directional
    var color = SIMD3<Float>(1, 1, 1)
    var intensity: Float = 1.0
    var direction = SIMD3<Float>(0, 0, -1)
}
