
import Foundation
import simd

class FlyCameraController {
    let invertYLook = false
    let eyeSpeed: Float = 6
    let radiansPerLookPoint: Float = 0.017

    let pointOfView: Node

    private var eye = SIMD3<Float>(0, 0, 0)
    private var look = SIMD3<Float>(0, 0, -1)
    private var up = SIMD3<Float>(0, 1, 0)

    init(pointOfView: Node) {
        self.pointOfView = pointOfView
    }

    func update(timestep: Float, lookDelta: SIMD2<Float>, moveDelta: SIMD2<Float>)  {
        let right = normalize(cross(look, up))
        var forward = look

        let deltaX = moveDelta[0], deltaZ = moveDelta[1]
        let movementDir = SIMD3<Float>(deltaX * right.x + deltaZ * forward.x,
                                       deltaX * right.y + deltaZ * forward.y,
                                       deltaX * right.z + deltaZ * forward.z)
        eye += movementDir * eyeSpeed * timestep

        let yaw = -lookDelta.x * radiansPerLookPoint
        let yawRotation = simd_quaternion(yaw, up)

        var pitch = lookDelta.y * radiansPerLookPoint
        if (invertYLook) { pitch *= -1.0 }
        let pitchRotation = simd_quaternion(pitch, right)

        let rotation = simd_mul(pitchRotation, yawRotation)
        forward = rotation.rotate(forward)

        look = normalize(forward)
        up = cross(right, look)

        pointOfView.transform = simd_float4x4(lookAt: eye + look,
                                              from: eye,
                                              up: up)
    }
}
