
import Foundation
import simd

class FlyCameraController {
    let invertYLook = false
    let eyeSpeed: Float = 6
    let radiansPerLookPoint: Float = 0.017
    let maximumPitchRadians = (Float.pi / 2) * 0.99

    let pointOfView: Node

    var eye = SIMD3<Float>(0, 0, 0)
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

        let angleToUp: Float = acos(dot(look, up))
        let angleToDown: Float = acos(dot(look, -up))
        let maxPitch = max(0.0, angleToUp - (.pi / 2 - maximumPitchRadians))
        let minPitch = max(0.0, angleToDown - (.pi / 2 - maximumPitchRadians))
        var pitch = lookDelta.y * radiansPerLookPoint
        if (invertYLook) { pitch *= -1.0 }
        pitch = max(-minPitch, min(pitch, maxPitch))
        let pitchRotation = simd_quaternion(pitch, right)

        let rotation = pitchRotation * yawRotation
        forward = rotation.rotate(forward)

        look = normalize(forward)

        pointOfView.transform = float4x4(lookAt: eye + look,
                                         from: eye,
                                         up: up)
    }
}
