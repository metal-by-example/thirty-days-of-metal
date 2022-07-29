
import Foundation
import ModelIO
import Metal

/// A type that holds the data representing a skinning hierarchy stored in an asset
class Skeleton {
    let name: String
    let jointPaths: [String]
    let inverseBindTransforms: [float4x4]
    let restTransforms: [float4x4]

    var jointCount: Int {
        return joints.count
    }

    private var joints = [Node]()

    init(_ mdlSkeleton: MDLSkeleton) {
        name = mdlSkeleton.name
        jointPaths = mdlSkeleton.jointPaths
        inverseBindTransforms = mdlSkeleton.jointBindTransforms.float4x4Array.map { $0.inverse }
        restTransforms = mdlSkeleton.jointRestTransforms.float4x4Array
        joints = makeSkeletonHierarchy(from: jointPaths)

        // Initialize the local transform of each joint to its resting transform
        for (jointIndex, joint) in zip(0..., joints) {
            joint.transform = restTransforms[jointIndex]
        }
    }

    func apply(animation: JointAnimation, at time: TimeInterval) {
        // Get the animated local transforms of the joints affected by the animation
        let animatedTransforms = animation.jointTransforms(at: time)
        // Since animations may not affect every joint, the loop below has two indices:
        // one holding the index of the joint in this skeleton, and one holding the
        // index of the joint in the animation's joints list. If a joint in this skeleton
        // does not appear in the animation, we reset it to its rest transformation.
        for (skeletonJointIndex, jointPath) in zip(0..., jointPaths) {
            if let animationJointIndex = animation.jointPaths.firstIndex(of: jointPath) {
                joints[skeletonJointIndex].transform = animatedTransforms[animationJointIndex]
            } else {
                joints[skeletonJointIndex].transform = restTransforms[skeletonJointIndex]
            }
        }
    }

    func copyTransforms(into buffer: MTLBuffer, at offset: Int) {
        // Regardless of whether a joint is animated, its total transformation is the
        // product of its world transformation (which bakes together the transformations
        // of its ancestors in the skeletal hierarchy) and its inverse bind transformation,
        // which transforms the joint from its native model space into local bind space.
        // This is where we compute the total transformations of all of the joints.
        var jointTransforms = zip(0..., joints).map { index, joint in
            return joint.worldTransform * inverseBindTransforms[index]
        }
        let transformPtr = buffer.contents().advanced(by: offset).assumingMemoryBound(to: float4x4.self)
        transformPtr.assign(from: &jointTransforms, count: jointTransforms.count)
    }

    private func makeSkeletonHierarchy(from jointPaths: [String]) -> [Node] {
        var joints = [Node]()
        var jointsForPaths = [String : Node]()
        for jointPath in jointPaths {
            let joint = Node()
            joint.name = jointPath
            jointsForPaths[jointPath] = joint
            joints.append(joint)
        }

        for jointPath in jointPaths {
            let child = jointsForPaths[jointPath]!
            let parentPath = (jointPath as NSString).deletingLastPathComponent as String
            let parent = jointsForPaths[parentPath]
            child.name = (jointPath as NSString).lastPathComponent as String
            parent?.addChildNode(child)
        }

        return joints
    }
}

/// A type that represents the transformations of a skinning hierarchy animated over time
class JointAnimation {
    let name: String
    let jointPaths: [String]
    let startTime: TimeInterval
    let duration: TimeInterval
    let translations: MDLAnimatedVector3Array
    let rotations: MDLAnimatedQuaternionArray
    let scales: MDLAnimatedVector3Array

    init(_ animation: MDLPackedJointAnimation) {
        name = animation.name
        jointPaths = animation.jointPaths
        translations = animation.translations
        rotations = animation.rotations
        scales = animation.scales

        startTime = animation.minimumTime
        duration = animation.maximumTime - startTime
    }

    func jointTransforms(at time: TimeInterval) -> [float4x4] {
        let translationsAtTime = translations.float3Array(atTime: time)
        let rotationsAtTime = rotations.floatQuaternionArray(atTime: time)
        let scalesAtTime = scales.float3Array(atTime: time)
        return zip(translationsAtTime, zip(rotationsAtTime, scalesAtTime)).map {
            let (translation, (orientation, scale)) = $0
            return float4x4(translation: translation, orientation: orientation, scale: scale)
        }
    }
}

/// A type that associates a skeleton with a base geometry bind pose
struct Skinner {
    let skeleton: Skeleton
    let geometryBindTransform: float4x4

    init(_ skeleton: Skeleton,
         _ geometryBindTransform: float4x4 = matrix_identity_float4x4)
    {
        self.skeleton = skeleton
        self.geometryBindTransform = geometryBindTransform
    }
}
