
import Foundation
import Metal
import MetalKit
import ModelIO

class Scene {
    let rootNode = Node()
    var animations = [(JointAnimation, Node)]()

    var nodes: [Node] {
        var nodeQueue = [rootNode]
        var flattenedNodes = [Node]()
        while !nodeQueue.isEmpty {
            let node = nodeQueue.removeFirst()
            flattenedNodes.append(node)
            nodeQueue.append(contentsOf: node.childNodes)
        }
        return flattenedNodes
    }

    init() {
    }

    init(url: URL, device: MTLDevice) {
        let bufferAllocator = MTKMeshBufferAllocator(device: device)

        let asset = MDLAsset(url: url, vertexDescriptor: nil, bufferAllocator: bufferAllocator)
        asset.loadTextures()

        var skeletons = [String : Skeleton]()
        let skeletonForMDLSkeleton: (MDLSkeleton) -> Skeleton = { mdlSkeleton in
            var cachedSkeleton: Skeleton! = skeletons[mdlSkeleton.name]
            if cachedSkeleton == nil {
                cachedSkeleton = Skeleton(mdlSkeleton)
                skeletons[mdlSkeleton.name] = cachedSkeleton
            }
            return cachedSkeleton
        }

        let topLevelCount = asset.count
        let topLevelObjects = (0..<topLevelCount).map { asset.object(at: $0) }
        var objectQueue = [MDLObject](topLevelObjects)
        var parentQueue = [Node?](repeating: nil, count: topLevelCount)
        while !objectQueue.isEmpty {
            let mdlObject = objectQueue.removeFirst()
            let parentNode = parentQueue.removeFirst() ?? rootNode
            let node = Node()
            node.name = mdlObject.name

            if let mdlMesh = mdlObject as? MDLMesh {
                node.mesh = try? Mesh(mesh: mdlMesh, device: device)
            }

            if let mdlSkeleton = mdlObject as? MDLSkeleton {
                node.skinner = Skinner(skeletonForMDLSkeleton(mdlSkeleton))
            }

            if let transformStack = mdlObject.transform {
                if (transformStack.keyTimes.count > 1) {
                    print("Warning: Animated transform stacks are not currently supported")
                }

                node.transform = transformStack.matrix
            }

            if let animationBinding = mdlObject.animationBind {
                if animationBinding.jointPaths != nil {
                    print("Warning: Animation bindings with explicit joint paths are not currently supported")
                }

                if let mdlAnimation = animationBinding.jointAnimation as? MDLPackedJointAnimation {
                    let animation = JointAnimation(mdlAnimation)
                    animations.append((animation, node))
                }

                if let mdlSkeleton = animationBinding.skeleton {
                    node.skinner = Skinner(skeletonForMDLSkeleton(mdlSkeleton),
                                           float4x4(animationBinding.geometryBindTransform))
                }
            }

            parentNode.addChildNode(node)

            objectQueue.append(contentsOf: mdlObject.children.objects)
            parentQueue.append(contentsOf: [Node](repeating: node, count: mdlObject.children.count))
        }
    }
}
