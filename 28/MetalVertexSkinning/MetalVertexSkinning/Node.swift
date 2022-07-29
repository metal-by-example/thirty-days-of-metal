
import Foundation
import MetalKit
import simd

class Node {
    var name = ""

    var mesh: Mesh?

    var transform: float4x4 = matrix_identity_float4x4

    var worldTransform: float4x4 {
        if let parent = parentNode {
            return parent.worldTransform * transform
        } else {
            return transform
        }
    }

    var position: SIMD3<Float> {
        return worldTransform.columns.3.xyz
    }

    weak var parentNode: Node?
    
    private(set) var childNodes = [Node]()

    var skinner: Skinner?

    private(set) var animation: JointAnimation?

    init() {
    }

    init(mesh: Mesh) {
        self.mesh = mesh
    }

    /// Search for a child node with the given name, optionally searching
    /// this node's subhierarchy recursively.
    func childNode(named name: String, recursive: Bool = true) -> Node? {
        if let child = childNodes.first(where: { $0.name == name } ) {
            return child
        } else if recursive {
            for child in childNodes {
                if let grandchild = child.childNode(named: name) {
                    return grandchild
                }
            }
        }
        return nil
    }

    func addChildNode(_ node: Node) {
        childNodes.append(node)
        node.parentNode = self
    }

    func removeFromParent() {
        parentNode?.removeChildNode(self)
    }

    private func removeChildNode(_ node: Node) {
        childNodes.removeAll { $0 === node }
    }

    func runAnimation(_ animation: JointAnimation) {
        self.animation = animation
    }

    func update(at time: TimeInterval) {
        if let animation = animation, let skinner = skinner {
            let localTime = max(0, time - animation.startTime)
            let loopTime = fmod(localTime, animation.duration)
            skinner.skeleton.apply(animation: animation, at: loopTime)
        }
    }
}
