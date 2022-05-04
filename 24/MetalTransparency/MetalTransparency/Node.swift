
import Foundation
import MetalKit
import simd

class Node {
    var mesh: MTKMesh?

    var texture: MTLTexture?

    var transform: simd_float4x4 = matrix_identity_float4x4

    var worldTransform: simd_float4x4 {
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

    init() {
    }

    init(mesh: MTKMesh) {
        self.mesh = mesh
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
}
