
import MetalKit
import ModelIO

extension MDLVertexDescriptor {
    var vertexAttributes: [MDLVertexAttribute] {
        return attributes as! [MDLVertexAttribute]
    }

    var bufferLayouts: [MDLVertexBufferLayout] {
        return layouts as! [MDLVertexBufferLayout]
    }
}

extension MDLMesh {
    var submeshArray: [MDLSubmesh] {
        return submeshes as! [MDLSubmesh]
    }
}

extension MDLAsset {
    var meshes: [MDLMesh] {
        return childObjects(of: MDLMesh.self) as? [MDLMesh] ?? []
    }
}
