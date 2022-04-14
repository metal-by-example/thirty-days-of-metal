
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
