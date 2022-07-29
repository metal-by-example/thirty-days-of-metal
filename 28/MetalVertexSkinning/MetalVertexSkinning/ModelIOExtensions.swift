
import ModelIO

extension MDLVertexDescriptor {
    var vertexAttributes: [MDLVertexAttribute] {
        return attributes as! [MDLVertexAttribute]
    }

    var bufferLayouts: [MDLVertexBufferLayout] {
        return layouts as! [MDLVertexBufferLayout]
    }
}

extension MDLObject {
    var animationBind: MDLAnimationBindComponent? {
        // There is no protocol corresponding to this component type, so we have to scan for it
        return components.filter({ $0 is MDLAnimationBindComponent }).first as? MDLAnimationBindComponent
    }
}

extension MDLMesh {
    var submeshArray: [MDLSubmesh] {
        return submeshes as! [MDLSubmesh]
    }
}

extension MDLPackedJointAnimation {
    var minimumTime: TimeInterval {
        return [translations, rotations, scales]
            .reduce(TimeInterval.greatestFiniteMagnitude) { return min($0, $1.minimumTime) }
    }

    var maximumTime: TimeInterval {
        return [translations, rotations, scales]
            .reduce(-TimeInterval.greatestFiniteMagnitude) { return max($0, $1.maximumTime) }
    }
}
