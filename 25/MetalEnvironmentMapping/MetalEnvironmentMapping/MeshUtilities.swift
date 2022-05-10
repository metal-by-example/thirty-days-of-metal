
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

extension MDLMesh {

    convenience init(torusWithRingRadius ringRadius: Float,
                     tubeRadius: Float,
                     tubeSegments: Int,
                     radialSegments: Int,
                     allocator: MDLMeshBufferAllocator?)
    {
        struct Vertex {
            var x, y, z: Float
            var nx, ny, nz: Float
            var u, v: Float
        }

        let ringVertexCount = radialSegments + 1
        let pipeVertexCount = tubeSegments + 1
        let vertexCount = ringVertexCount * pipeVertexCount
        let indexCount = 6 * radialSegments * tubeSegments

        let bufferAllocator = allocator ?? MDLMeshBufferDataAllocator()
        let vertexBufferLength = MemoryLayout<Vertex>.stride * vertexCount
        let vertexBuffer = bufferAllocator.newBuffer(vertexBufferLength, type: .vertex)
        let indexBufferLength = MemoryLayout<UInt32>.stride * indexCount
        let indexBuffer = bufferAllocator.newBuffer(indexBufferLength, type: .index)

        var vertices = [Vertex]()
        var v = 0
        var theta: Float = 0
        let dTheta = (Float.pi * 2) / Float(radialSegments)
        for _ in 0..<ringVertexCount {
            let cx = -sin(theta)
            let cy: Float = 0
            let cz = -cos(theta)
            let c = SIMD3<Float>(cx, cy, cz)

            let rX = SIMD3<Float>(-sin(theta), 0.0, -cos(theta))
            let rY = SIMD3<Float>(0.0, 1.0, 0.0)
            let rZ = SIMD3<Float>(cos(theta), 0.0, sin(theta))

            let ringFrame = float3x3(rX, rY, rZ)

            var phi: Float = 0
            let dPhi = (Float.pi * 2) / Float(tubeSegments)
            for _ in 0..<pipeVertexCount {
                let position = SIMD3<Float>(-cos(phi), sin(phi), 0.0)

                let p = (ringFrame * tubeRadius * position) + (ringRadius * c)
                let n = ringFrame * position
                let t = SIMD2<Float>(theta / (2 * .pi), phi / (2 * .pi))

                vertices.append(Vertex(x: p.x, y: p.y, z: p.z, nx: n.x, ny: n.y, nz: n.z, u: t.x, v: t.y))

                phi += dPhi
                v += 1
            }

            theta += dTheta
        }

        vertexBuffer.map().bytes.copyMemory(from: vertices, byteCount: vertexBufferLength)

        var i = 0
        var indices = [UInt32](repeating: 0, count: indexCount)
        for r in 0..<radialSegments {
            for p in 0..<tubeSegments {
                let bv = UInt32((r * pipeVertexCount) + p)
                indices[i] = bv; i += 1
                indices[i] = bv + 1; i += 1
                indices[i] = bv + UInt32(pipeVertexCount) + 1; i += 1
                indices[i] = bv + UInt32(pipeVertexCount) + 1; i += 1
                indices[i] = bv + UInt32(pipeVertexCount); i += 1
                indices[i] = bv; i += 1
            }
        }

        indexBuffer.map().bytes.copyMemory(from: indices, byteCount: indexBufferLength)

        let vertexDescriptor = MDLVertexDescriptor()

        let submesh = MDLSubmesh(indexBuffer: indexBuffer,
                                 indexCount: indexCount,
                                 indexType: .uint32,
                                 geometryType: .triangles,
                                 material: nil)

        self.init(vertexBuffer: vertexBuffer,
                  vertexCount: vertexCount,
                  descriptor: vertexDescriptor,
                  submeshes: [submesh])
    }
}
