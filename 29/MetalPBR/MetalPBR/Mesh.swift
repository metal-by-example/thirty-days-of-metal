
import Foundation
import Metal
import MetalKit
import ModelIO
import ImageIO

class MeshBuffer  {
    var buffer: MTLBuffer
    var length: Int
    var offset: Int

    init(buffer: MTLBuffer, offset: Int, length: Int) {
        self.buffer = buffer
        self.offset = offset
        self.length = length
    }

    init(_ mtkMeshBuffer: MTKMeshBuffer) {
        self.buffer = mtkMeshBuffer.buffer
        self.offset = mtkMeshBuffer.offset
        self.length = mtkMeshBuffer.length
    }
}

class Submesh {
    var primitiveType: MTLPrimitiveType
    var indexBuffer: MeshBuffer
    var indexType: MTLIndexType
    var indexCount: Int
    var material: Material?

    init(primitiveType: MTLPrimitiveType, indexBuffer: MeshBuffer, indexType: MTLIndexType, indexCount: Int) {
        self.primitiveType = primitiveType
        self.indexBuffer = indexBuffer
        self.indexType = indexType
        self.indexCount = indexCount
    }
}

class Mesh {
    enum Error : Swift.Error {
        case invalidAllocator
        case invalidDevice
        case invalidIndexType
        case invalidGeometryType
    }

    var vertexBuffers: [MeshBuffer]
    var vertexDescriptor: MDLVertexDescriptor
    var submeshes: [Submesh]
    var vertexCount: Int

    init(mesh: MDLMesh, device: MTLDevice) throws {
        let hasTexCoords = mesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeTextureCoordinate) != nil
        let hasNormals = mesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeNormal) != nil
        if (hasTexCoords && hasNormals) {
            mesh.addOrthTanBasis(forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                                 normalAttributeNamed: MDLVertexAttributeNormal,
                                 tangentAttributeNamed: MDLVertexAttributeTangent)
        }
        
        let hasSkinningData = mesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeJointWeights) != nil
        vertexDescriptor = hasSkinningData ? Mesh.skinnedVertexDescriptor : Mesh.defaultVertexDescriptor
        mesh.vertexDescriptor = vertexDescriptor

        vertexBuffers = []
        for mdlVertexBuffer in mesh.vertexBuffers {
            guard let mtkMeshBuffer = mdlVertexBuffer as? MTKMeshBuffer else {
                throw Error.invalidAllocator
            }

            if !mtkMeshBuffer.allocator.device.isEqual(device) {
                throw Error.invalidDevice
            }

            let vertexBuffer = MeshBuffer(mtkMeshBuffer)
            vertexBuffers.append(vertexBuffer)
        }

        vertexCount = mesh.vertexCount

        submeshes = []
        for mdlSubmesh in mesh.submeshArray {
            guard let mtkMeshBuffer = mdlSubmesh.indexBuffer as? MTKMeshBuffer else {
                throw Error.invalidAllocator
            }

            if !mtkMeshBuffer.allocator.device.isEqual(device) {
                throw Error.invalidDevice
            }

            guard let primitiveType = primitiveType(for: mdlSubmesh.geometryType) else {
                throw Error.invalidGeometryType
            }

            guard let indexType = indexType(for: mdlSubmesh.indexType) else {
                throw Error.invalidIndexType
            }

            let indexBuffer = MeshBuffer(mtkMeshBuffer)

            let submesh = Submesh(primitiveType: primitiveType,
                                  indexBuffer: indexBuffer,
                                  indexType: indexType,
                                  indexCount: mdlSubmesh.indexCount)

            if let mdlMaterial = mdlSubmesh.material {
                submesh.material = Material(mdlMaterial, device: device)
            }

            submeshes.append(submesh)
        }
    }
}

extension Mesh {
    static var defaultVertexDescriptor: MDLVertexDescriptor {
        let vertexDescriptor = MDLVertexDescriptor()
        vertexDescriptor.vertexAttributes[0].name = MDLVertexAttributePosition
        vertexDescriptor.vertexAttributes[0].format = .float3
        vertexDescriptor.vertexAttributes[0].offset = 0
        vertexDescriptor.vertexAttributes[0].bufferIndex = 0
        vertexDescriptor.vertexAttributes[1].name = MDLVertexAttributeNormal
        vertexDescriptor.vertexAttributes[1].format = .float3
        vertexDescriptor.vertexAttributes[1].offset = 12
        vertexDescriptor.vertexAttributes[1].bufferIndex = 0
        vertexDescriptor.vertexAttributes[2].name = MDLVertexAttributeTangent
        vertexDescriptor.vertexAttributes[2].format = .float4
        vertexDescriptor.vertexAttributes[2].offset = 24
        vertexDescriptor.vertexAttributes[2].bufferIndex = 0
        vertexDescriptor.vertexAttributes[3].name = MDLVertexAttributeTextureCoordinate
        vertexDescriptor.vertexAttributes[3].format = .float2
        vertexDescriptor.vertexAttributes[3].offset = 40
        vertexDescriptor.vertexAttributes[3].bufferIndex = 0
        vertexDescriptor.bufferLayouts[0].stride = 48
        return vertexDescriptor
    }

    static var skinnedVertexDescriptor: MDLVertexDescriptor {
        let vertexDescriptor = MDLVertexDescriptor()
        vertexDescriptor.vertexAttributes[0].name = MDLVertexAttributePosition
        vertexDescriptor.vertexAttributes[0].format = .float3
        vertexDescriptor.vertexAttributes[0].offset = 0
        vertexDescriptor.vertexAttributes[0].bufferIndex = 0
        vertexDescriptor.vertexAttributes[1].name = MDLVertexAttributeNormal
        vertexDescriptor.vertexAttributes[1].format = .float3
        vertexDescriptor.vertexAttributes[1].offset = 12
        vertexDescriptor.vertexAttributes[1].bufferIndex = 0
        vertexDescriptor.vertexAttributes[2].name = MDLVertexAttributeTangent
        vertexDescriptor.vertexAttributes[2].format = .float4
        vertexDescriptor.vertexAttributes[2].offset = 24
        vertexDescriptor.vertexAttributes[2].bufferIndex = 0
        vertexDescriptor.vertexAttributes[3].name = MDLVertexAttributeTextureCoordinate
        vertexDescriptor.vertexAttributes[3].format = .float2
        vertexDescriptor.vertexAttributes[3].offset = 40
        vertexDescriptor.vertexAttributes[3].bufferIndex = 0
        vertexDescriptor.vertexAttributes[4].name = MDLVertexAttributeJointIndices
        vertexDescriptor.vertexAttributes[4].format = .uShort4
        vertexDescriptor.vertexAttributes[4].offset = 48
        vertexDescriptor.vertexAttributes[4].bufferIndex = 0
        vertexDescriptor.vertexAttributes[5].name = MDLVertexAttributeJointWeights
        vertexDescriptor.vertexAttributes[5].format = .float4
        vertexDescriptor.vertexAttributes[5].offset = 56
        vertexDescriptor.vertexAttributes[5].bufferIndex = 0
        vertexDescriptor.bufferLayouts[0].stride = 72
        return vertexDescriptor
    }
}

fileprivate func indexType(for indexBitDepth: MDLIndexBitDepth) -> MTLIndexType? {
    switch indexBitDepth {
    case .uInt16 :
        return .uint16
    case .uInt32 :
        return .uint32
    default:
        return nil
    }
}

fileprivate func primitiveType(for geometryType: MDLGeometryType) -> MTLPrimitiveType? {
    switch geometryType{
    case .points:
        return .point
    case .lines:
        return .line
    case .triangles:
        return .triangle
    case .triangleStrips:
        return .triangleStrip
    default:
        return nil
    }
}
