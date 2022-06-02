
import Foundation
import Metal
import MetalKit
import ModelIO

class Material {
    var baseColor = SIMD4<Float>(1, 1, 1, 1)
    var baseColorTexture: MTLTexture?
    var normalTexture: MTLTexture?
    var displacementTexture: MTLTexture?
}

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

enum PatchType {
    case tri
    case quad
}

class PatchSubmesh {
    let patchType: PatchType
    let patchControlPointCount: Int
    let patchCount: Int
    let indexBuffer: MeshBuffer
    let indexType: MTLIndexType
    let indexCount: Int
    var material: Material?

    init(patchType: PatchType, indexBuffer: MeshBuffer, indexType: MTLIndexType, indexCount: Int) {
        self.patchType = patchType
        self.patchControlPointCount = (patchType == .tri) ? 3 : 4
        self.patchCount = indexCount / self.patchControlPointCount
        self.indexBuffer = indexBuffer
        self.indexType = indexType
        self.indexCount = indexCount
    }
}

class PatchMesh {
    enum Error : Swift.Error {
        case invalidAllocator
        case invalidDevice
        case invalidIndexType
        case invalidGeometryType
    }

    var vertexBuffers: [MeshBuffer]
    var vertexDescriptor: MDLVertexDescriptor
    var submeshes: [PatchSubmesh]
    var vertexCount: Int

    init(mesh: MDLMesh, device: MTLDevice) throws {
        let textureOptions: [MTKTextureLoader.Option : Any] = [
            .textureUsage : MTLTextureUsage.shaderRead.rawValue,
            .textureStorageMode : MTLStorageMode.private.rawValue,
            .generateMipmaps : true
        ]
        let normalTextureOptions: [MTKTextureLoader.Option : Any] = [
            .textureUsage : MTLTextureUsage.shaderRead.rawValue,
            .textureStorageMode : MTLStorageMode.private.rawValue,
            .generateMipmaps : true,
            .SRGB : false
        ]
        let textureLoader = MTKTextureLoader(device: device)

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

        vertexDescriptor = mesh.vertexDescriptor
        vertexCount = mesh.vertexCount

        submeshes = []
        for mdlSubmesh in mesh.submeshArray {
            // We only support meshes with uniform patch topology (triangles or quads) currently.
            precondition(mdlSubmesh.geometryType == .triangles || mdlSubmesh.geometryType == .quads)

            guard let mtkMeshBuffer = mdlSubmesh.indexBuffer as? MTKMeshBuffer else {
                throw Error.invalidAllocator
            }

            if !mtkMeshBuffer.allocator.device.isEqual(device) {
                throw Error.invalidDevice
            }

            guard let indexType = indexType(for: mdlSubmesh.indexType) else {
                throw Error.invalidIndexType
            }

            let indexBuffer = MeshBuffer(mtkMeshBuffer)

            let submesh = PatchSubmesh(patchType: (mdlSubmesh.geometryType == .triangles) ? .tri : .quad,
                                       indexBuffer: indexBuffer,
                                       indexType: indexType,
                                       indexCount: mdlSubmesh.indexCount)

            if let mdlMaterial = mdlSubmesh.material {
                let material = Material()
                if let baseColorProperty = mdlMaterial.property(with: MDLMaterialSemantic.baseColor) {
                    if baseColorProperty.type == .texture {
                        if let textureURL = baseColorProperty.urlValue {
                            material.baseColorTexture = try? textureLoader.newTexture(URL: textureURL,
                                                                                      options: textureOptions)
                        }
                    }
                }
                if let normalProperty = mdlMaterial.property(with: MDLMaterialSemantic.tangentSpaceNormal) {
                    if normalProperty.type == .texture {
                        if let textureURL = normalProperty.urlValue {
                            material.normalTexture = try? textureLoader.newTexture(URL: textureURL,
                                                                                   options: normalTextureOptions)
                        }
                    }
                }
                if let displacementProperty = mdlMaterial.property(with: MDLMaterialSemantic.displacement) {
                    if displacementProperty.type == .texture {
                        if let textureURL = displacementProperty.urlValue {
                            material.displacementTexture = try? textureLoader.newTexture(URL: textureURL,
                                                                                         options: textureOptions)
                        }
                    }
                }
                submesh.material = material
            }

            submeshes.append(submesh)
        }
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
