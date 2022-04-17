
import Cocoa
import Metal
import MetalKit
import simd

let MaxOutstandingFrameCount = 3
let MaxObjectCount = 16

struct NodeConstants {
    var modelViewProjectionMatrix: float4x4
}

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let view: MTKView

    var vertexDescriptor: MTLVertexDescriptor!
    var nodes = [Node]()
    var boxNode: Node!
    var sphereNode: Node!

    private var renderPipelineState: MTLRenderPipelineState!
    private var depthStencilState: MTLDepthStencilState!
    private var samplerState: MTLSamplerState!

    private var frameSemaphore = DispatchSemaphore(value: MaxOutstandingFrameCount)
    private var frameIndex: Int
    private var time: TimeInterval = 0

    private var constantBuffer: MTLBuffer!
    private let constantsSize: Int
    private let constantsStride: Int
    private var currentConstantBufferOffset: Int

    init(device: MTLDevice, view: MTKView) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.view = view
        self.frameIndex = 0
        self.constantsSize = MemoryLayout<NodeConstants>.size
        self.constantsStride = align(constantsSize, upTo: 256)
        self.currentConstantBufferOffset = 0

        super.init()

        view.device = device
        view.delegate = self
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColor(red: 0.95, green: 0.95, blue: 0.95, alpha: 1.0)

        makeResources()
        makePipeline()
    }

    func makeResources() {
        var texture: MTLTexture?
        let textureLoader = MTKTextureLoader(device: device)
        let options: [MTKTextureLoader.Option : Any] = [
            .textureUsage : MTLTextureUsage.shaderRead.rawValue,
            .textureStorageMode : MTLStorageMode.private.rawValue
        ]
        texture = try? textureLoader.newTexture(name: "uv_grid",
                                                scaleFactor: 1.0,
                                                bundle: nil,
                                                options: options)

        let allocator = MTKMeshBufferAllocator(device: device)

        let mdlVertexDescriptor = MDLVertexDescriptor()
        mdlVertexDescriptor.vertexAttributes[0].name = MDLVertexAttributePosition
        mdlVertexDescriptor.vertexAttributes[0].format = .float3
        mdlVertexDescriptor.vertexAttributes[0].offset = 0
        mdlVertexDescriptor.vertexAttributes[0].bufferIndex = 0
        mdlVertexDescriptor.vertexAttributes[1].name = MDLVertexAttributeNormal
        mdlVertexDescriptor.vertexAttributes[1].format = .float3
        mdlVertexDescriptor.vertexAttributes[1].offset = 12
        mdlVertexDescriptor.vertexAttributes[1].bufferIndex = 0
        mdlVertexDescriptor.vertexAttributes[2].name = MDLVertexAttributeTextureCoordinate
        mdlVertexDescriptor.vertexAttributes[2].format = .float2
        mdlVertexDescriptor.vertexAttributes[2].offset = 24
        mdlVertexDescriptor.vertexAttributes[2].bufferIndex = 0
        mdlVertexDescriptor.bufferLayouts[0].stride = 32

        let mdlBox = MDLMesh(boxWithExtent: SIMD3<Float>(1.4, 1.4, 1.4),
                              segments: SIMD3<UInt32>(1, 1, 1),
                              inwardNormals: false,
                              geometryType: .triangles,
                              allocator: allocator)
        mdlBox.vertexDescriptor = mdlVertexDescriptor

        let boxMesh = try! MTKMesh(mesh: mdlBox, device: device)

        let mdlSphere = MDLMesh(sphereWithExtent: SIMD3<Float>(1, 1, 1),
                              segments: SIMD2<UInt32>(24, 24),
                              inwardNormals: true,
                              geometryType: .triangles,
                              allocator: allocator)
        mdlSphere.vertexDescriptor = mdlVertexDescriptor

        let sphereMesh = try! MTKMesh(mesh: mdlSphere, device: device)

        boxNode = Node(mesh: boxMesh)
        boxNode.texture = texture

        sphereNode = Node(mesh: sphereMesh)
        sphereNode.texture = texture

        nodes = [boxNode, sphereNode]

        vertexDescriptor = MTKMetalVertexDescriptorFromModelIO(mdlVertexDescriptor)!

        constantBuffer = device.makeBuffer(length: constantsStride * MaxObjectCount * MaxOutstandingFrameCount,
                                           options: .storageModeShared)
        constantBuffer.label = "Dynamic Constant Buffer"

    }

    func makePipeline() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Unable to create default Metal library")
        }

        let renderPipelineDescriptor = MTLRenderPipelineDescriptor()

        renderPipelineDescriptor.vertexDescriptor = vertexDescriptor

        renderPipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        renderPipelineDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat

        renderPipelineDescriptor.vertexFunction = library.makeFunction(name: "vertex_main")!
        renderPipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragment_main")!

        do {
            renderPipelineState = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
        } catch {
            fatalError("Error while creating render pipeline state: \(error)")
        }

        let depthStencilDescriptor = MTLDepthStencilDescriptor()
        depthStencilDescriptor.depthCompareFunction = .less
        depthStencilDescriptor.isDepthWriteEnabled = true
        depthStencilState = device.makeDepthStencilState(descriptor: depthStencilDescriptor)!

        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.normalizedCoordinates = true
        samplerDescriptor.magFilter = .linear
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.mipFilter = .nearest
        samplerDescriptor.sAddressMode = .repeat
        samplerDescriptor.tAddressMode = .repeat
        samplerState = device.makeSamplerState(descriptor: samplerDescriptor)!
    }

    func updateConstants() {
        time += (1.0 / Double(view.preferredFramesPerSecond))
        let t = Float(time)

        let cameraPosition = SIMD3<Float>(0, 0, 5)
        let viewMatrix = simd_float4x4(translate: -cameraPosition)

        let aspectRatio = Float(view.drawableSize.width / view.drawableSize.height)
        let projectionMatrix = simd_float4x4(perspectiveProjectionFoVY: .pi / 3,
                                             aspectRatio: aspectRatio,
                                             near: 0.01,
                                             far: 100)

        let rotationAxis = normalize(SIMD3<Float>(0.3, 0.7, 0.1))
        let rotation = simd_float4x4(rotateAbout: rotationAxis, byAngle: t)

        boxNode.transform = simd_float4x4(translate: SIMD3<Float>(-2, 0, 0)) * rotation
        sphereNode.transform = simd_float4x4(translate: SIMD3<Float>(2, 0, 0)) * rotation

        for (objectIndex, node) in nodes.enumerated() {
            let transform = projectionMatrix * viewMatrix * node.worldTransform
            var constants = NodeConstants(modelViewProjectionMatrix: transform)

            let offset = constantBufferOffset(objectIndex: objectIndex, frameIndex: frameIndex)
            let constantsPointer = constantBuffer.contents().advanced(by: offset)
            constantsPointer.copyMemory(from: &constants, byteCount: constantsSize)
        }
    }

    func constantBufferOffset(objectIndex: Int, frameIndex: Int) -> Int {
        let frameConstantOffset = (frameIndex % MaxOutstandingFrameCount) * MaxObjectCount * constantsStride
        let objectConstantOffset = frameConstantOffset + (objectIndex * constantsStride)
        return objectConstantOffset
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
    }

    func draw(in view: MTKView) {
        frameSemaphore.wait()
        updateConstants()

        guard let renderPassDescriptor = view.currentRenderPassDescriptor else { return }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        let renderCommandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        renderCommandEncoder.setRenderPipelineState(renderPipelineState)

        renderCommandEncoder.setDepthStencilState(depthStencilState)
        renderCommandEncoder.setFrontFacing(.counterClockwise)
        renderCommandEncoder.setCullMode(.back)

        for (objectIndex, node) in nodes.enumerated() {
            guard let mesh = node.mesh else { continue }

            let offset = constantBufferOffset(objectIndex: objectIndex, frameIndex: frameIndex)
            renderCommandEncoder.setVertexBuffer(constantBuffer,
                                                 offset: offset,
                                                 index: 2)

            for (i, meshBuffer) in mesh.vertexBuffers.enumerated() {
                renderCommandEncoder.setVertexBuffer(meshBuffer.buffer,
                                                     offset: meshBuffer.offset,
                                                     index: i)
            }

            renderCommandEncoder.setFragmentTexture(node.texture, index: 0)
            renderCommandEncoder.setFragmentSamplerState(samplerState, index: 0)

            for submesh in mesh.submeshes {
                let indexBuffer = submesh.indexBuffer
                renderCommandEncoder.drawIndexedPrimitives(type: submesh.primitiveType,
                                                           indexCount: submesh.indexCount,
                                                           indexType: submesh.indexType,
                                                           indexBuffer: indexBuffer.buffer,
                                                           indexBufferOffset: indexBuffer.offset)
            }
        }

        renderCommandEncoder.endEncoding()

        commandBuffer.present(view.currentDrawable!)

        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.frameSemaphore.signal()
        }

        commandBuffer.commit()

        frameIndex += 1
    }
}
