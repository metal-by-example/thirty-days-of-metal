
import Foundation
import Metal
import MetalKit
import simd

let MaxOutstandingFrameCount = 3
let MaxConstantsSize = 1_024 * 1_024
let MinBufferAlignment = 256

struct InstanceConstants {
    var modelMatrix: float4x4
}

struct FrameConstants {
    var projectionMatrix: float4x4
    var viewMatrix: float4x4
    var lightCount: UInt32
}

struct LightConstants {
    var viewProjectionMatrix: simd_float4x4
    var intensity: simd_float3
    var position: simd_float3
    var direction: simd_float3
    var type: UInt32
}

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let view: MTKView

    var pointOfView = Node()
    var skyboxNode: Node!
    var nodes = [Node]()
    var lights = [Light]()

    var earthNode, cloudNode, atmosphereNode: Node!

    var nodeConstantOffsets = [Int]()

    private let mdlVertexDescriptor = MDLVertexDescriptor()
    private var vertexDescriptor: MTLVertexDescriptor!

    private var renderPipelineState: MTLRenderPipelineState!
    private var depthStencilState: MTLDepthStencilState!
    private var samplerState: MTLSamplerState!

    private var bufferAllocator: MTKMeshBufferAllocator!
    private var textureLoader: MTKTextureLoader!
    private let textureOptions: [MTKTextureLoader.Option : Any] = [
        .textureUsage : MTLTextureUsage.shaderRead.rawValue,
        .textureStorageMode : MTLStorageMode.private.rawValue,
        .origin : MTKTextureLoader.Origin.bottomLeft.rawValue
    ]

    private var constantBuffer: MTLBuffer!
    private var currentConstantBufferOffset = 0
    private var frameConstantsOffset = 0
    private var lightConstantsOffset = 0

    private var frameSemaphore = DispatchSemaphore(value: MaxOutstandingFrameCount)
    private var frameIndex = 0
    private var time: TimeInterval = 0

    init(device: MTLDevice, view: MTKView) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.view = view

        super.init()

        view.device = device
        view.delegate = self
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        view.sampleCount = 1

        bufferAllocator = MTKMeshBufferAllocator(device: device)
        textureLoader = MTKTextureLoader(device: device)

        makeScene()
        makeResources()
        makePipelines()
    }

    func makeScene() {
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

        vertexDescriptor = MTKMetalVertexDescriptorFromModelIO(mdlVertexDescriptor)!

        let skyboxTextureURL = Bundle.main.url(forResource: "starscape", withExtension: "png")!
        let skyboxTexture = try? textureLoader.newTexture(URL: skyboxTextureURL, options: textureOptions)
        let mdlBox = MDLMesh(boxWithExtent: SIMD3<Float>(100, 100, 100),
                             segments: SIMD3<UInt32>(1, 1, 1),
                             inwardNormals: true,
                             geometryType: .triangles,
                             allocator: bufferAllocator)
        let mtkBox = try! MTKMesh(mesh: mdlBox, device: device)
        skyboxNode = Node(mesh: mtkBox)
        skyboxNode.texture = skyboxTexture
        nodes.append(skyboxNode)

        let earthURL = Bundle.main.url(forResource: "earth", withExtension: "obj")!
        earthNode = loadAsset(earthURL)

        let cloudURL = Bundle.main.url(forResource: "clouds", withExtension: "obj")!
        cloudNode = loadAsset(cloudURL)

        let mdlPlane = MDLMesh(planeWithExtent: SIMD3<Float>(1.05, 1.05, 0.0),
                               segments: SIMD2<UInt32>(1, 1),
                               geometryType: .triangles,
                               allocator: bufferAllocator)
        mdlPlane.vertexDescriptor = mdlVertexDescriptor
        let mtkPlane = try! MTKMesh(mesh: mdlPlane, device: device)

        let atmosphereTextureURL = Bundle.main.url(forResource: "atmosphere", withExtension: "png")!
        let atmosphereTexture = try? textureLoader.newTexture(URL: atmosphereTextureURL, options: textureOptions)
        atmosphereNode = Node(mesh: mtkPlane)
        atmosphereNode.texture = atmosphereTexture

        nodes.append(contentsOf: [earthNode, cloudNode, atmosphereNode])

        let ambientLight = Light()
        ambientLight.type = .ambient
        ambientLight.intensity = 0.7

        let sunLight = Light()
        sunLight.type = .directional
        sunLight.intensity = 0.3
        sunLight.worldTransform = simd_float4x4(lookAt: SIMD3<Float>(0, 0, 0),
                                                from: SIMD3<Float>(1, 1, 1),
                                                up: SIMD3<Float>(0, 1, 0))

        lights = [ambientLight, sunLight]
    }

    func loadAsset(_ url: URL) -> Node {
        let node = Node()

        let mdlAsset = MDLAsset(url: url, vertexDescriptor: mdlVertexDescriptor, bufferAllocator: bufferAllocator)
        mdlAsset.loadTextures()

        let meshes = mdlAsset.childObjects(of: MDLMesh.self) as? [MDLMesh]
        if let mdlMesh = meshes?.first {
            let mesh = try! MTKMesh(mesh: mdlMesh, device: device)
            node.mesh = mesh

            if let mdlSubmesh = mdlMesh.submeshArray.first {
                let mdlMaterial = mdlSubmesh.material
                if let baseColorProperty = mdlMaterial?.property(with: MDLMaterialSemantic.baseColor) {
                    if baseColorProperty.type == .texture, let textureURL = baseColorProperty.urlValue {
                        node.texture = try? textureLoader.newTexture(URL: textureURL,
                                                                     options: textureOptions)
                    }
                }
            }
        }

        return node
    }

    func makeResources() {
        constantBuffer = device.makeBuffer(length: MaxConstantsSize, options: .storageModeShared)
        constantBuffer.label = "Dynamic Constant Buffer"
    }

    func makePipelines() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Unable to create default Metal library")
        }

        let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
        renderPipelineDescriptor.rasterSampleCount = view.sampleCount

        renderPipelineDescriptor.vertexDescriptor = vertexDescriptor

        renderPipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        renderPipelineDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat

        renderPipelineDescriptor.vertexFunction = library.makeFunction(name: "vertex_main")!
        renderPipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragment_main")!

        renderPipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        renderPipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .one
        renderPipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        renderPipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        renderPipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        renderPipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        renderPipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add

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
        samplerDescriptor.mipFilter = .linear
        samplerDescriptor.sAddressMode = .repeat
        samplerDescriptor.tAddressMode = .repeat
        samplerState = device.makeSamplerState(descriptor: samplerDescriptor)!
    }

    func allocateConstantStorage(size: Int, alignment: Int) -> Int {
        let effectiveAlignment = lcm(alignment, MinBufferAlignment)
        var allocationOffset = align(currentConstantBufferOffset, upTo: effectiveAlignment)
        if (allocationOffset + size >= MaxConstantsSize) {
            allocationOffset = 0
        }
        currentConstantBufferOffset = allocationOffset + size
        return allocationOffset
    }

    func updateFrameConstants() {
        let aspectRatio = Float(view.drawableSize.width / view.drawableSize.height)
        let projectionMatrix = simd_float4x4(perspectiveProjectionFoVY: .pi / 3,
                                             aspectRatio: aspectRatio,
                                             near: 0.01,
                                             far: 100)

        let viewMatrix = pointOfView.transform.inverse

        var constants = FrameConstants(projectionMatrix: projectionMatrix,
                                       viewMatrix: viewMatrix,
                                       lightCount: UInt32(lights.count))

        let layout = MemoryLayout<FrameConstants>.self
        frameConstantsOffset = allocateConstantStorage(size: layout.size, alignment: layout.stride)
        let constantsPointer = constantBuffer.contents().advanced(by: frameConstantsOffset)
        constantsPointer.copyMemory(from: &constants, byteCount: layout.size)
    }

    func updateLightConstants() {
        let layout = MemoryLayout<LightConstants>.self
        lightConstantsOffset = allocateConstantStorage(size: layout.stride * lights.count, alignment: layout.stride)
        let lightsBufferPointer = constantBuffer.contents().advanced(by: lightConstantsOffset).assumingMemoryBound(to: LightConstants.self)

        for (lightIndex, light) in lights.enumerated() {
            let shadowViewMatrix = light.worldTransform.inverse
            let shadowProjectionMatrix = light.projectionMatrix
            let shadowViewProjectionMatrix = shadowProjectionMatrix * shadowViewMatrix

            lightsBufferPointer[lightIndex] = LightConstants(viewProjectionMatrix: shadowViewProjectionMatrix,
                                                             intensity: light.color * light.intensity,
                                                             position: light.position,
                                                             direction: light.direction,
                                                             type: light.type.rawValue)
        }
    }

    func updateNodeConstants(_ timestep: Float) {
        let t = Float(time)

        skyboxNode.transform = simd_float4x4(translate: pointOfView.position)

        earthNode.transform = simd_float4x4(rotateAbout: SIMD3<Float>(0, 1, 0), byAngle: t * -0.5)
        cloudNode.transform = simd_float4x4(rotateAbout: SIMD3<Float>(0, 1, 0), byAngle: t * -0.8)

        atmosphereNode.transform = simd_float4x4(lookAt: pointOfView.position, from: SIMD3<Float>(0, 0, 0), up: SIMD3<Float>(0, 1, 0))

        nodeConstantOffsets.removeAll()
        for node in nodes {
            let layout = MemoryLayout<InstanceConstants>.self
            let offset = allocateConstantStorage(size: layout.stride, alignment: layout.stride)

            let instanceConstants = constantBuffer.contents().advanced(by: offset)
            var instance = InstanceConstants(modelMatrix: node.worldTransform)
            instanceConstants.copyMemory(from: &instance, byteCount: layout.size)

            nodeConstantOffsets.append(offset)
        }
    }

    func drawMainPass(renderPassDescriptor: MTLRenderPassDescriptor, commandBuffer: MTLCommandBuffer) {
        let renderCommandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        renderCommandEncoder.setRenderPipelineState(renderPipelineState)

        renderCommandEncoder.setDepthStencilState(depthStencilState)
        renderCommandEncoder.setFrontFacing(.counterClockwise)
        renderCommandEncoder.setCullMode(.back)

        renderCommandEncoder.setVertexBuffer(constantBuffer, offset: frameConstantsOffset, index: 3)
        renderCommandEncoder.setFragmentBuffer(constantBuffer, offset: frameConstantsOffset, index: 3)
        renderCommandEncoder.setFragmentBuffer(constantBuffer, offset: lightConstantsOffset, index: 4)

        for (nodeIndex, node) in nodes.enumerated() {
            guard let mesh = node.mesh else { continue }

            renderCommandEncoder.setVertexBuffer(constantBuffer, offset: nodeConstantOffsets[nodeIndex], index: 2)

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
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
    }

    func draw(in view: MTKView) {
        frameSemaphore.wait()

        let timestep = 1.0 / Double(view.preferredFramesPerSecond)
        time += timestep

        updateLightConstants()
        updateFrameConstants()
        updateNodeConstants(Float(timestep))

        guard let renderPassDescriptor = view.currentRenderPassDescriptor else { return }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        drawMainPass(renderPassDescriptor: renderPassDescriptor, commandBuffer: commandBuffer)

        commandBuffer.present(view.currentDrawable!)

        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.frameSemaphore.signal()
        }

        commandBuffer.commit()

        frameIndex += 1
    }
}
