
import Cocoa
import Metal
import MetalKit
import simd

let MaxOutstandingFrameCount = 3
let MaxConstantsSize = 1_024 * 1_024
let MinBufferAlignment = 256

struct NodeConstants {
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
    var direction: simd_float3
    var type: UInt32
}

struct ShadowConstants {
    var modelViewProjectionMatrix: simd_float4x4
}

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let view: MTKView

    var vertexDescriptor: MTLVertexDescriptor!

    let rasterSampleCount = 4
    var msaaColorTexture: MTLTexture?
    var msaaDepthTexture: MTLTexture?

    var nodes = [Node]()
    var lights = [Light]()

    var cowNode: Node!
    var floorNode: Node!
    var sunLight: Light!

    private var constantBuffer: MTLBuffer!
    private var currentConstantBufferOffset = 0
    private var frameConstantsOffset = 0
    private var lightConstantsOffset = 0
    private var nodeConstantsOffsets = [Int]()

    private var renderPipelineState: MTLRenderPipelineState!
    private var shadowRenderPipelineState: MTLRenderPipelineState!

    private var depthStencilState: MTLDepthStencilState!
    private var samplerState: MTLSamplerState!

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
        view.clearColor = MTLClearColor(red: 0.0, green: 0.5, blue: 0.95, alpha: 1.0)
        view.sampleCount = 1

        makeScene()
        makeResources()
        makePipelines()
    }

    func makeScene() {
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

        vertexDescriptor = MTKMetalVertexDescriptorFromModelIO(mdlVertexDescriptor)!

        let assetURL = Bundle.main.url(forResource: "spot_triangulated", withExtension: "obj")
        let mdlAsset = MDLAsset(url: assetURL, vertexDescriptor: mdlVertexDescriptor, bufferAllocator: allocator)

        mdlAsset.loadTextures()

        let meshes = mdlAsset.childObjects(of: MDLMesh.self) as? [MDLMesh]
        guard let mdlMesh = meshes?.first else {
            fatalError("Did not find any meshes in the Model I/O asset")
        }

        let textureLoader = MTKTextureLoader(device: device)
        let options: [MTKTextureLoader.Option : Any] = [
            .textureUsage : MTLTextureUsage.shaderRead.rawValue,
            .textureStorageMode : MTLStorageMode.private.rawValue,
            .origin : MTKTextureLoader.Origin.bottomLeft.rawValue
        ]

        var texture: MTLTexture?
        let firstSubmesh = mdlMesh.submeshes?.firstObject as? MDLSubmesh
        let material = firstSubmesh?.material
        if let baseColorProperty = material?.property(with: MDLMaterialSemantic.baseColor) {
            if baseColorProperty.type == .texture, let textureURL = baseColorProperty.urlValue {
                texture = try? textureLoader.newTexture(URL: textureURL, options: options)
            }
        }

        let mesh = try! MTKMesh(mesh: mdlMesh, device: device)

        cowNode = Node(mesh: mesh)
        cowNode.texture = texture

        let floorTextureURL = Bundle.main.url(forResource: "grass", withExtension: "png")!
        texture = try? textureLoader.newTexture(URL: floorTextureURL, options: options)

        let mdlPlane = MDLMesh(planeWithExtent: SIMD3<Float>(4, 0, 4),
                               segments: SIMD2<UInt32>(1, 1),
                               geometryType: .triangles,
                               allocator: allocator)
        let mtkPlane = try! MTKMesh(mesh: mdlPlane, device: device)

        floorNode = Node(mesh: mtkPlane)
        floorNode.texture = texture

        nodes = [floorNode, cowNode]

        let ambientLight = Light()
        ambientLight.type = .ambient
        ambientLight.intensity = 0.1

        sunLight = Light()
        sunLight.type = .directional
        sunLight.worldTransform = simd_float4x4(lookAt: SIMD3<Float>(0, 0, 0),
                                                from: SIMD3<Float>(1, 1, 1),
                                                up: SIMD3<Float>(0, 1, 0))
        sunLight.castsShadows = true

        lights = [ambientLight, sunLight]
    }

    func makeResources() {
        constantBuffer = device.makeBuffer(length: MaxConstantsSize,
                                           options: .storageModeShared)
        constantBuffer.label = "Dynamic Constant Buffer"

        let shadowMapSize = 2048
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float,
                                                                         width: shadowMapSize,
                                                                         height: shadowMapSize,
                                                                         mipmapped: false)
        textureDescriptor.storageMode = .private
        textureDescriptor.usage = [ .renderTarget, .shaderRead ]
        sunLight.shadowTexture = device.makeTexture(descriptor: textureDescriptor)!
    }

    func makePipelines() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Unable to create default Metal library")
        }

        let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
        renderPipelineDescriptor.rasterSampleCount = rasterSampleCount

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

        let shadowRenderPipelineDescriptor = MTLRenderPipelineDescriptor()
        shadowRenderPipelineDescriptor.vertexDescriptor = vertexDescriptor
        shadowRenderPipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        shadowRenderPipelineDescriptor.vertexFunction = library.makeFunction(name: "vertex_shadow")!

        do {
            shadowRenderPipelineState = try device.makeRenderPipelineState(descriptor: shadowRenderPipelineDescriptor)
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
        time += (1.0 / Double(view.preferredFramesPerSecond))

        let cameraPosition = SIMD3<Float>(0, 0.5, 2)
        let viewMatrix = simd_float4x4(translate: -cameraPosition)

        let aspectRatio = Float(view.drawableSize.width / view.drawableSize.height)
        let projectionMatrix = simd_float4x4(perspectiveProjectionFoVY: .pi / 3,
                                             aspectRatio: aspectRatio,
                                             near: 0.01,
                                             far: 100)

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
                                                             direction: light.direction,
                                                             type: light.type.rawValue)
        }
    }

    func updateNodeConstants() {
        let t = Float(time)

        let rotationAxis = normalize(SIMD3<Float>(0, 1, 0))
        let rotation = simd_float4x4(rotateAbout: rotationAxis, byAngle: t)

        cowNode.transform = rotation

        nodeConstantsOffsets.removeAll()
        for node in nodes {
            var constants = NodeConstants(modelMatrix: node.worldTransform)

            let layout = MemoryLayout<NodeConstants>.self
            let offset = allocateConstantStorage(size: layout.size, alignment: layout.stride)
            let constantsPointer = constantBuffer.contents().advanced(by: offset)
            constantsPointer.copyMemory(from: &constants, byteCount: layout.size)
            nodeConstantsOffsets.append(offset)
        }
    }

    func drawShadows(light: Light, commandBuffer: MTLCommandBuffer) {
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.depthAttachment.loadAction = .clear
        renderPassDescriptor.depthAttachment.storeAction = .store
        renderPassDescriptor.depthAttachment.clearDepth = 1.0
        renderPassDescriptor.depthAttachment.texture = light.shadowTexture!

        let renderCommandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        renderCommandEncoder.setRenderPipelineState(shadowRenderPipelineState)

        renderCommandEncoder.setDepthStencilState(depthStencilState)
        renderCommandEncoder.setFrontFacing(.counterClockwise)
        renderCommandEncoder.setCullMode(.back)

        let shadowViewMatrix = light.worldTransform.inverse
        let shadowProjectionMatrix = light.projectionMatrix
        let shadowViewProjectionMatrix = shadowProjectionMatrix * shadowViewMatrix

        renderCommandEncoder.setVertexBuffer(constantBuffer, offset: 0, index: 2)
        renderCommandEncoder.setFragmentBuffer(constantBuffer, offset: 0, index: 2)

        for node in nodes {
            guard let mesh = node.mesh else { continue }

            var constants = ShadowConstants(modelViewProjectionMatrix: shadowViewProjectionMatrix * node.worldTransform)
            let layout = MemoryLayout<ShadowConstants>.self
            let constantOffset = allocateConstantStorage(size: layout.size, alignment: layout.stride)
            constantBuffer.contents().advanced(by: constantOffset).copyMemory(from: &constants, byteCount: layout.size)


            renderCommandEncoder.setVertexBufferOffset(constantOffset, index: 2)
            renderCommandEncoder.setFragmentBufferOffset(constantOffset, index: 2)

            for (i, meshBuffer) in mesh.vertexBuffers.enumerated() {
                renderCommandEncoder.setVertexBuffer(meshBuffer.buffer,
                                                     offset: meshBuffer.offset,
                                                     index: i)
            }

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

    func drawMainPass(renderPassDescriptor: MTLRenderPassDescriptor, commandBuffer: MTLCommandBuffer) {
        let renderCommandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        renderCommandEncoder.setRenderPipelineState(renderPipelineState)

        renderCommandEncoder.setDepthStencilState(depthStencilState)
        renderCommandEncoder.setFrontFacing(.counterClockwise)
        renderCommandEncoder.setCullMode(.back)

        // Pre-bind node constants (vertex)
        renderCommandEncoder.setVertexBuffer(constantBuffer, offset: 0, index: 2)
        // Bind frame constants (vertex)
        renderCommandEncoder.setVertexBuffer(constantBuffer, offset: frameConstantsOffset, index: 3)
        // Bind frame constants (vertex)
        renderCommandEncoder.setFragmentBuffer(constantBuffer, offset: frameConstantsOffset, index: 3)
        // Bind light data (fragment)
        renderCommandEncoder.setFragmentBuffer(constantBuffer, offset: lightConstantsOffset, index: 4)

        renderCommandEncoder.setFragmentTexture(sunLight.shadowTexture!, index: 1)

        for (objectIndex, node) in nodes.enumerated() {
            guard let mesh = node.mesh else { continue }

            renderCommandEncoder.setVertexBufferOffset(nodeConstantsOffsets[objectIndex], index: 2)

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

    private func renderPassDescriptor(colorTexture: MTLTexture,
                                      depthTexture: MTLTexture?) -> MTLRenderPassDescriptor
    {
        let renderPassDescriptor = MTLRenderPassDescriptor()

        let msaaEnabled = (rasterSampleCount > 1)

        if msaaEnabled {
            renderPassDescriptor.colorAttachments[0].texture = msaaColorTexture
            renderPassDescriptor.colorAttachments[0].resolveTexture = colorTexture
            renderPassDescriptor.colorAttachments[0].loadAction = .clear
            renderPassDescriptor.colorAttachments[0].clearColor = view.clearColor
            renderPassDescriptor.colorAttachments[0].storeAction = .multisampleResolve

            renderPassDescriptor.depthAttachment.texture = msaaDepthTexture
            renderPassDescriptor.depthAttachment.resolveTexture = depthTexture
            renderPassDescriptor.depthAttachment.loadAction = .clear
            renderPassDescriptor.depthAttachment.clearDepth = 1.0
            renderPassDescriptor.depthAttachment.storeAction = .multisampleResolve
        } else {
            renderPassDescriptor.colorAttachments[0].texture = colorTexture
            renderPassDescriptor.colorAttachments[0].loadAction = .clear
            renderPassDescriptor.colorAttachments[0].clearColor = view.clearColor
            renderPassDescriptor.colorAttachments[0].storeAction = .store

            renderPassDescriptor.depthAttachment.texture = depthTexture
            renderPassDescriptor.depthAttachment.loadAction = .clear
            renderPassDescriptor.depthAttachment.clearDepth = view.clearDepth
            renderPassDescriptor.depthAttachment.storeAction = .dontCare
        }

        return renderPassDescriptor
    }

    private func makeMSAARenderTargetsIfNeeded() {
        if rasterSampleCount == 1 { return }

        let drawableWidth = Int(view.drawableSize.width)
        let drawableHeight = Int(view.drawableSize.height)

        if msaaColorTexture == nil ||
           msaaColorTexture?.width != drawableWidth ||
           msaaColorTexture?.height != drawableHeight ||
           msaaColorTexture?.sampleCount != rasterSampleCount
        {
            let textureDescriptor = MTLTextureDescriptor()
            textureDescriptor.textureType = .type2DMultisample
            textureDescriptor.sampleCount = rasterSampleCount
            textureDescriptor.pixelFormat = view.colorPixelFormat
            textureDescriptor.width = drawableWidth
            textureDescriptor.height = drawableHeight
            textureDescriptor.storageMode = .private
            textureDescriptor.usage = .renderTarget

            msaaColorTexture = device.makeTexture(descriptor: textureDescriptor)
            msaaColorTexture?.label = "MSAA x\(rasterSampleCount) Color"
        }

        if msaaDepthTexture == nil ||
           msaaDepthTexture?.width != drawableWidth ||
           msaaDepthTexture?.height != drawableHeight ||
           msaaDepthTexture?.sampleCount != rasterSampleCount
        {
            let textureDescriptor = MTLTextureDescriptor()
            textureDescriptor.textureType = .type2DMultisample
            textureDescriptor.sampleCount = rasterSampleCount
            textureDescriptor.pixelFormat = view.depthStencilPixelFormat
            textureDescriptor.width = drawableWidth
            textureDescriptor.height = drawableHeight
            textureDescriptor.storageMode = .private
            textureDescriptor.usage = .renderTarget

            msaaDepthTexture = device.makeTexture(descriptor: textureDescriptor)
            msaaDepthTexture?.label = "MSAA x\(rasterSampleCount) Depth"
        }
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        msaaColorTexture = nil
        msaaDepthTexture = nil
    }

    func draw(in view: MTKView) {
        frameSemaphore.wait()

        let initialConstantOffset = currentConstantBufferOffset

        makeMSAARenderTargetsIfNeeded()

        updateLightConstants()
        updateFrameConstants()
        updateNodeConstants()

        guard let drawable = view.currentDrawable else { return }
        let renderPassDescriptor = renderPassDescriptor(colorTexture: drawable.texture,
                                                        depthTexture: view.depthStencilTexture)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        drawShadows(light: sunLight, commandBuffer: commandBuffer)
        drawMainPass(renderPassDescriptor: renderPassDescriptor, commandBuffer: commandBuffer)

        commandBuffer.present(view.currentDrawable!)

        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.frameSemaphore.signal()
        }

        commandBuffer.commit()

        let constantSize = currentConstantBufferOffset - initialConstantOffset
        if (constantSize > MaxConstantsSize / MaxOutstandingFrameCount) {
            print("Insufficient constant storage: frame consumed \(constantSize) bytes of total \(MaxConstantsSize) bytes")
        }

        frameIndex += 1
    }
}
