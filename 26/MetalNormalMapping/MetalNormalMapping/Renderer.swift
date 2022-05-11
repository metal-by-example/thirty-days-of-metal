
import Foundation
import Metal
import MetalKit
import simd

let MaxOutstandingFrameCount = 3
let MaxConstantsSize = 1_024 * 1_024
let MinBufferAlignment = 256

enum RenderMode: UInt32 {
    case geometricNormals
    case normalMapped
}

struct InstanceConstants {
    var modelMatrix: float4x4
    var normalMatrix: float3x3
}

struct FrameConstants {
    var projectionMatrix: float4x4
    var viewMatrix: float4x4
    var lightCount: UInt32
    var renderMode: UInt32
}

struct LightConstants {
    var intensity: simd_float3
    var position: simd_float3
    var direction: simd_float3
    var type: UInt32
}

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let view: MTKView

    var renderMode = RenderMode.normalMapped

    var pointOfView = Node()
    var nodes = [Node]()
    var lights = [Light]()

    private var nodeConstantOffsets = [Int]()

    private let mdlVertexDescriptor = MDLVertexDescriptor()
    private var vertexDescriptor: MTLVertexDescriptor!

    private var renderPipelineState: MTLRenderPipelineState!
    private var depthStencilState: MTLDepthStencilState!
    private var samplerState: MTLSamplerState!

    private var bufferAllocator: MTKMeshBufferAllocator!
    private var textureLoader: MTKTextureLoader!

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
        view.clearColor = MTLClearColor(red: 0.0, green: 0.5, blue: 0.9, alpha: 1.0)
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
        mdlVertexDescriptor.vertexAttributes[2].name = MDLVertexAttributeTangent
        mdlVertexDescriptor.vertexAttributes[2].format = .float4
        mdlVertexDescriptor.vertexAttributes[2].offset = 24
        mdlVertexDescriptor.vertexAttributes[2].bufferIndex = 0
        mdlVertexDescriptor.vertexAttributes[3].name = MDLVertexAttributeTextureCoordinate
        mdlVertexDescriptor.vertexAttributes[3].format = .float2
        mdlVertexDescriptor.vertexAttributes[3].offset = 40
        mdlVertexDescriptor.vertexAttributes[3].bufferIndex = 0
        mdlVertexDescriptor.bufferLayouts[0].stride = 48

        vertexDescriptor = MTKMetalVertexDescriptorFromModelIO(mdlVertexDescriptor)!

        let assetURL = Bundle.main.url(forResource: "alley", withExtension: "obj")!
        let asset = MDLAsset(url: assetURL, vertexDescriptor: nil, bufferAllocator: bufferAllocator)
        asset.loadTextures()
        if let mdlMesh = asset.meshes.first {
            mdlMesh.addTangentBasis(forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                                    normalAttributeNamed: MDLVertexAttributeNormal,
                                    tangentAttributeNamed: MDLVertexAttributeTangent)
            mdlMesh.vertexDescriptor = mdlVertexDescriptor
            let mesh = try! Mesh(mesh: mdlMesh, device: device)
            let node = Node(mesh: mesh)
            nodes.append(node)
        }

        let ambientLight = Light()
        ambientLight.type = .ambient
        ambientLight.intensity = 0.05

        let localLight = Light()
        localLight.type = .omni
        localLight.intensity = 15.0
        localLight.worldTransform = float4x4(lookAt: SIMD3<Float>(0, 0, -1),
                                           from: SIMD3<Float>(0, 5, 0),
                                           up: SIMD3<Float>(0, 1, 0))

        let sunLight = Light()
        sunLight.type = .directional
        sunLight.intensity = 0.3
        sunLight.worldTransform = float4x4(lookAt: SIMD3<Float>(0, 0, 0),
                                           from: SIMD3<Float>(5, 5, 5),
                                           up: SIMD3<Float>(0, 1, 0))

        lights = [ambientLight, sunLight, localLight]
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

        do {
            renderPipelineState = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
        } catch {
            fatalError("Unable to create render pipeline state: \(error)")
        }

        let depthStencilDescriptor = MTLDepthStencilDescriptor()
        depthStencilDescriptor.depthCompareFunction = .lessEqual
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
        let projectionMatrix = float4x4(perspectiveProjectionFoVY: .pi / 3,
                                        aspectRatio: aspectRatio,
                                        near: 0.01,
                                        far: 100)

        let cameraMatrix = pointOfView.worldTransform
        let viewMatrix = cameraMatrix.inverse

        var constants = FrameConstants(projectionMatrix: projectionMatrix,
                                       viewMatrix: viewMatrix,
                                       lightCount: UInt32(lights.count),
                                       renderMode: renderMode.rawValue)

        let layout = MemoryLayout<FrameConstants>.self
        frameConstantsOffset = allocateConstantStorage(size: layout.size, alignment: layout.stride)
        let constantsPointer = constantBuffer.contents().advanced(by: frameConstantsOffset)
        constantsPointer.copyMemory(from: &constants, byteCount: layout.size)

    }

    func updateLightConstants() {
        let cameraMatrix = pointOfView.worldTransform
        let viewMatrix = cameraMatrix.inverse

        let layout = MemoryLayout<LightConstants>.self
        lightConstantsOffset = allocateConstantStorage(size: layout.stride * lights.count, alignment: layout.stride)
        let lightsBufferPointer = constantBuffer.contents().advanced(by: lightConstantsOffset).assumingMemoryBound(to: LightConstants.self)

        for (lightIndex, light) in lights.enumerated() {
            let lightModelViewMatrix = viewMatrix * light.worldTransform
            let lightPosition = lightModelViewMatrix.columns.3.xyz
            let lightDirection = -lightModelViewMatrix.columns.2.xyz

            lightsBufferPointer[lightIndex] = LightConstants(intensity: light.color * light.intensity,
                                                             position: lightPosition,
                                                             direction: lightDirection,
                                                             type: light.type.rawValue)
        }
    }

    func updateNodeConstants(_ timestep: Float) {
        let cameraMatrix = pointOfView.worldTransform
        let viewMatrix = cameraMatrix.inverse

        nodeConstantOffsets.removeAll()
        for node in nodes {
            let layout = MemoryLayout<InstanceConstants>.self
            let offset = allocateConstantStorage(size: layout.stride, alignment: layout.stride)

            let modelMatrix = node.worldTransform
            let modelViewMatrix = viewMatrix * modelMatrix
            let normalMatrix = modelViewMatrix.upperLeft3x3.transpose.inverse

            let instanceConstants = constantBuffer.contents().advanced(by: offset)
            var instance = InstanceConstants(modelMatrix: modelMatrix, normalMatrix: normalMatrix)
            instanceConstants.copyMemory(from: &instance, byteCount: layout.size)

            nodeConstantOffsets.append(offset)
        }
    }

    func drawMainPass(renderPassDescriptor: MTLRenderPassDescriptor, commandBuffer: MTLCommandBuffer) {
        let renderCommandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!

        renderCommandEncoder.setDepthStencilState(depthStencilState)
        renderCommandEncoder.setFrontFacing(.counterClockwise)
        renderCommandEncoder.setCullMode(.back)

        renderCommandEncoder.setRenderPipelineState(renderPipelineState)

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

            for submesh in mesh.submeshes {
                let material = submesh.material

                renderCommandEncoder.setFragmentTexture(material?.baseColorTexture, index: 0)
                renderCommandEncoder.setFragmentSamplerState(samplerState, index: 0)
                renderCommandEncoder.setFragmentTexture(material?.normalTexture, index: 1)

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
