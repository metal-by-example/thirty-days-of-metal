
import Foundation
import Metal
import MetalKit
import simd

let MaxOutstandingFrameCount = 3
let MaxConstantsSize = 1_024 * 1_024
let MinBufferAlignment = 256

struct VertexBufferIndex {
    static let vertexAttributes = 0
    static let instanceConstants = 1
    static let frameConstants = 2
    static let jointMatrices = 3
}

struct FragmentBufferIndex {
    static let frameConstants = 0
    static let lights = 1
    static let materialConstants = 2
}

struct FragmentTextureIndex {
    static let baseColor = 0
    static let normal = 1
}

struct InstanceConstants {
    var modelMatrix: float4x4
    var normalMatrix: float3x3
}

struct FrameConstants {
    var projectionMatrix: float4x4
    var viewMatrix: float4x4
    var lightCount: UInt32
}

struct LightConstants {
    var intensity: simd_float3
    var position: simd_float3
    var direction: simd_float3
    var type: UInt32
}

struct MaterialConstants {
    var baseColorFactor: SIMD4<Float>
    var shininess: Float
}

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let view: MTKView

    var scene = Scene()
    var lights = [Light]()
    var pointOfView = Node()
    var characterRootNode: Node?

    private var nodeConstantOffsets = [Int]()
    private var boundSkeleton: Skeleton? = nil

    private var defaultRenderPipelineState: MTLRenderPipelineState!
    private var skinnedRenderPipelineState: MTLRenderPipelineState!
    private var depthStencilState: MTLDepthStencilState!
    private var samplerState: MTLSamplerState!

    private var bufferAllocator: MTKMeshBufferAllocator!
    private var textureLoader: MTKTextureLoader!

    private var constantBuffer: MTLBuffer!
    private var currentConstantBufferOffset = 0
    private var frameConstantsOffset = 0
    private var lightConstantsOffset = 0
    private var jointMatrixOffset = 0

    private var frameSemaphore = DispatchSemaphore(value: MaxOutstandingFrameCount)
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
        view.clearColor = MTLClearColor(red: 0.55, green: 0.85, blue: 1.0, alpha: 1.0)
        view.sampleCount = 1

        bufferAllocator = MTKMeshBufferAllocator(device: device)
        textureLoader = MTKTextureLoader(device: device)

        makeScene()
        makeResources()
        makePipelines()
    }

    func makeScene() {
        let characterURL = Bundle.main.url(forResource: "Character", withExtension: "usdz")!
        let characterScene = Scene(url: characterURL, device: device)
        for childNode in characterScene.rootNode.childNodes {
            scene.rootNode.addChildNode(childNode)
        }

        let worldTransform = float4x4(translate: SIMD3<Float>(0, -1.0, 0))
        let worldURL = Bundle.main.url(forResource: "GrassPlatform", withExtension: "usdz")!
        let worldScene = Scene(url: worldURL, device: device)
        worldScene.rootNode.childNodes.first?.transform = worldTransform
        for childNode in worldScene.rootNode.childNodes {
            scene.rootNode.addChildNode(childNode)
        }

        characterRootNode = scene.rootNode.childNode(named: "CharacterArmature", recursive: true)

        if let defaultAnimation = characterScene.animations.first {
            let (animation, target) = defaultAnimation
            target.runAnimation(animation)
        }

        let ambientLight = Light()
        ambientLight.type = .ambient
        ambientLight.intensity = 0.05

        let sunLight = Light()
        sunLight.type = .directional
        sunLight.intensity = 0.9
        sunLight.worldTransform = float4x4(lookAt: SIMD3<Float>(0, 0, 0),
                                           from: SIMD3<Float>(5, 5, 5),
                                           up: SIMD3<Float>(0, 1, 0))

        lights = [ambientLight, sunLight]
    }

    func makeResources() {
        constantBuffer = device.makeBuffer(length: MaxConstantsSize, options: .storageModeShared)
        constantBuffer.label = "Dynamic Constant Buffer"
    }

    func makePipelines() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Unable to create default Metal library")
        }

        do {
            let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
            renderPipelineDescriptor.rasterSampleCount = view.sampleCount

            let mdlVertexDescriptor = Mesh.defaultVertexDescriptor
            renderPipelineDescriptor.vertexDescriptor = MTKMetalVertexDescriptorFromModelIO(mdlVertexDescriptor)

            renderPipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
            renderPipelineDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat

            renderPipelineDescriptor.vertexFunction = library.makeFunction(name: "vertex_main")!
            renderPipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragment_main")!
            defaultRenderPipelineState = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
        } catch {
            fatalError("Unable to create render pipeline state: \(error)")
        }

        do {
            let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
            renderPipelineDescriptor.rasterSampleCount = view.sampleCount

            let mdlVertexDescriptor = Mesh.skinnedVertexDescriptor
            renderPipelineDescriptor.vertexDescriptor = MTKMetalVertexDescriptorFromModelIO(mdlVertexDescriptor)

            renderPipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
            renderPipelineDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat

            renderPipelineDescriptor.vertexFunction = library.makeFunction(name: "skinned_vertex_main")!
            renderPipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragment_main")!
            skinnedRenderPipelineState = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
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
                                       lightCount: UInt32(lights.count))

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
        let lightsBufferPointer = constantBuffer.contents().advanced(by: lightConstantsOffset)
            .assumingMemoryBound(to: LightConstants.self)

        for (lightIndex, light) in zip(0..., lights) {
            let lightModelViewMatrix = viewMatrix * light.worldTransform
            let lightPosition = lightModelViewMatrix.columns.3.xyz
            let lightDirection = -lightModelViewMatrix.columns.2.xyz

            lightsBufferPointer[lightIndex] = LightConstants(intensity: light.color * light.intensity,
                                                             position: lightPosition,
                                                             direction: lightDirection,
                                                             type: light.type.rawValue)
        }
    }

    func updateNodeConstants(_ time: TimeInterval) {
        let cameraMatrix = pointOfView.worldTransform
        let viewMatrix = cameraMatrix.inverse

        // Character root motion
        if let character = characterRootNode {
            let circuitRadius: Float = 3.0
            let circuitDuration: TimeInterval = 6.0
            let runTime = fmod(time, circuitDuration)
            let runAngle = Float((2.0 * .pi * runTime) / circuitDuration)
            let position = SIMD3<Float>(circuitRadius * cos(runAngle), 0, circuitRadius * sin(runAngle))
            let rotation = float4x4(rotateAbout: SIMD3<Float>(0, 1, 0), byAngle: runAngle)
            let translation = float4x4(translate: position)
            character.transform = translation * rotation
        }

        nodeConstantOffsets.removeAll()
        for node in scene.nodes {
            node.update(at: time)

            let layout = MemoryLayout<InstanceConstants>.self
            let offset = allocateConstantStorage(size: layout.stride, alignment: layout.stride)

            let modelMatrix = node.worldTransform
            let modelViewMatrix = viewMatrix * modelMatrix
            let normalMatrix = modelViewMatrix.upperLeft3x3.transpose.inverse

            let instanceConstants = constantBuffer.contents().advanced(by: offset)
            var instance = InstanceConstants(modelMatrix: modelMatrix,
                                             normalMatrix: normalMatrix)
            instanceConstants.copyMemory(from: &instance, byteCount: layout.size)

            nodeConstantOffsets.append(offset)
        }
    }

    func drawMainPass(renderPassDescriptor: MTLRenderPassDescriptor, commandBuffer: MTLCommandBuffer) {
        let renderCommandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!

        renderCommandEncoder.setDepthStencilState(depthStencilState)
        renderCommandEncoder.setFrontFacing(.counterClockwise)
        renderCommandEncoder.setCullMode(.back)

        renderCommandEncoder.setVertexBuffer(constantBuffer, offset: frameConstantsOffset,
                                             index: VertexBufferIndex.frameConstants)
        renderCommandEncoder.setFragmentBuffer(constantBuffer, offset: frameConstantsOffset,
                                               index: FragmentBufferIndex.frameConstants)
        renderCommandEncoder.setFragmentBuffer(constantBuffer, offset: lightConstantsOffset,
                                               index: FragmentBufferIndex.lights)

        boundSkeleton = nil
        for (nodeIndex, node) in zip(0..., scene.nodes) {
            guard let mesh = node.mesh else { continue }

            let transformLayout = MemoryLayout<float4x4>.self
            if let skinner = node.skinner {
                renderCommandEncoder.setRenderPipelineState(skinnedRenderPipelineState)

                let skeleton = skinner.skeleton
                if (skeleton !== boundSkeleton) {
                    jointMatrixOffset = allocateConstantStorage(size: transformLayout.stride * skeleton.jointCount,
                                                                alignment: transformLayout.stride)
                    skeleton.copyTransforms(into: constantBuffer, at: jointMatrixOffset)
                    renderCommandEncoder.setVertexBuffer(constantBuffer, offset: jointMatrixOffset,
                                                         index: VertexBufferIndex.jointMatrices)
                    boundSkeleton = skeleton
                }
            } else {
                renderCommandEncoder.setRenderPipelineState(defaultRenderPipelineState)

                var defaultJointMatrix = matrix_identity_float4x4
                renderCommandEncoder.setVertexBytes(&defaultJointMatrix, length: transformLayout.stride,
                                                    index: VertexBufferIndex.jointMatrices)
            }

            renderCommandEncoder.setVertexBuffer(constantBuffer, offset: nodeConstantOffsets[nodeIndex],
                                                 index: VertexBufferIndex.instanceConstants)

            for (bufferIndex, meshBuffer) in zip(0..., mesh.vertexBuffers) {
                renderCommandEncoder.setVertexBuffer(meshBuffer.buffer,
                                                     offset: meshBuffer.offset,
                                                     index: bufferIndex)
            }

            for submesh in mesh.submeshes {
                if let material = submesh.material {
                    renderCommandEncoder.setFragmentTexture(material.baseColorTexture,
                                                            index: FragmentTextureIndex.baseColor)
                    renderCommandEncoder.setFragmentSamplerState(samplerState, index: 0)
                    renderCommandEncoder.setFragmentTexture(material.normalTexture,
                                                            index: FragmentTextureIndex.normal)
                    var materialConstants = MaterialConstants(baseColorFactor: material.baseColor,
                                                              shininess: material.shininess)
                    renderCommandEncoder.setFragmentBytes(&materialConstants,
                                                          length: MemoryLayout.stride(ofValue: materialConstants),
                                                          index: FragmentBufferIndex.materialConstants)
                }

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
        updateNodeConstants(time)

        guard let renderPassDescriptor = view.currentRenderPassDescriptor else { return }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        drawMainPass(renderPassDescriptor: renderPassDescriptor, commandBuffer: commandBuffer)

        commandBuffer.present(view.currentDrawable!)

        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.frameSemaphore.signal()
        }

        commandBuffer.commit()
    }
}
