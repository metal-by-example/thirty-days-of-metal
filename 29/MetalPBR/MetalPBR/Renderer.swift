
import Foundation
import Metal
import MetalKit
import simd

let MaxOutstandingFrameCount = 3
let MaxConstantsSize = 1_024 * 1_024
let MinBufferAlignment = 256

struct VertexBufferIndex {
    static let vertexAttributes = 0
    static let instanceConstants = 8
    static let frameConstants = 9
    static let lights = 10
    static let jointMatrices = 11
}

struct FragmentBufferIndex {
    static let frameConstants = 0
    static let lights = 1
    static let materialConstants = 2
}

struct FragmentTextureIndex {
    static let baseColor = 0
    static let normal = 1
    static let emissive = 2
    static let metalness = 3
    static let roughness = 4
    static let ambientOcclusion = 5
}

struct InstanceConstants {
    var modelMatrix: float4x4
    var normalMatrix: float3x3
}

struct FrameConstants {
    var viewMatrix: float4x4
    var viewProjectionMatrix: float4x4
    var lightCount: UInt32
}

struct LightConstants {
    var position: SIMD4<Float>
    var direction: SIMD4<Float>
    var intensity: SIMD4<Float>
}

struct MaterialConstants {
    var baseColorFactor: SIMD4<Float>
    var emissiveColor: SIMD4<Float>
    var metalnessFactor: Float
    var roughnessFactor: Float
    var occlusionWeight: Float
    var opacity: Float
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

    private var bufferAllocator: MTKMeshBufferAllocator!

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
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColor(red: 0.95, green: 0.95, blue: 0.95, alpha: 1.0)
        view.sampleCount = 4

        bufferAllocator = MTKMeshBufferAllocator(device: device)

        makeScene()
        makeResources()
        makePipelines()
    }

    func makeScene() {
        let assetNames = ["BlueTile", "MetalTiles", "Concrete", "RedPlastic", "RoughMetal", "Wood" ]

        let spherePositions = [
            SIMD3<Float>(-1.2,  0.6, 0.0),
            SIMD3<Float>( 0.0,  0.6, 0.0),
            SIMD3<Float>( 1.2,  0.6, 0.0),
            SIMD3<Float>(-1.2, -0.6, 0.0),
            SIMD3<Float>( 0.0, -0.6, 0.0),
            SIMD3<Float>( 1.2, -0.6, 0.0)
        ]

        for (assetName, position) in zip(assetNames, spherePositions) {
            let assetURL = Bundle.main.url(forResource: assetName, withExtension: "usdz")!
            let assetScene = Scene(url: assetURL, device: device)
            assetScene.rootNode.transform = simd_float4x4(translate: position)
            scene.rootNode.addChildNode(assetScene.rootNode)
        }

        let leftLight = Light()
        leftLight.type = .omni
        leftLight.intensity = 25
        leftLight.worldTransform = float4x4(lookAt: SIMD3<Float>(0, 0, 0),
                                            from: SIMD3<Float>(-6, 1, 1),
                                            up: SIMD3<Float>(0, 1, 0))

        let rightLight = Light()
        rightLight.type = .omni
        rightLight.intensity = 50
        rightLight.worldTransform = float4x4(lookAt: SIMD3<Float>(0, 0, 0),
                                             from: SIMD3<Float>(6, 1, 1),
                                             up: SIMD3<Float>(0, 1, 0))

        let rearLight = Light()
        rearLight.type = .omni
        rearLight.intensity = 50
        rearLight.worldTransform = float4x4(lookAt: SIMD3<Float>(0, 0, 0),
                                              from: SIMD3<Float>(0, 1, -6),
                                              up: SIMD3<Float>(0, 1, 0))

        let frontLight = Light()
        frontLight.type = .omni
        frontLight.intensity = 150
        frontLight.worldTransform = float4x4(lookAt: SIMD3<Float>(0, 0, 0),
                                             from: SIMD3<Float>(0, 0, 6),
                                             up: SIMD3<Float>(0, 1, 0))

        lights = [leftLight, rightLight, rearLight, frontLight]
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

            renderPipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
            renderPipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .one
            renderPipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            renderPipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
            renderPipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
            renderPipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            renderPipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add

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
                                        far: 1000)

        let cameraMatrix = pointOfView.worldTransform
        let viewMatrix = cameraMatrix.inverse

        let viewProjectionMatrix = projectionMatrix * viewMatrix

        var constants = FrameConstants(viewMatrix: viewMatrix,
                                       viewProjectionMatrix: viewProjectionMatrix,
                                       lightCount: UInt32(lights.count))

        let layout = MemoryLayout<FrameConstants>.self
        frameConstantsOffset = allocateConstantStorage(size: layout.size, alignment: layout.stride)
        let constantsPointer = constantBuffer.contents().advanced(by: frameConstantsOffset)
        constantsPointer.copyMemory(from: &constants, byteCount: layout.size)

        let lightConstantsLayout = MemoryLayout<LightConstants>.self
        lightConstantsOffset = allocateConstantStorage(size: lightConstantsLayout.stride * lights.count,
                                                       alignment: lightConstantsLayout.stride)
        let lightsBufferPointer = constantBuffer.contents().advanced(by: lightConstantsOffset)
            .assumingMemoryBound(to: LightConstants.self)

        for (lightIndex, light) in zip(0..., lights) {
            let lightModelViewMatrix = viewMatrix * light.worldTransform
            let lightPosition = lightModelViewMatrix.columns.3.xyz
            // We use a convention in which the light's direction is along the +Z axis of its transform.
            // Normalizing this produces the (eye-space) L vector.
            let lightDirection = lightModelViewMatrix.columns.2.xyz

            let directionW: Float = light.type == .directional ? 0.0 : 1.0
            lightsBufferPointer[lightIndex] = LightConstants(position: SIMD4<Float>(lightPosition, 1.0),
                                                             direction: SIMD4<Float>(lightDirection, directionW),
                                                             intensity: SIMD4<Float>(light.color  * light.intensity, 1.0))
        }
    }

    func updateNodeConstants(_ time: TimeInterval) {
        let cameraMatrix = pointOfView.worldTransform
        let viewMatrix = cameraMatrix.inverse

        nodeConstantOffsets.removeAll()
        for node in scene.nodes {
            node.update(at: time)

            let layout = MemoryLayout<InstanceConstants>.self
            let offset = allocateConstantStorage(size: layout.stride, alignment: layout.stride)

            let modelMatrix = node.worldTransform
            let modelViewMatrix = viewMatrix  * modelMatrix
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
                    renderCommandEncoder.setVertexBuffer(constantBuffer,
                                                         offset: jointMatrixOffset,
                                                         index: VertexBufferIndex.jointMatrices)
                    boundSkeleton = skeleton
                }
            } else {
                renderCommandEncoder.setRenderPipelineState(defaultRenderPipelineState)

                var defaultJointMatrix = matrix_identity_float4x4
                renderCommandEncoder.setVertexBytes(&defaultJointMatrix,
                                                    length: transformLayout.stride,
                                                    index: VertexBufferIndex.jointMatrices)
            }

            renderCommandEncoder.setVertexBuffer(constantBuffer,
                                                 offset: nodeConstantOffsets[nodeIndex],
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
                    renderCommandEncoder.setFragmentTexture(material.emissiveTexture,
                                                            index: FragmentTextureIndex.emissive)
                    renderCommandEncoder.setFragmentTexture(material.metalnessTexture,
                                                            index: FragmentTextureIndex.metalness)
                    renderCommandEncoder.setFragmentTexture(material.roughnessTexture,
                                                            index: FragmentTextureIndex.roughness)
                    renderCommandEncoder.setFragmentTexture(material.occlusionTexture,
                                                            index: FragmentTextureIndex.ambientOcclusion)
                    renderCommandEncoder.setFragmentTexture(material.normalTexture,
                                                            index: FragmentTextureIndex.normal)

                    var materialConstants = MaterialConstants(baseColorFactor: material.baseColor,
                                                              emissiveColor: material.emissiveColor,
                                                              metalnessFactor: material.metalnessFactor,
                                                              roughnessFactor: material.roughnessFactor,
                                                              occlusionWeight: 1.0,
                                                              opacity: material.opacity)

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
