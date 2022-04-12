
import Cocoa
import Metal
import MetalKit
import simd

let MaxOutstandingFrameCount = 3

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let view: MTKView

    let useIndexedMesh = false
    let mesh: SimpleMesh

    var time: TimeInterval = 0.0

    private var renderPipelineState: MTLRenderPipelineState!

    private var frameSemaphore = DispatchSemaphore(value: MaxOutstandingFrameCount)
    private var frameIndex: Int

    private var constantBuffer: MTLBuffer!
    private let constantsSize: Int
    private let constantsStride: Int
    private var currentConstantBufferOffset: Int

    init(device: MTLDevice, view: MTKView) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.view = view
        self.frameIndex = 0
        self.constantsSize = MemoryLayout<simd_float4x4>.size
        self.constantsStride = align(constantsSize, upTo: 256)
        self.currentConstantBufferOffset = 0

        let color = SIMD4<Float>(0.0, 0.5, 0.8, 1.0)
        mesh = useIndexedMesh ?
            SimpleMesh(indexedPlanarPolygonSideCount: 11, radius: 250, color: color, device: device) :
            SimpleMesh(planarPolygonSideCount: 11, radius: 250, color: color, device: device)

        super.init()

        view.device = device
        view.delegate = self
        view.clearColor = MTLClearColor(red: 0.95, green: 0.95, blue: 0.95, alpha: 1.0)

        makeResources()
        makePipeline()
    }

    func makeResources() {
        constantBuffer = device.makeBuffer(length: constantsStride * MaxOutstandingFrameCount,
                                            options: .storageModeShared)
        constantBuffer.label = "Dynamic Constant Buffer"
    }

    func makePipeline() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Unable to create default Metal library")
        }

        let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
        renderPipelineDescriptor.vertexDescriptor = mesh.vertexDescriptor

        renderPipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat

        renderPipelineDescriptor.vertexFunction = library.makeFunction(name: "vertex_main")!
        renderPipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragment_main")!

        do {
            renderPipelineState = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
        } catch {
            fatalError("Error while creating render pipeline state: \(error)")
        }
    }

    func updateConstants() {
        //time += 1.0 / Double(view.preferredFramesPerSecond)
        let t = Float(time)

        let rotationRate: Float = 1.5
        let rotationAngle = rotationRate * t
        let rotationMatrix = simd_float4x4(rotateZ: rotationAngle)

        let modelMatrix = rotationMatrix

        let aspectRatio = Float(view.drawableSize.width / view.drawableSize.height)
        let canvasWidth: Float = 800
        let canvasHeight = canvasWidth / aspectRatio
        let projectionMatrix = simd_float4x4(orthographicProjectionWithLeft: -canvasWidth / 2,
                                             top: canvasHeight / 2,
                                             right: canvasWidth / 2,
                                             bottom: -canvasHeight / 2,
                                             near: 0.0,
                                             far: 1.0)

        var transformMatrix = projectionMatrix * modelMatrix

        currentConstantBufferOffset = (frameIndex % MaxOutstandingFrameCount) * constantsStride
        let constants = constantBuffer.contents().advanced(by: currentConstantBufferOffset)
        constants.copyMemory(from: &transformMatrix, byteCount: constantsSize)
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

        renderCommandEncoder.setVertexBuffer(constantBuffer, offset: currentConstantBufferOffset, index: 2)

        for (i, vertexBuffer) in mesh.vertexBuffers.enumerated() {
            renderCommandEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: i)
        }

        if let indexBuffer = mesh.indexBuffer {
            renderCommandEncoder.drawIndexedPrimitives(type: mesh.primitiveType,
                                                       indexCount: mesh.indexCount,
                                                       indexType: mesh.indexType,
                                                       indexBuffer: indexBuffer,
                                                       indexBufferOffset: 0)
        } else {
            renderCommandEncoder.drawPrimitives(type: mesh.primitiveType,
                                                vertexStart: 0,
                                                vertexCount: mesh.vertexCount)
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
