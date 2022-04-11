
import Cocoa
import Metal
import MetalKit
import simd

let MaxOutstandingFrameCount = 3

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let view: MTKView

    var time: TimeInterval = 0.0

    private var renderPipelineState: MTLRenderPipelineState!
    private var vertexBuffer: MTLBuffer!

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

        super.init()

        view.device = device
        view.delegate = self
        view.clearColor = MTLClearColor(red: 0.95, green: 0.95, blue: 0.95, alpha: 1.0)

        makePipeline()
        makeResources()
    }

    func makePipeline() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Unable to create default Metal library")
        }

        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float2
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.attributes[1].format = .float4
        vertexDescriptor.attributes[1].offset = MemoryLayout<Float>.stride * 2
        vertexDescriptor.attributes[1].bufferIndex = 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<Float>.stride * 6

        let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
        renderPipelineDescriptor.vertexDescriptor = vertexDescriptor

        renderPipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat

        renderPipelineDescriptor.vertexFunction = library.makeFunction(name: "vertex_main")!
        renderPipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragment_main")!

        do {
            renderPipelineState = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
        } catch {
            fatalError("Error while creating render pipeline state: \(error)")
        }
    }

    func makeResources() {
        var vertexData: [Float] = [
        //    x     y       r    g    b    a
            -100,  -20,    1.0, 0.0, 1.0, 1.0,
             100,  -60,    0.0, 1.0, 1.0, 1.0,
              30,  100,    1.0, 1.0, 0.0, 1.0,
        ]
        vertexBuffer = device.makeBuffer(bytes: &vertexData,
                                         length: MemoryLayout<Float>.stride * vertexData.count,
                                         options: .storageModeShared)

        constantBuffer = device.makeBuffer(length: constantsStride * MaxOutstandingFrameCount,
                                            options: .storageModeShared)
    }

    func updateConstants() {
        time += 1.0 / Double(view.preferredFramesPerSecond)
        let t = Float(time)

        let pulseRate: Float = 1.5
        let scaleFactor = 1.0 + 0.5 * cos(pulseRate * t)
        let scale = SIMD2<Float>(scaleFactor, scaleFactor)
        let scaleMatrix = simd_float4x4(scale2D: scale)

        let rotationRate: Float = 2.5
        let rotationAngle = rotationRate * t
        let rotationMatrix = simd_float4x4(rotateZ: rotationAngle)

        let orbitalRadius: Float = 200
        let translation = orbitalRadius * SIMD2<Float>(cos(t), sin(t))
        let translationMatrix = simd_float4x4(translate2D: translation)

        let modelMatrix = translationMatrix * rotationMatrix * scaleMatrix

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
        renderCommandEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        renderCommandEncoder.setVertexBuffer(constantBuffer, offset: currentConstantBufferOffset, index: 1)
        renderCommandEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        renderCommandEncoder.endEncoding()

        commandBuffer.present(view.currentDrawable!)

        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.frameSemaphore.signal()
        }

        commandBuffer.commit()

        frameIndex += 1
    }
}
