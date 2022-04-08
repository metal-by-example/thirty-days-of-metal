
import Cocoa
import Metal
import MetalKit

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let view: MTKView
    private var renderPipelineState: MTLRenderPipelineState!
    private var vertexBuffer: MTLBuffer!

    init(device: MTLDevice, view: MTKView) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.view = view

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
            -0.8,  0.4,    1.0, 0.0, 1.0, 1.0,
             0.4, -0.8,    0.0, 1.0, 1.0, 1.0,
             0.8,  0.8,    1.0, 1.0, 0.0, 1.0,
        ]
        vertexBuffer = device.makeBuffer(bytes: &vertexData,
                                         length: MemoryLayout<Float>.stride * vertexData.count,
                                         options: .storageModeShared)
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
    }

    func draw(in view: MTKView) {
        guard let renderPassDescriptor = view.currentRenderPassDescriptor else { return }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        let renderCommandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        renderCommandEncoder.setRenderPipelineState(renderPipelineState)
        renderCommandEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        renderCommandEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        renderCommandEncoder.endEncoding()

        commandBuffer.present(view.currentDrawable!)
        commandBuffer.commit()
    }
}
