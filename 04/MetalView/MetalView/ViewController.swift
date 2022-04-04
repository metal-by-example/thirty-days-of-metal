import Cocoa
import Metal
import MetalKit

class ViewController: NSViewController, MTKViewDelegate {

    @IBOutlet weak var metalView: MTKView!
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!

    override func viewDidLoad() {
        super.viewDidLoad()

        device = MTLCreateSystemDefaultDevice()
        metalView.device = device
        metalView.clearColor = MTLClearColor(red: 0.0, green: 0.5, blue: 1.0, alpha: 1.0)
        metalView.delegate = self

        commandQueue = device.makeCommandQueue()!
    }

    // MARK: - MTKViewDelegate methods

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
    }

    func draw(in view: MTKView) {
        let commandBuffer = commandQueue.makeCommandBuffer()!

        guard let renderPassDescriptor = view.currentRenderPassDescriptor else {
            print("Didn't get a render pass descriptor from MTKView; dropping frame...")
            return
        }

        let renderPassEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        renderPassEncoder.endEncoding()

        commandBuffer.present(view.currentDrawable!)

        commandBuffer.commit()
    }
}
