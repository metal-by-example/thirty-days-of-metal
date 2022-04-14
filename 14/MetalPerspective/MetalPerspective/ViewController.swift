
import Cocoa
import Metal
import MetalKit

class ViewController: NSViewController {
    @IBOutlet weak var mtkView: MTKView!
    var renderer: Renderer!

    override func viewDidLoad() {
        super.viewDidLoad()

        let device = MTLCreateSystemDefaultDevice()!
        renderer = Renderer(device: device, view: mtkView)
    }
}
