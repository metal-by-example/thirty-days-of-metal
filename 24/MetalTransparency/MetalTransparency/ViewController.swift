
#if os(macOS)
import Cocoa
typealias BaseViewController = NSViewController
#elseif os(iOS)
import UIKit
typealias BaseViewController = UIViewController
#endif

import MetalKit
import GameController

enum VirtualKey: Int {
    case ANSI_A     = 0x00
    case ANSI_S     = 0x01
    case ANSI_D     = 0x02
    case ANSI_W     = 0x0D
    case leftArrow  = 0x7B
    case rightArrow = 0x7C
    case downArrow  = 0x7D
    case upArrow    = 0x7E
}

class ViewController: BaseViewController {

    @IBOutlet weak var mtkView: MTKView!
    var renderer: Renderer!

    var cameraController: FlyCameraController!

    var gameController: GCController?
    var virtualController: Any?

    var keysPressed = [Bool](repeating: false, count: Int(UInt16.max))
    var previousMousePoint = CGPoint.zero
    var currentMousePoint = CGPoint.zero

    private var observers = [Any]()

    deinit {
        for observer in observers {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        let device = mtkView.preferredDevice ?? MTLCreateSystemDefaultDevice()!
        renderer = Renderer(device: device, view: mtkView)

        cameraController = FlyCameraController(pointOfView: renderer.pointOfView)
        cameraController.eye = SIMD3<Float>(0, 0, 8)

        let frameDuration = 1.0 / Double(mtkView.preferredFramesPerSecond)
        Timer.scheduledTimer(withTimeInterval: frameDuration, repeats: true) { [weak self] _ in
            self?.updateCamera(Float(frameDuration))
        }

        registerControllerObservers()

#if os(iOS)
        if #available(iOS 15.0, *) {
            connectVirtualController()
        }
#endif
    }

#if os(macOS)
    override func viewDidAppear() {
        view.window?.makeFirstResponder(self)
    }
#endif

    override func becomeFirstResponder() -> Bool {
        return true
    }

    func updateCamera(_ timestep: Float) {
        if let gamepad = gameController?.extendedGamepad {
            let lookX = gamepad.rightThumbstick.xAxis.value
            let lookZ = gamepad.rightThumbstick.yAxis.value
            let lookDelta = SIMD2<Float>(lookX, lookZ)

            let moveZ = gamepad.leftThumbstick.yAxis.value
            let moveDelta = SIMD2<Float>(0, moveZ)

            cameraController.update(timestep: timestep,
                                    lookDelta: lookDelta,
                                    moveDelta: moveDelta)
        } else {
            let cursorDeltaX = Float(currentMousePoint.x - previousMousePoint.x)
            let cursorDeltaY = Float(currentMousePoint.y - previousMousePoint.y)
            previousMousePoint = currentMousePoint

            let forwardPressed = keysPressed[VirtualKey.ANSI_W.rawValue]
            let backwardPressed = keysPressed[VirtualKey.ANSI_S.rawValue]
            let leftPressed = keysPressed[VirtualKey.ANSI_A.rawValue]
            let rightPressed = keysPressed[VirtualKey.ANSI_D.rawValue]

            let deltaX: Float = (leftPressed ? -1.0 : 0.0) + (rightPressed ? 1.0 : 0.0)
            let deltaZ: Float = (backwardPressed ? -1.0 : 0.0) + (forwardPressed ? 1.0 : 0.0)

            let mouseDelta = SIMD2<Float>(cursorDeltaX, cursorDeltaY)
            let keyDelta = SIMD2<Float>(deltaX, deltaZ)
            cameraController.update(timestep: timestep,
                                    lookDelta: mouseDelta,
                                    moveDelta: keyDelta)
        }
    }

    //MARK: - Keyboard and Mouse Input

#if os(macOS)
    override func mouseDown(with event: NSEvent) {
        let mouseLocation = self.view.convert(event.locationInWindow, from: nil)
        currentMousePoint = mouseLocation
        previousMousePoint = mouseLocation
    }

    override func mouseDragged(with event: NSEvent) {
        let mouseLocation = self.view.convert(event.locationInWindow, from: nil)
         previousMousePoint = currentMousePoint
         currentMousePoint = mouseLocation
    }

    override func mouseUp(with event: NSEvent) {
        let mouseLocation = self.view.convert(event.locationInWindow, from: nil)
        previousMousePoint = mouseLocation
        currentMousePoint = mouseLocation
    }

    override func keyDown(with event: NSEvent) {
        keysPressed[Int(event.keyCode)] = true
    }

    override func keyUp(with event: NSEvent) {
        keysPressed[Int(event.keyCode)] = false
    }
#endif

    //MARK: - Game Controller Support

    private func registerControllerObservers() {
        let connectionObserver = NotificationCenter.default.addObserver(
            forName: NSNotification.Name.GCControllerDidConnect,
            object: nil,
            queue: nil)
        { [weak self] notification in
            if let controller = notification.object as? GCController {
                self?.controllerDidConnect(controller)
            }
        }

        let disconnectionObserver = NotificationCenter.default.addObserver(
            forName: NSNotification.Name.GCControllerDidDisconnect,
            object: nil,
            queue: nil)
        { [weak self] notification in
            if let controller = notification.object as? GCController {
                self?.controllerDidDisconnect(controller)
            }
        }

        observers = [connectionObserver, disconnectionObserver]
    }

    private func controllerDidConnect(_ controller: GCController) {
        gameController = controller
    }

    private func controllerDidDisconnect(_ controller: GCController) {
        gameController = nil
    }

#if os(iOS)
    @available(iOS 15.0, *)
    private func connectVirtualController() {
        let controllerConfig = GCVirtualController.Configuration()
        controllerConfig.elements = [
            GCInputLeftThumbstick,
            GCInputRightThumbstick,
        ]
        let controller = GCVirtualController(configuration: controllerConfig)
        controller.connect()
        virtualController = controller
    }

    @available(iOS 15.0, *)
    private func disconnectVirtualController() {
        (virtualController as? GCVirtualController)?.disconnect()
        virtualController = nil
    }
#endif
}
