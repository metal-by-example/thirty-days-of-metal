import Metal

let device = MTLCreateSystemDefaultDevice()!

let sourceBuffer = device.makeBuffer(length: 16, options: [])!
let destBuffer = device.makeBuffer(length: 16, options: [])!

let points = sourceBuffer.contents().bindMemory(to: SIMD2<Float>.self,
                                                capacity: 2)
points[0] = SIMD2<Float>(10, 10)
points[1] = SIMD2<Float>(100, 100)

let commandQueue = device.makeCommandQueue()!

let commandBuffer = commandQueue.makeCommandBuffer()!

let blitCommandEncoder = commandBuffer.makeBlitCommandEncoder()!
blitCommandEncoder.copy(from: sourceBuffer,
                        sourceOffset: 0,
                        to: destBuffer,
                        destinationOffset: 0,
                        size: MemoryLayout<SIMD2<Float>>.stride * 2)
blitCommandEncoder.endEncoding()

commandBuffer.addCompletedHandler { completedCommandBuffer in
    let outPoints = destBuffer.contents().bindMemory(to: SIMD2<Float>.self,
                                                     capacity: 2)
    let p1 = outPoints[1]
    print("p1 in destination buffer is \(p1)")
}

commandBuffer.commit()
