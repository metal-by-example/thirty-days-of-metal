import Metal

let device = MTLCreateSystemDefaultDevice()!

let buffer = device.makeBuffer(length: 16, options: [])!

print("Buffer is \(buffer.length) bytes in length")

let contents = buffer.contents()

let points = buffer.contents().bindMemory(to: SIMD2<Float>.self,
                                          capacity: 2)
points[0] = SIMD2<Float>(10, 10)
points[1] = SIMD2<Float>(100, 100)

let p1 = points[1]
print("p1 is \(p1)")
