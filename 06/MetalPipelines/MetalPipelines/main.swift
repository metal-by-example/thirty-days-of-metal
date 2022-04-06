
import Metal

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

guard let library = device.makeDefaultLibrary() else {
    fatalError("Unable to create default shader library")
}

let kernelFunction = library.makeFunction(name: "add_two_values")!

let computePipeline = try device.makeComputePipelineState(function: kernelFunction)

let elementCount = 256
let inputBufferA = device.makeBuffer(length: MemoryLayout<Float>.stride * elementCount,
                                     options: .storageModeShared)!
let inputBufferB = device.makeBuffer(length: MemoryLayout<Float>.stride * elementCount,
                                     options: .storageModeShared)!
let outputBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * elementCount,
                                     options: .storageModeShared)!

let inputsA = inputBufferA.contents().assumingMemoryBound(to: Float.self)
let inputsB = inputBufferB.contents().assumingMemoryBound(to: Float.self)
for i in 0..<elementCount {
    inputsA[i] = Float(i)
    inputsB[i] = Float(elementCount - i)
}

let commandBuffer = commandQueue.makeCommandBuffer()!

let commandEncoder = commandBuffer.makeComputeCommandEncoder()!

commandEncoder.setComputePipelineState(computePipeline)

commandEncoder.setBuffer(inputBufferA, offset: 0, index: 0)
commandEncoder.setBuffer(inputBufferB, offset: 0, index: 1)
commandEncoder.setBuffer(outputBuffer, offset: 0, index: 2)

let threadsPerThreadgroup = MTLSize(width: 32, height: 1, depth: 1)
let threadgroupCount = MTLSize(width: 8, height: 1, depth: 1)
commandEncoder.dispatchThreadgroups(threadgroupCount,
                                    threadsPerThreadgroup: threadsPerThreadgroup)

commandEncoder.endEncoding()

commandBuffer.addCompletedHandler { _ in
    let outputs = outputBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<elementCount {
        print("Output element \(i) is \(outputs[i])")
    }
}

commandBuffer.commit()

// We don't always want to wait until the GPU finishes executing our commands,
// but in this case, the program can exit before the completion block is called,
// so we wait here to ensure the results actually print out.
commandBuffer.waitUntilCompleted()
