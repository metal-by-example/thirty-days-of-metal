
import Metal

let device = MTLCreateSystemDefaultDevice()!

guard let library = device.makeDefaultLibrary() else {
    fatalError("Unable to create default shader library")
}

for name in library.functionNames {
    let function = library.makeFunction(name: name)!
    print("\(function)")
}
