
import Metal
import MetalKit

extension SIMD4 where Self.Scalar == Float {
    init(color: CGColor, in colorSpace: CGColorSpace) {
        if let matchedColor = color.converted(to: colorSpace, intent: .defaultIntent, options: nil),
           let components = matchedColor.components
        {
            self.init(components.map { Scalar($0) })
        } else {
            self.init(repeating: 1.0)
        }
    }
}

class Material {
    var baseColorTexture: MTLTexture?
    var baseColor = SIMD4<Float>(1, 1, 1, 1)

    var emissiveTexture: MTLTexture?
    var emissiveColor  = SIMD4<Float>(0, 0, 0, 1)

    var metalnessTexture: MTLTexture?
    var metalnessFactor: Float = 0.0

    var roughnessTexture: MTLTexture?
    var roughnessFactor: Float = 0.5

    var normalTexture: MTLTexture?
    var occlusionTexture: MTLTexture?

    var opacity: Float = 1.0

    init(_ mdlMaterial: MDLMaterial, device: MTLDevice) {
        let colorSpace = CGColorSpace(name: CGColorSpace.linearSRGB)!

        let textureLoaderOptions: [MTKTextureLoader.Option : Any] = [
            .origin : MTKTextureLoader.Origin.flippedVertically,
            .generateMipmaps : true
        ]
        let textureLoader = MTKTextureLoader(device: device)

        if let baseColorProperty = mdlMaterial.property(with: MDLMaterialSemantic.baseColor) {
            switch baseColorProperty.type {
            case .float3:
                baseColor = SIMD4<Float>(baseColorProperty.float3Value, 1)
            case .float4:
                baseColor = baseColorProperty.float4Value
            case .color:
                let color = baseColorProperty.color ?? CGColor.white
                baseColor = SIMD4<Float>(color: color, in: colorSpace)
            case .texture:
                if let mdlTexture = baseColorProperty.textureSamplerValue?.texture {
                    baseColorTexture = try? textureLoader.newTexture(texture: mdlTexture, options: textureLoaderOptions)
                }
            default:
                break
            }
        }
        if let opacityProperty = mdlMaterial.property(with: MDLMaterialSemantic.opacity) {
            switch opacityProperty.type {
            case .float:
                opacity = opacityProperty.floatValue
            default:
                break
            }
        }
        if let emissiveProperty = mdlMaterial.property(with: MDLMaterialSemantic.emission) {
            switch emissiveProperty.type {
            case .float3:
                emissiveColor = SIMD4<Float>(emissiveProperty.float3Value, 1)
            case .float4:
                emissiveColor = emissiveProperty.float4Value
            case .color:
                let color = emissiveProperty.color ?? CGColor.white
                emissiveColor = SIMD4<Float>(color: color, in: colorSpace)
            case .texture:
                if let mdlTexture = emissiveProperty.textureSamplerValue?.texture {
                    emissiveTexture = try? textureLoader.newTexture(texture: mdlTexture, options: textureLoaderOptions)
                }
            default:
                break
            }
        }
        if let normalProperty = mdlMaterial.property(with: MDLMaterialSemantic.tangentSpaceNormal) {
            switch normalProperty.type {
            case .texture:
                if let mdlTexture = normalProperty.textureSamplerValue?.texture {
                    normalTexture = try? textureLoader.newTexture(texture: mdlTexture, options: textureLoaderOptions)
                }
            default:
                break
            }
        }
        if let roughnessProperty = mdlMaterial.property(with: MDLMaterialSemantic.roughness) {
            switch roughnessProperty.type {
            case .float:
                roughnessFactor = roughnessProperty.floatValue
            case .texture:
                if let mdlTexture = roughnessProperty.textureSamplerValue?.texture {
                    roughnessTexture = try? textureLoader.newTexture(texture: mdlTexture, options: textureLoaderOptions)
                }
                roughnessFactor = 1.0
            default:
                break
            }
        }
        if let metalnessProperty = mdlMaterial.property(with: MDLMaterialSemantic.metallic) {
            switch metalnessProperty.type {
            case .float:
                metalnessFactor = metalnessProperty.floatValue
            case .texture:
                if let mdlTexture = metalnessProperty.textureSamplerValue?.texture {
                    metalnessTexture = try? textureLoader.newTexture(texture: mdlTexture, options: textureLoaderOptions)
                }
                metalnessFactor = 1.0
            default:
                break
            }
        }
        if let occlusionProperty = mdlMaterial.property(with: MDLMaterialSemantic.ambientOcclusion) {
            if occlusionProperty.type == .texture {
                if let mdlTexture = occlusionProperty.textureSamplerValue?.texture {
                    occlusionTexture = try? textureLoader.newTexture(texture: mdlTexture, options: textureLoaderOptions)
                }
            }
        }
    }
}
