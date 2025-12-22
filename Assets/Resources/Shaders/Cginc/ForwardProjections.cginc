#include "Projections.cginc"

//UnityLight Helper (Moved up for scope visibility)
UnityLight MainLight(half3 normalWorld)
{
	UnityLight l;
#ifdef LIGHTMAP_OFF

	l.color = _LightColor0.rgb;
	l.dir = _WorldSpaceLightPos0.xyz;
	l.ndotl = LambertTerm(normalWorld, l.dir);
#else
	// no light specified by the engine
	// analytical light might be extracted from Lightmap data later on in the shader depending on the Lightmap type
	l.color = half3(0.f, 0.f, 0.f);
	l.ndotl = 0.f;
	l.dir = half3(0.f, 0.f, 0.f);
#endif

	return l;
}
UnityLight AdditiveLight(half3 normalWorld, half3 lightDir, half atten)
{
	UnityLight l;

	l.color = _LightColor0.rgb;
	l.dir = lightDir;
#ifndef USING_DIRECTIONAL_LIGHT
	l.dir = normalize(l.dir);
#endif
	l.ndotl = LambertTerm(normalWorld, l.dir);

	// shadow the light
	l.color *= atten;
	return l;
}
UnityIndirect ZeroIndirect()
{
	UnityIndirect ind;
	ind.diffuse = 0;
	ind.specular = 0;
	return ind;
}

// Oren-Nayar diffuse term
float Decal_OrenNayarTerm(float3 N, float3 L, float3 V, float roughness)
{
    float NdotL = saturate(dot(N, L));
    float NdotV = saturate(dot(N, V));
    if (NdotL <= 0.0 || NdotV <= 0.0)
    {
        return 0.0;
    }

    float sigma = saturate(roughness);
    float sigma2 = sigma * sigma;

    float A = 1.0 - (0.5 * sigma2 / (sigma2 + 0.33));
    float B = 0.45 * sigma2 / (sigma2 + 0.09);

    float sinThetaI = sqrt(1.0 - NdotL * NdotL);
    float sinThetaR = sqrt(1.0 - NdotV * NdotV);

    float maxCos = 0.0;
    if (sinThetaI > 1e-4 && sinThetaR > 1e-4)
    {
        float3 Lperp = normalize(L - N * NdotL);
        float3 Vperp = normalize(V - N * NdotV);
        maxCos = max(0.0, dot(Lperp, Vperp));
    }

    float sinAlpha;
    float tanBeta;
    if (NdotL > NdotV)
    {
        sinAlpha = sinThetaI;
        tanBeta = sinThetaR / max(NdotV, 1e-4);
    }
    else
    {
        sinAlpha = sinThetaR;
        tanBeta = sinThetaI / max(NdotL, 1e-4);
    }

    // clamp tanBeta to avoid large spikes near grazing angles
    tanBeta = min(tanBeta, 2.0);

    return NdotL * (A + B * maxCos * sinAlpha * tanBeta);
}

// Chan Diffuse BRDF (Call of Duty: WWII fit)
// Fixed halfLambertMix and chanEdgeSmoothness values for decals
static const float _Decal_HalfLambertMix = 0.7;
static const float _Decal_ChanEdgeSmoothness = 0.35;

float3 Decal_ChanDiffuse(float3 N, float3 L, float3 V, float3 H, float gloss, float3 albedo)
{
    float baseNdotL = saturate(dot(N, L));
    float edgeSmoothness = saturate(_Decal_ChanEdgeSmoothness);
    float edgeExponent = lerp(1.0, 0.35, edgeSmoothness);
    float NdotL = pow(baseNdotL, edgeExponent);
    float NdotV = saturate(dot(N, V));
    float LdotH = saturate(dot(L, H));
    float NdotH = saturate(dot(N, H));

    // 1. Rough Foundation
    float f0 = LdotH + pow(1.0 - LdotH, 5.0);

    // 2. Smooth Foundation (Disney-like)
    float f1 = (1.0 - 0.75 * pow(1.0 - NdotL, 5.0)) * (1.0 - 0.75 * pow(1.0 - NdotV, 5.0));

    // 3. Interpolation
    float t = saturate(2.2 * gloss - 0.5);

    // 4. Combine Base
    float fd = lerp(f0, f1, t);

    // 5. Retroreflective Bump (clamped to prevent fireflies)
    float fb_factor = min((34.5 * gloss * gloss) - (59.0 * gloss) + 24.5, 8.0);
    float fb_exp = max(73.2 * gloss - 21.2, 8.9) * sqrt(max(NdotH, 0.01));
    float fb = fb_factor * LdotH * exp2(-fb_exp);

    // Chan diffuse term
    float chanTerm = fd + fb;

    // Oren-Nayar term
    float roughness = saturate(1.0 - gloss);
    float orenTerm = Decal_OrenNayarTerm(N, L, V, roughness);

    // Mix Chan diffuse with Oren-Nayar (clamped to sane range)
    float mixedTerm = min(lerp(chanTerm, orenTerm, saturate(_Decal_HalfLambertMix)), 2.0);

    // Final BRDF
    return (albedo / UNITY_PI) * mixedTerm;
}

// GGX Distribution
float Decal_DistributionGGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = UNITY_PI * denom * denom;

    return nom / denom;
}

// Geometry Schlick GGX
float Decal_GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

// Geometry Smith
float Decal_GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = Decal_GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = Decal_GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// Fresnel Schlick
float3 Decal_FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

// Fresnel Schlick with roughness (for ambient reflections)
float3 Decal_FresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
{
    float smoothness = 1.0 - roughness;
    return F0 + (max(float3(smoothness, smoothness, smoothness), F0) - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

//Dynamic Lighting Integration
//Check if DynamicLighting is available via one of its internal macros or defines
#if defined(DYNLIT_FRAGMENT_INTERNAL)

struct v2f_dynlit {
    float3 world : TEXCOORD0;
    float2 uv1 : TEXCOORD1;
    float3 normal : NORMAL;
};

//Redefine the macro to match our custom struct and conditional arguments
#ifdef DYNLIT_FRAGMENT_LIGHT
    #undef DYNLIT_FRAGMENT_LIGHT
#endif

#define DYNLIT_FRAGMENT_LIGHT_OUT_PARAMETERS inout float3 albedo, inout float3 specColor, inout float oneMinusReflectivity, inout float oneMinusRoughness, inout float3 N, inout float3 V, inout float3 Lo
#define DYNLIT_FRAGMENT_LIGHT_IN_PARAMETERS albedo, specColor, oneMinusReflectivity, oneMinusRoughness, N, V, Lo

#if defined(DYNAMIC_LIGHTING_DYNAMIC_GEOMETRY_DISTANCE_CUBES) || defined(DYNAMIC_LIGHTING_DYNAMIC_GEOMETRY_ANGULAR)
    #define DYNLIT_FRAGMENT_LIGHT void dynlit_frag_light(v2f_dynlit i, uint triangle_index, bool is_front_face, int bvhLightIndex, inout DynamicLight light, inout DynamicTriangle dynamic_triangle, DYNLIT_FRAGMENT_LIGHT_OUT_PARAMETERS)
#else
    #define DYNLIT_FRAGMENT_LIGHT void dynlit_frag_light(v2f_dynlit i, uint triangle_index, bool is_front_face, inout DynamicLight light, inout DynamicTriangle dynamic_triangle, DYNLIT_FRAGMENT_LIGHT_OUT_PARAMETERS)
#endif

DYNLIT_FRAGMENT_LIGHT
{
    #define GENERATE_NORMAL N
    #include "Packages/de.alpacait.dynamiclighting/AlpacaIT.DynamicLighting/Shaders/Generators/LightProcessor.cginc"
    
    float3 L = light_direction;
    float3 H = normalize(V + L);
    float gloss = oneMinusRoughness;
    float roughnessLocal = saturate(1.0 - gloss);
    roughnessLocal = max(roughnessLocal, 0.15);
    
    // Calculate F0 based on metallic workflow
    // Note: For decals we derive metallic from specColor intensity
    float metallic = saturate(dot(specColor, float3(0.3333, 0.3333, 0.3333)) * 4.0);
    float3 F0 = lerp(0.04, albedo, metallic);
    
    // Radiance from dynamic light
#if defined(DYNAMIC_LIGHTING_BOUNCE) && !defined(DYNAMIC_LIGHTING_INTEGRATED_GRAPHICS)
    float3 radiance = (light.color * light.intensity * attenuation) + (light.bounceColor * light.intensity * attenuation * bounce);
#else
    float3 radiance = (light.color * light.intensity * attenuation);
#endif

    // Chan diffuse BRDF
    float3 chanDiffuse = Decal_ChanDiffuse(N, L, V, H, gloss, albedo);
    
    // GGX specular
    float NDF = Decal_DistributionGGX(N, H, roughnessLocal);
    float G = Decal_GeometrySmith(N, V, L, roughnessLocal);
    float3 F_light = Decal_FresnelSchlick(max(dot(H, V), 0.0), F0);
    
    // Energy conservation
    float3 kS_light = F_light;
    float3 kD_light = 1.0 - kS_light;
    kD_light *= 1.0 - metallic;
    
    // Cook-Torrance specular
    float3 numerator = NDF * G * F_light;
    float denominator = 4.0 * max(dot(N, V), 0.01) * max(NdotL, 0.01) + 0.01;
    float3 specularLighting = min(numerator / denominator, 4.0);
    
    // Diffuse uses Chan BRDF weighted by kD
    float3 diffuseLighting = chanDiffuse * kD_light;
    
    // Add to outgoing radiance Lo (clamped per-light contribution to prevent fireflies)
#if defined(DYNAMIC_LIGHTING_BOUNCE) && !defined(DYNAMIC_LIGHTING_INTEGRATED_GRAPHICS)
    Lo += min((diffuseLighting + specularLighting) * radiance * NdotL * map, 8.0);
    Lo += min((kD_light * albedo / UNITY_PI) * radiance * bounce, 4.0);
#else
    Lo += min((diffuseLighting + specularLighting) * radiance * NdotL * map, 8.0);
#endif
}

#endif // DYNLIT_FRAGMENT_INTERNAL

//Lighting
inline fixed LightAttenuation(float3 posWorld, float2 screenPos)
{
	fixed atten = 1;

	//Correct LightCoords per pixel	
#ifdef POINT
	float3 LightCoord = mul(unity_WorldToLight, float4(posWorld, 1)).xyz;
	atten = (tex2D(_LightTexture0, dot(LightCoord, LightCoord).rr).UNITY_ATTEN_CHANNEL);
#endif
#ifdef SPOT
	float4 LightCoord = mul(unity_WorldToLight, float4(posWorld, 1));
	atten = ((LightCoord.z > 0) * UnitySpotCookie(LightCoord) * UnitySpotAttenuate(LightCoord.xyz));
#endif
#ifdef DIRECTIONAL
	atten = 1;
#endif
#ifdef POINT_COOKIE
	float3 LightCoord = mul(unity_WorldToLight, float4(posWorld, 1)).xyz;
	atten = (tex2D(_LightTextureB0, dot(LightCoord, LightCoord).rr).UNITY_ATTEN_CHANNEL * texCUBE(_LightTexture0, LightCoord).w);
#endif
#ifdef DIRECTIONAL_COOKIE
	float2 LightCoord = mul(unity_WorldToLight, float4(posWorld, 1)).xy;
	atten = (tex2D(_LightTexture0, LightCoord).w);
#endif

	//Correct ShadowCoords per pixel
#if defined (SHADOWS_SCREEN)
#if defined(UNITY_NO_SCREENSPACE_SHADOWS)
	float4 ShadowCoord = mul(unity_WorldToShadow[0], float4(posWorld, 1));
	atten *= unitySampleShadow(ShadowCoord);
#else
	atten *= tex2D(_ShadowMapTexture, screenPos).r;
#endif
#endif
#if defined (SHADOWS_DEPTH) && defined (SPOT)
	//Spot
	float4 ShadowCoord = mul(unity_WorldToShadow[0], float4(posWorld, 1));
	atten *= UnitySampleShadowmap(ShadowCoord);
#endif
#if defined (SHADOWS_CUBE)
	//Point
	float3 ShadowCoord = posWorld - _LightPositionRange.xyz;
	atten *= UnitySampleShadowmap(ShadowCoord);
#endif

	return atten;
}
inline fixed ShadowAttenuation(float3 posWorld, float2 screenPos)
{
	fixed atten = 1;

	//Correct ShadowCoords per pixel
#if defined (SHADOWS_SCREEN)
#if defined(UNITY_NO_SCREENSPACE_SHADOWS)
	float4 ShadowCoord = mul(unity_WorldToShadow[0], float4(posWorld, 1));
	atten *= unitySampleShadow(ShadowCoord);
#else
	atten *= tex2D(_ShadowMapTexture, screenPos).r;
#endif
#endif
#if defined (SHADOWS_DEPTH) && defined (SPOT)
	//Spot
	float4 ShadowCoord = mul(unity_WorldToShadow[0], float4(posWorld, 1));
	atten *= UnitySampleShadowmap(ShadowCoord);
#endif
#if defined (SHADOWS_CUBE)
	//Point
	float3 ShadowCoord = posWorld - _LightPositionRange.xyz;
	atten *= UnitySampleShadowmap(ShadowCoord);
#endif

	return atten;
}
inline float3 LightDiriction(float3 posWorld)
{
	//Calculate LightDirection
	return _WorldSpaceLightPos0.xyz - posWorld * _WorldSpaceLightPos0.w;
}

//Vertex program - Projection
struct ProjectionInput
{
    float4 pos : SV_POSITION;
    float4 screenPos : TEXCOORD0;
    float3 ray : TEXCOORD1;

    half3 worldForward : TEXCOORD2;
    half3 worldUp : TEXCOORD3;

    half3 eyeVec : TEXCOORD4;

	UNITY_FOG_COORDS(5)
	UNITY_VERTEX_INPUT_INSTANCE_ID
};

ProjectionInput vertProjection(VertexInput v)
{
	ProjectionInput o;

	UNITY_SETUP_INSTANCE_ID (v);
    UNITY_TRANSFER_INSTANCE_ID (v, o);

	o.pos = UnityObjectToClipPos (v.vertex);
	o.screenPos = ComputeScreenPos(o.pos);
	o.ray = UnityObjectToViewPos(v.vertex) * float3(-1, -1, 1);

	float4 posWorld = mul(unity_ObjectToWorld, v.vertex);
	o.eyeVec = posWorld.xyz - _WorldSpaceCameraPos;

	o.worldForward = mul((float3x3)unity_ObjectToWorld, float3(0, 0, 1));
	o.worldUp = mul((float3x3)unity_ObjectToWorld, float3(1, 0, 0)); //(Now Right)

	UNITY_TRANSFER_FOG(o, o.pos);
	return o;
}

//Vertex program - OmniDecal
struct OmniDecalInput
{
	float4 pos : SV_POSITION;
	float4 screenPos : TEXCOORD0;
	float3 ray : TEXCOORD1;
	UNITY_FOG_COORDS(2)
	UNITY_VERTEX_INPUT_INSTANCE_ID
};
OmniDecalInput vertOmniDecal(VertexInput v)
{
	OmniDecalInput o;

	UNITY_SETUP_INSTANCE_ID (v);
    UNITY_TRANSFER_INSTANCE_ID (v, o);

	o.pos = UnityObjectToClipPos (v.vertex);
	o.screenPos = ComputeScreenPos(o.pos);
	o.ray = UnityObjectToViewPos(v.vertex) * float3(-1, -1, 1);

	UNITY_TRANSFER_FOG(o, o.pos);
	return o;
}

//Output
half4 Output(half3 color, half occlusion)
{
	#ifdef _AlphaTest
	return half4(color, 1);
	#else
	return half4(color, occlusion);
	#endif
}


half4 Custom_BRDF_ToonMetallic(half3 diffColor, half3 specColor, half oneMinusReflectivity, half oneMinusRoughness, half3 normal, half3 viewDir, UnityLight light, UnityIndirect gi, half metallic, float3 posWorld)
{
    float3 N = normal;
    float3 V = viewDir;
    float3 L = light.dir;
    
    float roughness = 1.0 - oneMinusRoughness;
    float gloss = oneMinusRoughness;
    roughness = max(roughness, 0.15); // minimum roughness prevents fireflies

    float3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);

    // Calculate F0 based on metallic workflow
    float3 F0 = lerp(0.04, diffColor, metallic);

    //-------------------------------------------------------------------
    // Direct Lighting: Cook-Torrance BRDF with Chan Diffuse
    //-------------------------------------------------------------------
    
    // Chan diffuse BRDF
    float3 chanDiffuse = Decal_ChanDiffuse(N, L, V, H, gloss, diffColor);
    
    // Specular: Cook-Torrance
    float roughnessLocal = max(roughness, 0.15);
    float NDF = Decal_DistributionGGX(N, H, roughnessLocal);
    float G = Decal_GeometrySmith(N, V, L, roughnessLocal);
    float3 F_light = Decal_FresnelSchlick(max(dot(H, V), 0.0), F0);
    
    // Energy conservation
    float3 kS_light = F_light;
    float3 kD_light = 1.0 - kS_light;
    kD_light *= 1.0 - metallic; // Pure metals have no diffuse
    
    // Cook-Torrance specular
    float3 numerator = NDF * G * F_light;
    float denominator = 4.0 * max(NdotV, 0.01) * max(NdotL, 0.01) + 0.01;
    float3 specular = min(numerator / denominator, 4.0); // Aggressive clamp
    
    // Direct lighting contribution
    float3 radiance = light.color * NdotL;
    float3 directColor = (kD_light * chanDiffuse + specular) * radiance;

    //-------------------------------------------------------------------
    // Indirect Lighting: Reflection probes with box projection/blending
    //-------------------------------------------------------------------
    
    // Fresnel for ambient with roughness consideration
    float3 F = Decal_FresnelSchlickRoughness(max(NdotV, 0.0), F0, roughness);
    
    float3 kS = F;
    float3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;
    
    // Reflection direction
    float3 reflection = reflect(-V, N);

    // Sample primary reflection probe with box projection
    float3 refl0 = reflection;
    #ifdef UNITY_SPECCUBE_BOX_PROJECTION
        refl0 = BoxProjectedCubemapDirection(refl0, posWorld, unity_SpecCube0_ProbePosition, unity_SpecCube0_BoxMin, unity_SpecCube0_BoxMax);
    #endif
    float3 skyColor = DecodeHDR(UNITY_SAMPLE_TEXCUBE_LOD(unity_SpecCube0, refl0, roughness * 4.0), unity_SpecCube0_HDR);

    // Sample secondary reflection probe and blend
    #ifdef UNITY_SPECCUBE_BLENDING
        float blendLerp = unity_SpecCube0_BoxMin.w;
        if (blendLerp < 0.99999)
        {
            float3 refl1 = reflection;
            #ifdef UNITY_SPECCUBE_BOX_PROJECTION
                refl1 = BoxProjectedCubemapDirection(refl1, posWorld, unity_SpecCube1_ProbePosition, unity_SpecCube1_BoxMin, unity_SpecCube1_BoxMax);
            #endif
            
            float3 skyColor1 = DecodeHDR(UNITY_SAMPLE_TEXCUBE_SAMPLER_LOD(unity_SpecCube1, unity_SpecCube0, refl1, roughness * 4.0), unity_SpecCube1_HDR);
            skyColor = lerp(skyColor1, skyColor, blendLerp);
        }
    #endif

    float3 specularReflection = skyColor * F;
    
    // Ambient diffuse
    float3 ambient = kD * diffColor * gi.diffuse;
    
    // Final indirect
    float3 indirectColor = ambient + specularReflection;
    
    return half4(directColor + indirectColor, 1.0);
}

//Metallic Programs
half4 fragForwardMetallic(ProjectionInput i) : SV_Target
{
	//Setup Instance Data
	UNITY_SETUP_INSTANCE_ID (i);

	//Generate projection
	Projection projection = CalculateProjection(i.screenPos, i.ray, i.worldForward);

	//Generate base data
	FragmentCommonData fragment = FragmentMetallic(projection, i.worldUp, i.eyeVec);

	//Setup Light
	UnityLight mainLight = MainLight(fragment.normalWorld);
	half atten = ShadowAttenuation(projection.posWorld, projection.screenPos);

	//Setup GI
	UnityGI gi = FragmentGI(fragment, 1, half4(0,0,0,0), atten, mainLight, true);

	// Material parameters
	float3 N = fragment.normalWorld;
	float3 V = -fragment.eyeVec;
	float roughness = 1.0 - fragment.oneMinusRoughness;
	float gloss = fragment.oneMinusRoughness;
	roughness = max(roughness, 0.15);
	float metallic = fragment.metallic;
	float3 albedo = fragment.diffColor;
	
	// Calculate F0 based on metallic workflow
	// For metals (metallic=1), F0 should use albedo color but we need a minimum reflectance
	// Real metals always have some reflectance (typically 0.5-0.9)
	// We use the albedo for metallic tint but ensure minimum F0 of 0.04
	float3 baseF0 = lerp(0.04, albedo, metallic);
	// Ensure minimum reflectance for metallic surfaces (avoids black reflections with dark albedo)
	float3 F0 = lerp(baseF0, max(baseF0, 0.04), metallic);
	float NdotV = max(dot(N, V), 0.0);
	
	//-------------------------------------------------------------------
	// Sample Reflection Probes
	// Use the underlying surface normal for reflections, not the decal's normal map
	//-------------------------------------------------------------------
	float3 surfaceNormal = normalize(projection.normal);
	float3 reflection = reflect(-V, surfaceNormal);
	
	// Sample primary reflection probe with box projection
	float3 refl0 = reflection;
	#ifdef UNITY_SPECCUBE_BOX_PROJECTION
		refl0 = BoxProjectedCubemapDirection(refl0, projection.posWorld, unity_SpecCube0_ProbePosition, unity_SpecCube0_BoxMin, unity_SpecCube0_BoxMax);
	#endif
	float3 skyColor = DecodeHDR(UNITY_SAMPLE_TEXCUBE_LOD(unity_SpecCube0, refl0, roughness * 4.0), unity_SpecCube0_HDR);

	// Sample secondary reflection probe and blend
	#ifdef UNITY_SPECCUBE_BLENDING
		float blendLerp = unity_SpecCube0_BoxMin.w;
		if (blendLerp < 0.99999)
		{
			float3 refl1 = reflection;
			#ifdef UNITY_SPECCUBE_BOX_PROJECTION
				refl1 = BoxProjectedCubemapDirection(refl1, projection.posWorld, unity_SpecCube1_ProbePosition, unity_SpecCube1_BoxMin, unity_SpecCube1_BoxMax);
			#endif
			
			float3 skyColor1 = DecodeHDR(UNITY_SAMPLE_TEXCUBE_SAMPLER_LOD(unity_SpecCube1, unity_SpecCube0, refl1, roughness * 4.0), unity_SpecCube1_HDR);
			skyColor = lerp(skyColor1, skyColor, blendLerp);
		}
	#endif
	
	// Fresnel for reflections with roughness consideration
	// Use surface normal NdotV for reflection fresnel (matches the reflection direction)
	float NdotV_surface = max(dot(surfaceNormal, V), 0.0);
	// Use native DynamicLighting function when available (matches DynamicLightingMetallic shader)
	#if defined(DYNAMIC_LIGHTING_LIT)
	float3 F = fresnelSchlickRoughness(NdotV_surface, F0, roughness);
	#else
	float3 F = Decal_FresnelSchlickRoughness(NdotV_surface, F0, roughness);
	#endif
	float3 specularReflection = skyColor * F;
	
	//-------------------------------------------------------------------
	// Calculate ambient and direct lighting (will be scaled by dynamic_ambient_color)
	//-------------------------------------------------------------------
	float3 kS = F;
	float3 kD = 1.0 - kS;
	kD *= 1.0 - metallic;
	
	// Start with ambient
	float3 ambient = kD * albedo * gi.indirect.diffuse;
	
	// Direct lighting from main light
	float3 L = gi.light.dir;
	float3 H = normalize(V + L);
	float NdotL = max(dot(N, L), 0.0);
	
	float3 chanDiffuse = Decal_ChanDiffuse(N, L, V, H, gloss, albedo);
	
	// Cook-Torrance specular for direct light
	float roughnessLocal = max(roughness, 0.15);
	float NDF = Decal_DistributionGGX(N, H, roughnessLocal);
	float G = Decal_GeometrySmith(N, V, L, roughnessLocal);
	float3 F_light = Decal_FresnelSchlick(max(dot(H, V), 0.0), F0);
	
	float3 kS_light = F_light;
	float3 kD_light = 1.0 - kS_light;
	kD_light *= 1.0 - metallic;
	
	float3 numerator = NDF * G * F_light;
	float denominator = 4.0 * max(NdotV, 0.01) * max(NdotL, 0.01) + 0.01;
	float3 directSpecular = min(numerator / denominator, 4.0);
	
	float3 radiance = gi.light.color * NdotL;
	float3 directColor = (kD_light * chanDiffuse + directSpecular) * radiance;
	
	// Combine ambient + direct (this gets scaled by dynamic_ambient_color)
	float3 Lo = float3(0,0,0);
	
	//Apply dynamic lighting
	#if defined(DYNAMIC_LIGHTING_LIT)
    {
        v2f_dynlit i_dyn;
        i_dyn.world = projection.posWorld;
        i_dyn.uv1 = float2(0,0);
        i_dyn.normal = fragment.normalWorld;

        #define i i_dyn
        uint triangle_index = 0;
        bool is_front_face = true;
        
        float3 dyn_albedo = fragment.diffColor;
        float3 dyn_specColor = fragment.specColor;
        float dyn_oneMinusReflectivity = fragment.oneMinusReflectivity;
        float dyn_oneMinusRoughness = fragment.oneMinusRoughness;
        float3 dyn_N = fragment.normalWorld;
        float3 dyn_V = -fragment.eyeVec;
        float3 dyn_Lo = float3(0,0,0);
        
        #undef albedo
        #undef specColor
        #undef oneMinusReflectivity
        #undef oneMinusRoughness
        #undef N
        #undef V
        #define albedo dyn_albedo
        #define specColor dyn_specColor
        #define oneMinusReflectivity dyn_oneMinusReflectivity
        #define oneMinusRoughness dyn_oneMinusRoughness
        #define N dyn_N
        #define V dyn_V
        #define Lo dyn_Lo
        
        DYNLIT_FRAGMENT_INTERNAL
        
        #undef albedo
        #undef specColor
        #undef oneMinusReflectivity
        #undef oneMinusRoughness
        #undef N
        #undef V
        #undef Lo
        #undef i
        
        Lo = dyn_Lo;
    }
    

    float3 color = (kD * albedo * dynamic_ambient_color + Lo) * fragment.occlusion + specularReflection;
	#else
	// Without dynamic lighting, use standard ambient + direct + reflections
	float3 color = (ambient + directColor) * fragment.occlusion + specularReflection;
	#endif
	
	color += EmissionAlpha(projection.localUV);

	UNITY_APPLY_FOG(i.fogCoord, color);
	return Output(color, fragment.occlusion);
}
half4 fragForwardAddMetallic(ProjectionInput i) : SV_Target
{	
	//Generate projection
	Projection projection = CalculateProjection(i.screenPos, i.ray, i.worldForward);

	//Generate base data
	FragmentCommonData fragment = FragmentMetallic(projection, i.worldUp, i.eyeVec);

	//Calculate lighting data
	float atten = LightAttenuation(projection.posWorld, projection.screenPos);
	float3 lightDir = LightDiriction(projection.posWorld);

	//Setup Light
	UnityLight light = AdditiveLight(fragment.normalWorld, lightDir, atten);
	UnityIndirect noIndirect = ZeroIndirect();

	half4 c = Custom_BRDF_ToonMetallic(fragment.diffColor, fragment.specColor, fragment.oneMinusReflectivity, fragment.oneMinusRoughness, fragment.normalWorld, -fragment.eyeVec, light, noIndirect, fragment.metallic, projection.posWorld);

	//Apply dynamic lighting ambient
	#if defined(DYNAMIC_LIGHTING_LIT)
	c.rgb *= dynamic_ambient_color;
	#endif

	UNITY_APPLY_FOG_COLOR(i.fogCoord, c.rgb, half4(0.0, 0.0, 0.0, 0.0));
	return Output(c.rgb, fragment.occlusion);
}

//Specular Programs
half4 fragForwardSpecular(ProjectionInput i) : SV_Target
{
	//Setup Instance Data
	UNITY_SETUP_INSTANCE_ID (i);

	//Generate projection
	Projection projection = CalculateProjection(i.screenPos, i.ray, i.worldForward);

	//Generate base data
	FragmentCommonData fragment = FragmentSpecular(projection, i.worldUp, i.eyeVec);

	//Setup Light
	UnityLight mainLight = MainLight(fragment.normalWorld);
	half atten = ShadowAttenuation(projection.posWorld, projection.screenPos);

	//Setup GI
	UnityGI gi = FragmentGI(fragment, 1, half4(0,0,0,0), atten, mainLight, true);

	//Calculate final output
	half4 c = UNITY_BRDF_PBS(fragment.diffColor, fragment.specColor, fragment.oneMinusReflectivity, fragment.oneMinusRoughness, fragment.normalWorld, -fragment.eyeVec, gi.light, gi.indirect);
	c.rgb += UNITY_BRDF_GI(fragment.diffColor, fragment.specColor, fragment.oneMinusReflectivity, fragment.oneMinusRoughness, fragment.normalWorld, -fragment.eyeVec, 1, gi);
	c.rgb += EmissionAlpha(projection.localUV);

	//Apply dynamic lighting ambient
	#if defined(DYNAMIC_LIGHTING_LIT)
	c.rgb *= dynamic_ambient_color;
    {
        v2f_dynlit i_dyn;
        i_dyn.world = projection.posWorld;
        i_dyn.uv1 = float2(0,0);
        i_dyn.normal = fragment.normalWorld;

        #define i i_dyn
        uint triangle_index = 0;
        bool is_front_face = true;
        
        float3 albedo = fragment.diffColor;
        float3 specColor = fragment.specColor;
        float oneMinusReflectivity = fragment.oneMinusReflectivity;
        float oneMinusRoughness = fragment.oneMinusRoughness;
        float3 N = fragment.normalWorld;
        float3 V = -fragment.eyeVec;
        float3 Lo = float3(0,0,0);
        
        DYNLIT_FRAGMENT_INTERNAL
        
        c.rgb += Lo;
        #undef i
    }
	#endif

	UNITY_APPLY_FOG(i.fogCoord, c.rgb);
	return Output(c.rgb, fragment.occlusion);
}
half4 fragForwardAddSpecular(ProjectionInput i) : SV_Target
{
	//Generate projection
	Projection projection = CalculateProjection(i.screenPos, i.ray, i.worldForward);

	//Generate base data
	FragmentCommonData fragment = FragmentSpecular(projection, i.worldUp, i.eyeVec);

	//Calculate lighting data
	float atten = LightAttenuation(projection.posWorld, projection.screenPos);
	float3 lightDir = LightDiriction(projection.posWorld);

	//Setup Light
	UnityLight light = AdditiveLight(fragment.normalWorld, lightDir, atten);
	UnityIndirect noIndirect = ZeroIndirect();

	half4 c = UNITY_BRDF_PBS(fragment.diffColor, fragment.specColor, fragment.oneMinusReflectivity, fragment.oneMinusRoughness, fragment.normalWorld, -fragment.eyeVec, light, noIndirect);

	//Apply dynamic lighting ambient
	#if defined(DYNAMIC_LIGHTING_LIT)
	c.rgb *= dynamic_ambient_color;
	#endif

	UNITY_APPLY_FOG_COLOR(i.fogCoord, c.rgb, half4(0, 0, 0, 0));
	return Output(c.rgb, fragment.occlusion);
}