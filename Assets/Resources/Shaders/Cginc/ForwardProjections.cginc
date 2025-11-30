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

#ifdef DYNAMIC_LIGHTING_DYNAMIC_GEOMETRY_DISTANCE_CUBES
    #define DYNLIT_FRAGMENT_LIGHT void dynlit_frag_light(v2f_dynlit i, uint triangle_index, bool is_front_face, int bvhLightIndex, inout DynamicLight light, inout DynamicTriangle dynamic_triangle, DYNLIT_FRAGMENT_LIGHT_OUT_PARAMETERS)
#else
    #define DYNLIT_FRAGMENT_LIGHT void dynlit_frag_light(v2f_dynlit i, uint triangle_index, bool is_front_face, inout DynamicLight light, inout DynamicTriangle dynamic_triangle, DYNLIT_FRAGMENT_LIGHT_OUT_PARAMETERS)
#endif

DYNLIT_FRAGMENT_LIGHT
{
    #define GENERATE_NORMAL N
    #include "Packages/de.alpacait.dynamiclighting/AlpacaIT.DynamicLighting/Shaders/Generators/LightProcessor.cginc"
    
    UnityLight dLight;
    dLight.color = light.color * light.intensity * attenuation * map;
    dLight.dir = light_direction;
    dLight.ndotl = NdotL;
    
    UnityIndirect dIndirect = ZeroIndirect();
    
    half4 c = UNITY_BRDF_PBS(albedo, specColor, oneMinusReflectivity, oneMinusRoughness, N, V, dLight, dIndirect);
    
    Lo += c.rgb;
    
#if defined(DYNAMIC_LIGHTING_BOUNCE) && !defined(DYNAMIC_LIGHTING_INTEGRATED_GRAPHICS)
    float3 radiance = (light.color * light.intensity * attenuation);
    Lo += (albedo / UNITY_PI) * radiance * bounce;
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

//Helpers matching DynaLightToon logic
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

float Decal_GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float Decal_GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = Decal_GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = Decal_GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

float3 Decal_FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

half4 Custom_BRDF_ToonMetallic(half3 diffColor, half3 specColor, half oneMinusReflectivity, half oneMinusRoughness, half3 normal, half3 viewDir, UnityLight light, UnityIndirect gi, half metallic)
{
    float3 N = normal;
    float3 V = viewDir;
    float3 L = light.dir;
    
    float roughness = 1.0 - oneMinusRoughness;
    roughness = max(roughness, 0.02);

    float3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);

    // Specular (Cook-Torrance)
    float NDF = Decal_DistributionGGX(N, H, roughness);
    float G = Decal_GeometrySmith(N, V, L, roughness);
    float3 F0 = specColor; 
    float3 F = Decal_FresnelSchlick(max(dot(H, V), 0.0), F0);

    float3 numerator = NDF * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.0001;
    float3 specular = numerator / denominator;
    
    // Apply metallic scaling as per Toon shader
    specular *= metallic; 

    // Apply lighting
    // UnityLight.ndotl is N.L
    float3 radiance = light.color * light.ndotl; 
    
    // Direct Color
    float3 color = (diffColor + specular) * radiance;
    
    // Indirect Color (GI)
    // Standard Fresnel for environment
    float3 F_ind = Decal_FresnelSchlick(NdotV, F0); 
    float3 indirectDiffuse = gi.diffuse * diffColor;
    float3 indirectSpecular = gi.specular * F_ind;
    
    float3 indirect = indirectDiffuse + indirectSpecular;
    
    return half4(color + indirect, 1.0);
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

	//Calculate final output
	half4 c = Custom_BRDF_ToonMetallic(fragment.diffColor, fragment.specColor, fragment.oneMinusReflectivity, fragment.oneMinusRoughness, fragment.normalWorld, -fragment.eyeVec, gi.light, gi.indirect, fragment.metallic);
	//c.rgb += UNITY_BRDF_GI(fragment.diffColor, fragment.specColor, fragment.oneMinusReflectivity, fragment.oneMinusRoughness, fragment.normalWorld, -fragment.eyeVec, 1, gi);
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

	half4 c = Custom_BRDF_ToonMetallic(fragment.diffColor, fragment.specColor, fragment.oneMinusReflectivity, fragment.oneMinusRoughness, fragment.normalWorld, -fragment.eyeVec, light, noIndirect, fragment.metallic);

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