#include "ForwardProjections.cginc"

#if DYNAMIC_LIGHTING_LIT

// Define the lighting parameters for the DYNLIT macros
#define DYNLIT_FRAGMENT_LIGHT_OUT_PARAMETERS inout half3 albedo, inout half3 N, inout half3 V, inout half3 Lo
#define DYNLIT_FRAGMENT_LIGHT_IN_PARAMETERS albedo, N, V, Lo

// Per-light processing macro (this will be called for each dynamic light)
DYNLIT_FRAGMENT_LIGHT
{
	// Generate light data (this macro provides: light, light_direction, light_distanceSqr, NdotL, attenuation, map, etc.)
	#define GENERATE_NORMAL N
	#include "Packages/de.alpacait.dynamiclighting/AlpacaIT.DynamicLighting/Shaders/Generators/LightProcessor.cginc"
	
	// Calculate per-light radiance
	#if defined(DYNAMIC_LIGHTING_BOUNCE) && !defined(DYNAMIC_LIGHTING_INTEGRATED_GRAPHICS)
	half3 radiance = (light.color * light.intensity * attenuation) + (light.bounceColor * light.intensity * attenuation * bounce);
	#else
	half3 radiance = (light.color * light.intensity * attenuation);
	#endif
	
	// Add diffuse contribution
	Lo += (albedo / UNITY_PI) * radiance * map * saturate(NdotL);
	
	#if defined(DYNAMIC_LIGHTING_BOUNCE) && !defined(DYNAMIC_LIGHTING_INTEGRATED_GRAPHICS)
	Lo += (albedo / UNITY_PI) * radiance * bounce;
	#endif
}

half4 fragAdd(ProjectionInput i) : SV_Target
{
	//Setup Instance Data
	UNITY_SETUP_INSTANCE_ID(i);

	//Generate projection
	Projection projection = CalculateProjection(i.screenPos, i.ray, i.worldForward);

	//Generate base data
	FragmentCommonData fragment = FragmentUnlit(projection, i.worldUp, i.eyeVec);

	//Setup lighting parameters
	half3 albedo = fragment.diffColor;
	half3 N = fragment.normalWorld;
	half3 V = fragment.eyeVec;
	half3 Lo = half3(0.0, 0.0, 0.0);
	
	// Calculate dynamic lighting using the DYNLIT macros
	{
		v2f_dynlit i_dyn;
		i_dyn.world = fragment.posWorld;
		i_dyn.uv1 = float2(0,0);
		i_dyn.normal = fragment.normalWorld;
		
		// Shadow 'i' with our v2f struct for the macro
		#define i i_dyn
		
		// Define triangle_index
		uint triangle_index = 0;
		
		// Define is_front_face (assume front facing for projected decals)
		bool is_front_face = true;
		
		DYNLIT_FRAGMENT_INTERNAL
        
        #undef i
	}
	
	//Combine ambient and dynamic lighting
	half3 ambient = albedo * dynamic_ambient_color;
	half3 c = ambient + Lo;
	
	//Apply occlusion
	c *= fragment.occlusion;
	
	//Apply fog
	UNITY_APPLY_FOG(i.fogCoord, c);
	
	return Output(c, 1);
}

half4 fragMultiply(ProjectionInput i) : SV_Target
{
	//Setup Instance Data
	UNITY_SETUP_INSTANCE_ID(i);

	//Generate projection
	Projection projection = CalculateProjection(i.screenPos, i.ray, i.worldForward);

	//Generate base data
	FragmentCommonData fragment = FragmentUnlit(projection, i.worldUp, i.eyeVec);

	//Setup lighting parameters
	half3 albedo = fragment.diffColor;
	half3 N = fragment.normalWorld;
	half3 V = fragment.eyeVec;
	half3 Lo = half3(0.0, 0.0, 0.0);
	
	// Calculate dynamic lighting using the DYNLIT macros
	{
		v2f_dynlit i_dyn;
		i_dyn.world = fragment.posWorld;
		i_dyn.uv1 = float2(0,0);
		i_dyn.normal = fragment.normalWorld;
		
		// Shadow 'i' with our v2f struct for the macro
		#define i i_dyn
		
		// Define triangle_index
		uint triangle_index = 0;
		
		// Define is_front_face
		bool is_front_face = true;
		
		DYNLIT_FRAGMENT_INTERNAL
        
        #undef i
	}
	
	//Combine ambient and dynamic lighting
	half3 ambient = albedo * dynamic_ambient_color;
	half3 c = ambient + Lo;
	
	//Apply occlusion (multiplicative blend needs special handling)
	c = lerp(half3(1,1,1), c, fragment.occlusion);
	
	//Apply fog
	UNITY_APPLY_FOG(i.fogCoord, c);
	
	return Output(c, 1);
}

half4 fragUnlit(ProjectionInput i) : SV_Target
{
	//Setup Instance Data
	UNITY_SETUP_INSTANCE_ID(i);

	//Generate projection
	Projection projection = CalculateProjection(i.screenPos, i.ray, i.worldForward);

	//Generate base data
	FragmentCommonData fragment = FragmentUnlit(projection, i.worldUp, i.eyeVec);

	//Setup lighting parameters
	half3 albedo = fragment.diffColor;
	half3 N = fragment.normalWorld;
	half3 V = fragment.eyeVec;
	half3 Lo = half3(0.0, 0.0, 0.0);
	
	// Calculate dynamic lighting using the DYNLIT macros
	{
		v2f_dynlit i_dyn;
		i_dyn.world = fragment.posWorld;
		i_dyn.uv1 = float2(0,0);
		i_dyn.normal = fragment.normalWorld;
		
		// Shadow 'i' with our v2f struct for the macro
		#define i i_dyn
		
		// Define triangle_index
		uint triangle_index = 0;
		
		// Define is_front_face
		bool is_front_face = true;
		
		DYNLIT_FRAGMENT_INTERNAL
        
        #undef i
	}
	
	//Combine ambient and dynamic lighting
	half3 ambient = albedo * dynamic_ambient_color;
	half3 c = ambient + Lo;
	
	//Apply fog
	UNITY_APPLY_FOG(i.fogCoord, c);
	return Output(c, fragment.occlusion);
}

#else

half4 fragAdd(ProjectionInput i) : SV_Target
{
	//Setup Instance Data
	UNITY_SETUP_INSTANCE_ID(i);

	//Generate projection
	Projection projection = CalculateProjection(i.screenPos, i.ray, i.worldForward);

	//Generate base data
	FragmentCommonData fragment = FragmentUnlit(projection, i.worldUp, i.eyeVec);

	//Grab color
	half3 c = fragment.diffColor;

	//Apply fog
	UNITY_APPLY_FOG(i.fogCoord, c);

	//Apply occlusion
	c *= fragment.occlusion;
	
	return Output(c, 1);
}

half4 fragMultiply(ProjectionInput i) : SV_Target
{
	//Setup Instance Data
	UNITY_SETUP_INSTANCE_ID(i);

	//Generate projection
	Projection projection = CalculateProjection(i.screenPos, i.ray, i.worldForward);

	//Generate base data
	FragmentCommonData fragment = FragmentUnlit(projection, i.worldUp, i.eyeVec);

	//Grab color
	half3 c = fragment.diffColor;

	//Apply fog
	UNITY_APPLY_FOG(i.fogCoord, c);

	//Apply occlusion
	c = lerp(half3(1,1,1) , c, fragment.occlusion);
	
	return Output(c, 1);
}

half4 fragUnlit(ProjectionInput i) : SV_Target
{
	//Setup Instance Data
	UNITY_SETUP_INSTANCE_ID(i);

	//Generate projection
	Projection projection = CalculateProjection(i.screenPos, i.ray, i.worldForward);

	//Generate base data
	FragmentCommonData fragment = FragmentUnlit(projection, i.worldUp, i.eyeVec);

	//Grab color
	half3 c = fragment.diffColor;

	//Apply fog
	UNITY_APPLY_FOG(i.fogCoord, c);
	return Output(c, fragment.occlusion);
}

#endif
