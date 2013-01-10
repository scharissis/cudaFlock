/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#define STRINGIFY(A) #A

const char *vertexShader_cosmos = STRINGIFY(

void main()                                                        
{                                                                      
    float pointSize = 1000.0 * gl_Point.size;                           
    vec4 vert = gl_Vertex;												
    vert.w = 0.75;			// was 1.0											
    vec3 pos_eye = vec3 (gl_ModelViewMatrix * vert);                   
    gl_PointSize = max(1.0, pointSize / (1.0 - pos_eye.z));            
    gl_TexCoord[0] = gl_MultiTexCoord0;                                
    gl_Position = ftransform();                                        
    gl_FrontColor = gl_Color;                                          
    gl_FrontSecondaryColor = gl_SecondaryColor;                       
} 
);


const char *pixelShader_cosmos =  STRINGIFY(

uniform sampler2D splatTexture;                                        
    
void main()                                                            
{                                                                      
    vec4 color2    = gl_SecondaryColor;                                   
    vec4 color     = (0.6 + 0.4 * gl_Color) * texture2D(splatTexture, gl_TexCoord[0].st); 
    gl_FragColor   = color * color2;												 
	 gl_FragColor.w = 0.2;												
} 
);
