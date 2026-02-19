#version 330

// Forward-rendering fragment shader for skinned meshes with shadow mapping

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;
in vec4 fragPositionLightSpace;

uniform vec4 colDiffuse;
uniform vec3 sunDir;
uniform sampler2D shadowMap;
uniform float lightClipNear;
uniform float lightClipFar;

out vec4 finalColor;

float LinearDepth(float depth, float near, float far)
{
    return (2.0 * near) / (far + near - depth * (far - near));
}

void main()
{
    vec3 normal = normalize(fragNormal);

    // Shadow map lookup
    vec3 posLS = fragPositionLightSpace.xyz / fragPositionLightSpace.w;
    posLS = posLS * 0.5 + 0.5;

    float clip = float(
        posLS.x > 0.0 && posLS.x < 1.0 &&
        posLS.y > 0.0 && posLS.y < 1.0);

    float fd = LinearDepth(posLS.z, lightClipNear, lightClipFar);
    float sd = texture(shadowMap, posLS.xy).r;
    float shadow = 1.0 - clip * float(fd - 0.0001 > sd);

    // Directional lighting using actual sun direction
    float diff = max(dot(normal, -sunDir), 0.0);
    float ambient = 0.3;
    float lighting = ambient + diff * 0.7 * shadow;

    vec3 color = fragColor.rgb * colDiffuse.rgb * lighting;

    finalColor = vec4(color, fragColor.a * colDiffuse.a);
}
