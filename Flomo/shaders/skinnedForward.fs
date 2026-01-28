#version 330

// Simple forward-rendering fragment shader for skinned meshes

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;

uniform vec4 colDiffuse;

out vec4 finalColor;

void main()
{
    // Simple directional lighting
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
    vec3 normal = normalize(fragNormal);

    float diff = max(dot(normal, lightDir), 0.0);
    float ambient = 0.3;
    float lighting = ambient + diff * 0.7;

    vec3 color = fragColor.rgb * colDiffuse.rgb * lighting;

    finalColor = vec4(color, fragColor.a * colDiffuse.a);
}
