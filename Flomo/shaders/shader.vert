#version 330 core

in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;

uniform int isCapsule;
uniform vec3 capsulePosition;
uniform vec4 capsuleRotation;
uniform float capsuleHalfLength;
uniform float capsuleRadius;

uniform mat4 matModel;
uniform mat4 matNormal;
uniform mat4 matView;
uniform mat4 matProjection;

out vec3 fragPosition;
out vec2 fragTexCoord;
out vec3 fragNormal;

#define AO_RATIO_MAX 4.0
#define AO_CAPSULES_MAX 32
#define SHADOW_CAPSULES_MAX 64
#define PI 3.141592653589793

vec3 Rotate(vec4 q, vec3 v)
{
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

vec3 CapsuleStretch(vec3 pos, float hlength, float radius)
{
    vec3 scaled = pos * radius;
    scaled.x = scaled.x > 0.0 ? scaled.x + hlength : scaled.x - hlength;
    return scaled;
}

void main()
{
    fragTexCoord = vertexTexCoord;

    if (isCapsule == 1)
    {
        fragPosition = Rotate(capsuleRotation,
            CapsuleStretch(vertexPosition,
            capsuleHalfLength, capsuleRadius)) + capsulePosition;

        fragNormal = Rotate(capsuleRotation, vertexNormal);
    }
    else
    {
        fragPosition = vec3(matModel * vec4(vertexPosition, 1.0));
        fragNormal = normalize(vec3(matNormal * vec4(vertexNormal, 1.0)));
    }

    gl_Position = matProjection * matView * vec4(fragPosition, 1.0);
}