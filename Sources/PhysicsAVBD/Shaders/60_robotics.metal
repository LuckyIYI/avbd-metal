// ----------------------------------------------------------------------------
// Robotics mode: parallel top-down pixel observations for world models.
// One thread per (env, pixel): analytic rasterization of the Push-T scene —
// goal T (green), block T (red), pusher finger (blue) — into RGB uint8.
// No geometry pass, no cameras: reads body poses directly.
// ----------------------------------------------------------------------------

struct PushTEnvGPU {
    uint4 ids;          // tip, bar, stem, pad
    float4 frame;       // env center xy, goal xy
    float4 goal;        // x = goal yaw; y = half window; z,w pad
};

inline bool inBox(float2 p, float2 c, float yaw, float2 half2) {
    float cs = cos(yaw), sn = sin(yaw);
    float2 d = p - c;
    float2 l = float2(cs * d.x + sn * d.y, -sn * d.x + cs * d.y);
    return fabs(l.x) <= half2.x && fabs(l.y) <= half2.y;
}

kernel void pusht_obs(
    device const float4* posLin     [[buffer(0)]],
    device const float4* posAng     [[buffer(1)]],
    device const PushTEnvGPU* envs  [[buffer(2)]],
    device uchar* out               [[buffer(3)]],   // numEnvs*res*res*3
    constant uint& res              [[buffer(4)]],
    uint3 tid                       [[thread_position_in_grid]])
{
    uint e = tid.z;
    uint px = tid.x, py = tid.y;
    if (px >= res || py >= res) return;
    PushTEnvGPU env = envs[e];
    float halfWin = env.goal.y;
    float2 world = env.frame.xy
        + (float2(px, py) / float(res - 1) - 0.5) * (2.0 * halfWin);

    float3 rgb = float3(0.06);                       // background

    // goal T (green, drawn under everything)
    float2 g = env.frame.zw + env.frame.xy;
    float gy = env.goal.x;
    float cs = cos(gy), sn = sin(gy);
    float2 barC = g + float2(-sn, cs) * 0.125;
    float2 stemC = g + float2(sn, -cs) * 0.325;
    if (inBox(world, barC, gy, float2(0.5, 0.125)) ||
        inBox(world, stemC, gy, float2(0.125, 0.325))) {
        rgb = float3(0.10, 0.55, 0.12);
    }

    // block T (red): bar + stem boxes in their own frames
    for (uint k = 0; k < 2; k++) {
        uint b = k == 0 ? env.ids.y : env.ids.z;
        float4 q = posAng[b];
        float3 fwd = q_rotate(q, float3(1, 0, 0));
        float yaw = atan2(fwd.y, fwd.x);
        float2 c = posLin[b].xy;
        float2 h = k == 0 ? float2(0.5, 0.125) : float2(0.125, 0.325);
        if (inBox(world, c, yaw, h)) rgb = float3(0.85, 0.15, 0.12);
    }

    // pusher finger (blue disc)
    float2 t = posLin[env.ids.x].xy;
    if (distance(world, t) < 0.13) rgb = float3(0.2, 0.4, 0.95);

    uint idx = ((e * res + py) * res + px) * 3;
    out[idx + 0] = uchar(saturate(rgb.x) * 255.0);
    out[idx + 1] = uchar(saturate(rgb.y) * 255.0);
    out[idx + 2] = uchar(saturate(rgb.z) * 255.0);
}
