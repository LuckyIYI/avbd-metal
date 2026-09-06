// Metal adaptation of the FXAA 3.11 quality edge search by Timothy Lottes.
// Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of NVIDIA CORPORATION nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS
// OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
// NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

let antialiasingShaderSource = """
inline float edgeLuma(float3 rgb) { return dot(rgb, float3(0.299, 0.587, 0.114)); }
fragment float4 edge_antialiasing_fragment(FSOut in [[stage_in]], texture2d<float> image [[texture(0)]]) {
    // The unorm view exposes display-encoded values: edge detection and
    // reconstruction happen after tone mapping, without decoding sRGB twice.
    constexpr sampler bilinear(filter::linear, address::clamp_to_edge);
    int2 size = int2(image.get_width(), image.get_height());
    int2 p = int2(in.position.xy);
    float2 texel = 1.0 / float2(size);
    float2 uv = (float2(p) + 0.5) * texel;
    float3 center = image.read(uint2(p)).rgb;
    float m = edgeLuma(center);
    float n = edgeLuma(image.read(uint2(clamp(p+int2(0,-1), int2(0), size-1))).rgb);
    float s = edgeLuma(image.read(uint2(clamp(p+int2(0, 1), int2(0), size-1))).rgb);
    float e = edgeLuma(image.read(uint2(clamp(p+int2(1, 0), int2(0), size-1))).rgb);
    float w = edgeLuma(image.read(uint2(clamp(p+int2(-1,0), int2(0), size-1))).rgb);
    float high = max(m,max(max(n,s),max(e,w))), low = min(m,min(min(n,s),min(e,w)));
    float range = high-low;
    if (range < max(0.025, high*0.125)) return float4(sRGBToLinearExact(center),1);
    float nw = edgeLuma(image.read(uint2(clamp(p+int2(-1,-1),int2(0),size-1))).rgb);
    float ne = edgeLuma(image.read(uint2(clamp(p+int2( 1,-1),int2(0),size-1))).rgb);
    float sw = edgeLuma(image.read(uint2(clamp(p+int2(-1, 1),int2(0),size-1))).rgb);
    float se = edgeLuma(image.read(uint2(clamp(p+int2( 1, 1),int2(0),size-1))).rgb);
    float horizontal = 2*abs(n+s-2*m) + abs(ne+se-2*e) + abs(nw+sw-2*w);
    float vertical = 2*abs(e+w-2*m) + abs(ne+nw-2*n) + abs(se+sw-2*s);
    bool alongX = horizontal >= vertical;
    float a = alongX ? n : w, b = alongX ? s : e;
    bool towardA = abs(a-m) >= abs(b-m);
    float gradient = max(abs(a-m),abs(b-m));
    float edgeAverage = 0.5*(m+(towardA ? a : b));
    float2 across = alongX ? float2(0,texel.y) : float2(texel.x,0);
    across *= towardA ? -1.0 : 1.0;
    float2 along = alongX ? float2(texel.x,0) : float2(0,texel.y);
    float2 edge = uv + across*0.5, left = edge-along, right = edge+along;
    float endLeft = edgeLuma(image.sample(bilinear,left).rgb)-edgeAverage;
    float endRight = edgeLuma(image.sample(bilinear,right).rgb)-edgeAverage;
    bool doneLeft = abs(endLeft) >= gradient*0.25, doneRight = abs(endRight) >= gradient*0.25;
    const float distances[4] = {1.5,2.0,4.0,12.0};
    for (uint i = 0; i < 4 && !(doneLeft && doneRight); ++i) {
        if (!doneLeft) {
            left -= along*distances[i];
            endLeft = edgeLuma(image.sample(bilinear,left).rgb)-edgeAverage;
            doneLeft = abs(endLeft) >= gradient*0.25;
        }
        if (!doneRight) {
            right += along*distances[i];
            endRight = edgeLuma(image.sample(bilinear,right).rgb)-edgeAverage;
            doneRight = abs(endRight) >= gradient*0.25;
        }
    }
    float dl = alongX ? uv.x-left.x : uv.y-left.y;
    float dr = alongX ? right.x-uv.x : right.y-uv.y;
    float end = dl < dr ? endLeft : endRight;
    float edgeOffset = (end < 0) != (m-edgeAverage < 0) ? 0.5-min(dl,dr)/(dl+dr) : 0;
    float subpixel = saturate(abs((2*(n+s+e+w)+nw+ne+sw+se)/12.0-m)/range);
    subpixel = subpixel*subpixel*(3-2*subpixel);
    float offset = max(edgeOffset, subpixel*subpixel*0.5);
    return float4(sRGBToLinearExact(image.sample(bilinear,uv+across*offset).rgb),1);
}
"""
