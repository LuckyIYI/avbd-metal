# Native mesh-normal diagnostic

Launch with `AVBD_NORMAL_PASS=1` to replace native PBR fragment shading with normalized, interpolated mesh normals mapped from [-1,1] to [0,1]. This is a shader-construction-time diagnostic: restart/recreate the renderer after changing the environment variable.

It bypasses material textures, procedural normal maps, lighting and display transforms. It visualizes the mesh's transformed shading normals, not face normals or the final material normal. Normal rendering remains unchanged when the variable is unset. This is not an HQ denoising guide or an exported normal AOV.
