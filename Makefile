# AVBD Metal build helpers

.PHONY: build test verify-core verify-mlx-rl app cli bench clean ml-tool \
	app-ml ios-ml generate-arachne-assets verify-arachne-assets \
	verify-arachne-policy verify-policy-evidence verify-panda-provenance \
	verify-h1-provenance

build:
	swift build -c release

test:
	swift test

# Local core merge gate: every checked-in generated/provenance contract plus
# the complete SwiftPM suite. None of these checks needs network access.
verify-core: verify-arachne-assets verify-policy-evidence \
	verify-panda-provenance verify-h1-provenance
	swift test

# Xcode-package the MLX/RL tests, then run their bundle serially so the MLX
# opt-in reaches XCTest (xcodebuild filters custom environment variables).
verify-mlx-rl:
	xcodebuild \
	  -scheme avbd-metal-Package \
	  -configuration Debug \
	  -destination 'platform=macOS,arch=arm64' \
	  -derivedDataPath .xcbuild-test \
	  -clonedSourcePackagesDirPath .xcbuild-test/SourcePackages \
	  -disableAutomaticPackageResolution \
	  -onlyUsePackageVersionsFromResolvedFile \
	  -parallel-testing-enabled NO \
	  -only-testing:AVBDTests/RLFrameworkTests \
	  build-for-testing
	AVBD_MLX_INTEGRATION_TESTS=1 xcrun xctest \
	  -XCTest RLFrameworkTests \
	  .xcbuild-test/Build/Products/Debug/AVBDTests.xctest

cli: build
	@echo "binary: .build/release/avbd"

# Wrap the release executable + resource bundle into a double-clickable .app
app: build
	rm -rf AVBD.app
	mkdir -p AVBD.app/Contents/MacOS AVBD.app/Contents/Resources
	cp .build/release/AVBDApp AVBD.app/Contents/MacOS/AVBDApp
	cp -R .build/release/avbd-metal_AVBDCore.bundle AVBD.app/Contents/Resources/
	printf '%s\n' \
	  '<?xml version="1.0" encoding="UTF-8"?>' \
	  '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">' \
	  '<plist version="1.0"><dict>' \
	  '  <key>CFBundleName</key><string>AVBD</string>' \
	  '  <key>CFBundleDisplayName</key><string>AVBD Metal</string>' \
	  '  <key>CFBundleIdentifier</key><string>dev.avbd.metal</string>' \
	  '  <key>CFBundleExecutable</key><string>AVBDApp</string>' \
	  '  <key>CFBundlePackageType</key><string>APPL</string>' \
	  '  <key>CFBundleShortVersionString</key><string>1.0</string>' \
	  '  <key>NSHighResolutionCapable</key><true/>' \
	  '  <key>LSMinimumSystemVersion</key><string>14.0</string>' \
	  '</dict></plist>' > AVBD.app/Contents/Info.plist
	codesign --force --sign - AVBD.app 2>/dev/null || true
	@echo "built AVBD.app"

bench: build
	.build/release/avbd bench boxpile --frames 100 --scale 3

clean:
	rm -rf .build AVBD.app

# ML tool: MLX requires xcodebuild (SwiftPM cannot compile its Metal shaders)
ml-tool:
	xcodebuild -scheme avbd -configuration Release -destination 'platform=macOS' -derivedDataPath .xcbuild build -quiet
	@echo "binary: .xcbuild/Build/Products/Release/avbd"

# App with working MLX policy mode (xcodebuild; SwiftPM cannot build MLX shaders).
# Accepted policy evidence is verified before any checkpoint enters the bundle.
app-ml: verify-policy-evidence
	xcodebuild -scheme AVBDApp -configuration Release -destination 'platform=macOS' -derivedDataPath .xcbuild build -quiet
	rm -rf AVBD.app
	mkdir -p AVBD.app/Contents/MacOS AVBD.app/Contents/Resources/checkpoints/external/unitree-h1
	mkdir -p AVBD.app/Contents/Resources/checkpoints/humanoid-isaac-flat-v2/qualification
	mkdir -p AVBD.app/Contents/Resources/checkpoints/humanoid-isaac-goal-v0
	mkdir -p AVBD.app/Contents/Resources/checkpoints/arachne15-velocity-v0
	mkdir -p AVBD.app/Contents/Resources/checkpoints/arachne15-goal-v0
	cp .xcbuild/Build/Products/Release/AVBDApp AVBD.app/Contents/MacOS/AVBDApp
	cp -R .xcbuild/Build/Products/Release/avbd-metal_AVBDCore.bundle AVBD.app/Contents/Resources/
	cp checkpoints/README.md AVBD.app/Contents/Resources/checkpoints/
	cp checkpoints/external/unitree-h1/LICENSE \
	  checkpoints/external/unitree-h1/manifest.json \
	  checkpoints/external/unitree-h1/policy.safetensors \
	  AVBD.app/Contents/Resources/checkpoints/external/unitree-h1/
	# v0/v1 remain repository-only lineage; ship the accepted epoch-2 v2 bundle.
	cp checkpoints/humanoid-isaac-flat-v2/deployment-manifest.json \
	  checkpoints/humanoid-isaac-flat-v2/metadata.json \
	  checkpoints/humanoid-isaac-flat-v2/policy.safetensors \
	  checkpoints/humanoid-isaac-flat-v2/requalification-manifest.json \
	  checkpoints/humanoid-isaac-flat-v2/training-state.json \
	  AVBD.app/Contents/Resources/checkpoints/humanoid-isaac-flat-v2/
	cp checkpoints/humanoid-isaac-flat-v2/qualification/aggregate.json \
	  checkpoints/humanoid-isaac-flat-v2/qualification/eval-seed-51001.json \
	  checkpoints/humanoid-isaac-flat-v2/qualification/eval-seed-51002.json \
	  checkpoints/humanoid-isaac-flat-v2/qualification/eval-seed-51003.json \
	  checkpoints/humanoid-isaac-flat-v2/qualification/eval-seed-51004.json \
	  AVBD.app/Contents/Resources/checkpoints/humanoid-isaac-flat-v2/qualification/
	cp checkpoints/humanoid-isaac-goal-v0/evaluation.json \
	  checkpoints/humanoid-isaac-goal-v0/metadata.json \
	  checkpoints/humanoid-isaac-goal-v0/policy.safetensors \
	  checkpoints/humanoid-isaac-goal-v0/training-state.json \
	  AVBD.app/Contents/Resources/checkpoints/humanoid-isaac-goal-v0/
	cp checkpoints/arachne15-velocity-v0/evaluation.json \
	  checkpoints/arachne15-velocity-v0/metadata.json \
	  checkpoints/arachne15-velocity-v0/policy.safetensors \
	  checkpoints/arachne15-velocity-v0/training-state.json \
	  AVBD.app/Contents/Resources/checkpoints/arachne15-velocity-v0/
	cp checkpoints/arachne15-goal-v0/deployment-manifest.json \
	  checkpoints/arachne15-goal-v0/metadata.json \
	  checkpoints/arachne15-goal-v0/policy.safetensors \
	  checkpoints/arachne15-goal-v0/training-state.json \
	  AVBD.app/Contents/Resources/checkpoints/arachne15-goal-v0/
	-cp .xcbuild/Build/Products/Release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib AVBD.app/Contents/Resources/ 2>/dev/null
	-cp -R .xcbuild/Build/Products/Release/mlx-swift_Cmlx.bundle AVBD.app/Contents/Resources/ 2>/dev/null
	printf '%s\n' \
	  '<?xml version="1.0" encoding="UTF-8"?>' \
	  '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">' \
	  '<plist version="1.0"><dict>' \
	  '  <key>CFBundleName</key><string>AVBD</string>' \
	  '  <key>CFBundleDisplayName</key><string>AVBD Metal</string>' \
	  '  <key>CFBundleIdentifier</key><string>dev.avbd.metal</string>' \
	  '  <key>CFBundleExecutable</key><string>AVBDApp</string>' \
	  '  <key>CFBundlePackageType</key><string>APPL</string>' \
	  '  <key>CFBundleShortVersionString</key><string>1.0</string>' \
	  '  <key>NSHighResolutionCapable</key><true/>' \
	  '  <key>LSMinimumSystemVersion</key><string>14.0</string>' \
	  '</dict></plist>' > AVBD.app/Contents/Info.plist
	codesign --force --sign - AVBD.app 2>/dev/null || true
	@echo "built AVBD.app (ML-enabled)"

# Compile the reusable MLX runtime for a physical iOS device. Xcode must have
# the matching iOS platform installed; no signing is required for this library.
ios-ml:
	xcodebuild -scheme AVBDLearn -configuration Release \
	  -destination 'generic/platform=iOS' -derivedDataPath .xcbuild-ios \
	  -clonedSourcePackagesDirPath .xcbuild/SourcePackages \
	  -disableAutomaticPackageResolution \
	  -onlyUsePackageVersionsFromResolvedFile build -quiet

# Rewrite the tracked MJCF files from their generator, then validate both the
# robot-tree and SwiftPM-resource versions against packaged visual meshes.
# Rebuilding CAD explicitly installs refreshed visual meshes first.
generate-arachne-assets:
	Robots/Arachne15/scripts/build_sim.sh

# Read-only CI guard: fail when generated MJCF is stale, a required packaged
# mesh is missing, or the structural/mass/articulation contract is violated.
# It does not depend on ignored Robots/*/build output, so it works in a clone.
verify-arachne-assets:
	python3 Robots/Arachne15/sim/generate_model.py --check
	python3 Robots/Arachne15/sim/validate_model.py
	python3 Robots/Arachne15/analysis/reveal_pose.py --check

# Verify the cleanly sourced Panda plant, first-party pusher and explicitly
# attributed ManiSkill Push-T task boundary.
verify-panda-provenance:
	python3 Tools/verify_panda_provenance.py

# Verify the generated Unitree/Menagerie collision support sets and the exact
# Isaac Lab configuration sources adapted by the H1 task.
verify-h1-provenance:
	python3 Tools/generate_unitree_h1_collision_hulls.py --verify
	python3 Tools/verify_isaac_lab_h1_provenance.py

# Validate the tracked bundle, exact inference parity, and Metal latency.
verify-arachne-policy: ml-tool
	.xcbuild/Build/Products/Release/avbd verify-policy-rl \
	  arachne15-goal-v0 \
	  --checkpoint checkpoints/arachne15-goal-v0 \
	  --frames 500 --json

# Rebuild every accepted result from its raw, immutable evidence without MLX.
# The verifier discovers accepted entries from PolicyReplayCatalog and fails
# closed when a future evidence shape has not been added to the contract.
verify-policy-evidence:
	python3 Tools/verify_policy_evidence.py
