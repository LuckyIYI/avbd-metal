# GPU Sim package and AVBD development-workspace build helpers

# macOS renamex_np with RENAME_SWAP (0x2) replaces an existing bundle in one
# namespace operation; os.rename does the same when no bundle exists yet.
# Staging below .build keeps both paths on one volume.
ATOMIC_APP_PUBLISH := python3 -c 'import ctypes, os, sys; src, dst = sys.argv[1:3]; f = ctypes.CDLL(None, use_errno=True).renamex_np; f.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]; f.restype = ctypes.c_int; rc = f(os.fsencode(src), os.fsencode(dst), 0x2) if os.path.lexists(dst) else (os.rename(src, dst) or 0); err = ctypes.get_errno(); sys.exit("atomic app publish failed: " + os.strerror(err)) if rc else None'
DEV_BUILD_DIR := Development/.build
PHYSICS_RESOURCE_BUNDLE := gpu-sim_PhysicsAVBD.bundle
DEMOS_RESOURCE_BUNDLE := gpu-sim_GPUSimDemos.bundle
ROBOTICS_RESOURCE_BUNDLE := avbd-metal_Robotics.bundle

.PHONY: build workspace-build test simulator-test workspace-test \
	verify-package-consumer verify-renderer-consumer verify-core verify-mlx-rl verify-release \
	app cli bench clean clean-all ml-tool \
	app-ml ios-ml verify-gpusim-ios verify-arachne-assets \
	verify-convex-assets \
	verify-arachne-policy verify-policy-evidence verify-panda-provenance \
	verify-h1-provenance verify-architecture

build:
	swift build -c release

workspace-build:
	swift build --package-path Development -c release

simulator-test:
	swift test

workspace-test:
	swift test --package-path Development

test: simulator-test workspace-test

verify-package-consumer:
	swift run --package-path IntegrationTests/PackageConsumer PackageConsumer

verify-renderer-consumer:
	swift run --package-path IntegrationTests/RendererPackageConsumer RendererPackageConsumer

# Compile the public simulator facade and optional renderer for the second
# platform declared by the root manifest. No signing or device is required.
verify-gpusim-ios:
	xcodebuild -scheme GPUSim -configuration Release \
	  -destination 'generic/platform=iOS' \
	  -derivedDataPath .xcbuild-gpusim-ios \
	  CODE_SIGNING_ALLOWED=NO build -quiet
	xcodebuild -scheme GPUSimRenderer -configuration Release \
	  -destination 'generic/platform=iOS' \
	  -derivedDataPath .xcbuild-gpusim-ios \
	  CODE_SIGNING_ALLOWED=NO build -quiet

# Local core merge gate: every checked-in generated/provenance contract plus
# the complete SwiftPM suite. None of these checks needs network access.
verify-core: verify-architecture verify-package-consumer verify-renderer-consumer verify-arachne-assets verify-convex-assets \
	verify-policy-evidence \
	verify-panda-provenance verify-h1-provenance
	swift test
	swift test --package-path Development

verify-architecture:
	python3 Tools/verify_architecture.py
	PYTHONPYCACHEPREFIX=/tmp/avbd-architecture-pycache \
	  python3 -m unittest Tools.tests.test_verify_architecture

# Convex collision assets are cooked offline. Ordinary builds need neither
# Python packages nor CoACD: the merge gate validates the canonical checked-in
# bytes, provenance, topology, and the deterministic cooker contract.
verify-convex-assets:
	PYTHONPYCACHEPREFIX=/tmp/avbd-convex-pycache \
	  python3 -S -m unittest Tools.tests.test_cook_convex_asset
	python3 -S Tools/cook_convex_asset.py \
	  --verify Sources/GPUSimDemos/Assets/convex/concave-u.avbdconvex.json \
	  --input Sources/GPUSimDemos/Assets/convex/concave-u.obj \
	  --debug-obj Sources/GPUSimDemos/Assets/convex/concave-u.debug.obj
	python3 -S Tools/cook_convex_asset.py \
	  --verify Sources/GPUSimDemos/Assets/convex/classic/stanford-bunny.avbdconvex.json \
	  --input Sources/GPUSimDemos/Assets/classic/stanford-bunny.obj \
	  --debug-obj Sources/GPUSimDemos/Assets/convex/classic/stanford-bunny.debug.obj
	python3 -S Tools/cook_convex_asset.py \
	  --verify Sources/GPUSimDemos/Assets/convex/classic/stanford-dragon.avbdconvex.json \
	  --input Sources/GPUSimDemos/Assets/classic/stanford-dragon.obj \
	  --debug-obj Sources/GPUSimDemos/Assets/convex/classic/stanford-dragon.debug.obj
	python3 -S Tools/cook_convex_asset.py \
	  --verify Sources/GPUSimDemos/Assets/convex/classic/stanford-armadillo.avbdconvex.json \
	  --input Sources/GPUSimDemos/Assets/classic/stanford-armadillo.obj \
	  --debug-obj Sources/GPUSimDemos/Assets/convex/classic/stanford-armadillo.debug.obj
	python3 -S Tools/cook_convex_asset.py \
	  --verify Sources/GPUSimDemos/Assets/convex/classic/utah-teapot.avbdconvex.json \
	  --input Sources/GPUSimDemos/Assets/classic/utah-teapot.obj \
	  --debug-obj Sources/GPUSimDemos/Assets/convex/classic/utah-teapot.debug.obj

# Xcode-package the MLX/RL tests, then run their bundle serially so the MLX
# opt-in reaches XCTest (xcodebuild filters custom environment variables).
verify-mlx-rl:
	cd Development && xcodebuild \
	  -scheme avbd-metal-Package \
	  -configuration Debug \
	  -destination 'platform=macOS,arch=arm64' \
	  -derivedDataPath ../.xcbuild-test \
	  -clonedSourcePackagesDirPath ../.xcbuild-test/SourcePackages \
	  -disableAutomaticPackageResolution \
	  -onlyUsePackageVersionsFromResolvedFile \
	  -parallel-testing-enabled NO \
	  -only-testing:AVBDTests/RLFrameworkTests \
	  -only-testing:AVBDTests/VectorPolicyRunnerValidationTests \
	  build-for-testing
	AVBD_MLX_INTEGRATION_TESTS=1 xcrun xctest \
	  -XCTest RLFrameworkTests,VectorPolicyRunnerValidationTests \
	  .xcbuild-test/Build/Products/Debug/AVBDTests.xctest

# Full arm64 Mac release gate: hermetic source/provenance checks, the complete
# simulator suite, packaged MLX integration, and the exact distributable app.
verify-release: verify-core verify-mlx-rl app-ml

cli: workspace-build
	@echo "binary: $(DEV_BUILD_DIR)/release/avbd"

# Wrap the release executable + owned resource bundles into a double-clickable .app
app: workspace-build
	@set -eu; \
	  mkdir -p .build; \
	  staging_root="$$(mktemp -d .build/avbd-app-stage.XXXXXX)"; \
	  staged_app="$$staging_root/AVBD.app"; \
	  cleanup() { rm -rf "$$staging_root"; }; \
	  trap cleanup EXIT HUP INT TERM; \
	  mkdir -p "$$staged_app/Contents/MacOS" "$$staged_app/Contents/Resources"; \
	  cp $(DEV_BUILD_DIR)/release/AVBDApp "$$staged_app/Contents/MacOS/AVBDApp"; \
	  cp -R $(DEV_BUILD_DIR)/release/$(PHYSICS_RESOURCE_BUNDLE) \
	    $(DEV_BUILD_DIR)/release/$(DEMOS_RESOURCE_BUNDLE) \
	    $(DEV_BUILD_DIR)/release/$(ROBOTICS_RESOURCE_BUNDLE) \
	    "$$staged_app/Contents/Resources/"; \
	  cp THIRD_PARTY_NOTICES.md "$$staged_app/Contents/Resources/"; \
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
	  '</dict></plist>' > "$$staged_app/Contents/Info.plist"; \
	  test -x "$$staged_app/Contents/MacOS/AVBDApp"; \
	  test -d "$$staged_app/Contents/Resources/$(PHYSICS_RESOURCE_BUNDLE)"; \
	  test -d "$$staged_app/Contents/Resources/$(DEMOS_RESOURCE_BUNDLE)"; \
	  test -d "$$staged_app/Contents/Resources/$(ROBOTICS_RESOURCE_BUNDLE)"; \
	  test -s "$$staged_app/Contents/Resources/THIRD_PARTY_NOTICES.md"; \
	  plutil -lint "$$staged_app/Contents/Info.plist" >/dev/null; \
	  codesign --force --deep --sign - "$$staged_app"; \
	  codesign --verify --deep --strict "$$staged_app"; \
	  $(ATOMIC_APP_PUBLISH) "$$staged_app" AVBD.app; \
	  echo "built AVBD.app"

bench: workspace-build
	$(DEV_BUILD_DIR)/release/avbd bench boxpile --frames 100 --scale 3

clean:
	rm -rf .build Development/.build IntegrationTests/PackageConsumer/.build \
	  IntegrationTests/RendererPackageConsumer/.build AVBD.app

# Deliberately expensive cleanup for every ignored Xcode build cache. Normal
# `clean` keeps these caches so rebuilding the MLX app does not start cold.
clean-all: clean
	rm -rf .xcbuild .xcbuild-app .xcbuild-ios .xcbuild-gpusim-ios .xcbuild-test

# ML tool: MLX requires xcodebuild (SwiftPM cannot compile its Metal shaders)
ml-tool:
	cd Development && xcodebuild -scheme avbd -configuration Release -destination 'platform=macOS' \
	  -derivedDataPath ../.xcbuild \
	  -clonedSourcePackagesDirPath ../.xcbuild/SourcePackages \
	  -disableAutomaticPackageResolution \
	  -onlyUsePackageVersionsFromResolvedFile build -quiet
	@echo "binary: .xcbuild/Build/Products/Release/avbd"

# App with working MLX policy mode (xcodebuild; SwiftPM cannot build MLX shaders).
# Accepted policy evidence is verified before any checkpoint enters the bundle.
# Historical/development actors remain repository-only; release packaging
# contains only bundles with accepted or external-parity release-index evidence.
app-ml: verify-policy-evidence
	cd Development && xcodebuild -scheme avbd -configuration Release -destination 'platform=macOS' \
	  -derivedDataPath ../.xcbuild \
	  -clonedSourcePackagesDirPath ../.xcbuild/SourcePackages \
	  -disableAutomaticPackageResolution \
	  -onlyUsePackageVersionsFromResolvedFile build -quiet
	cd Development && xcodebuild -scheme AVBDApp -configuration Release -destination 'platform=macOS' \
	  -derivedDataPath ../.xcbuild \
	  -clonedSourcePackagesDirPath ../.xcbuild/SourcePackages \
	  -disableAutomaticPackageResolution \
	  -onlyUsePackageVersionsFromResolvedFile build -quiet
	@set -eu; \
	  mkdir -p .build; \
	  staging_root="$$(mktemp -d .build/avbd-app-stage.XXXXXX)"; \
	  staged_app="$$staging_root/AVBD.app"; \
	  cleanup() { rm -rf "$$staging_root"; }; \
	  trap cleanup EXIT HUP INT TERM; \
	  mkdir -p "$$staged_app/Contents/MacOS" \
	    "$$staged_app/Contents/Resources/checkpoints/external/unitree-h1"; \
	  mkdir -p \
	    "$$staged_app/Contents/Resources/checkpoints/humanoid-isaac-flat-v2/qualification" \
	    "$$staged_app/Contents/Resources/checkpoints/arachne15-velocity-v1/qualification/nominal" \
	    "$$staged_app/Contents/Resources/checkpoints/arachne15-velocity-v1/qualification/validation-collision" \
	    "$$staged_app/Contents/Resources/checkpoints/arachne15-goal-v1/qualification/nominal" \
	    "$$staged_app/Contents/Resources/checkpoints/arachne15-goal-v1/qualification/validation-collision"; \
	  cp .xcbuild/Build/Products/Release/AVBDApp "$$staged_app/Contents/MacOS/AVBDApp"; \
	  cp .xcbuild/Build/Products/Release/avbd "$$staged_app/Contents/MacOS/avbd"; \
	  cp -R .xcbuild/Build/Products/Release/$(PHYSICS_RESOURCE_BUNDLE) \
	    .xcbuild/Build/Products/Release/$(DEMOS_RESOURCE_BUNDLE) \
	    .xcbuild/Build/Products/Release/$(ROBOTICS_RESOURCE_BUNDLE) \
	    "$$staged_app/Contents/Resources/"; \
	  cp THIRD_PARTY_NOTICES.md "$$staged_app/Contents/Resources/"; \
	  cp checkpoints/README.md checkpoints/policy-release-index.json \
	    "$$staged_app/Contents/Resources/checkpoints/"; \
	  cp checkpoints/external/unitree-h1/LICENSE \
	  checkpoints/external/unitree-h1/manifest.json \
	  checkpoints/external/unitree-h1/policy-bundle.json \
	  checkpoints/external/unitree-h1/policy.safetensors \
	  "$$staged_app/Contents/Resources/checkpoints/external/unitree-h1/"; \
	  cp checkpoints/humanoid-isaac-flat-v2/deployment-manifest.json \
	  checkpoints/humanoid-isaac-flat-v2/metadata.json \
	  checkpoints/humanoid-isaac-flat-v2/policy-bundle.json \
	  checkpoints/humanoid-isaac-flat-v2/policy.safetensors \
	  checkpoints/humanoid-isaac-flat-v2/requalification-manifest.json \
	  checkpoints/humanoid-isaac-flat-v2/training-state.json \
	  "$$staged_app/Contents/Resources/checkpoints/humanoid-isaac-flat-v2/"; \
	  cp checkpoints/humanoid-isaac-flat-v2/qualification/aggregate.json \
	  checkpoints/humanoid-isaac-flat-v2/qualification/eval-seed-51001.json \
	  checkpoints/humanoid-isaac-flat-v2/qualification/eval-seed-51002.json \
	  checkpoints/humanoid-isaac-flat-v2/qualification/eval-seed-51003.json \
	  checkpoints/humanoid-isaac-flat-v2/qualification/eval-seed-51004.json \
	  "$$staged_app/Contents/Resources/checkpoints/humanoid-isaac-flat-v2/qualification/"; \
	  cp checkpoints/arachne15-velocity-v1/deployment-manifest.json \
	  checkpoints/arachne15-velocity-v1/metadata.json \
	  checkpoints/arachne15-velocity-v1/policy-bundle.json \
	  checkpoints/arachne15-velocity-v1/policy.safetensors \
	  checkpoints/arachne15-velocity-v1/requalification-manifest.json \
	  checkpoints/arachne15-velocity-v1/training-state.json \
	  "$$staged_app/Contents/Resources/checkpoints/arachne15-velocity-v1/"; \
	  cp checkpoints/arachne15-velocity-v1/qualification/nominal/aggregate.json \
	  checkpoints/arachne15-velocity-v1/qualification/nominal/eval-seed-61001.json \
	  checkpoints/arachne15-velocity-v1/qualification/nominal/eval-seed-61002.json \
	  checkpoints/arachne15-velocity-v1/qualification/nominal/eval-seed-61003.json \
	  checkpoints/arachne15-velocity-v1/qualification/nominal/eval-seed-61004.json \
	  "$$staged_app/Contents/Resources/checkpoints/arachne15-velocity-v1/qualification/nominal/"; \
	  cp checkpoints/arachne15-velocity-v1/qualification/validation-collision/aggregate.json \
	  checkpoints/arachne15-velocity-v1/qualification/validation-collision/eval-seed-61501.json \
	  checkpoints/arachne15-velocity-v1/qualification/validation-collision/eval-seed-61502.json \
	  checkpoints/arachne15-velocity-v1/qualification/validation-collision/eval-seed-61503.json \
	  checkpoints/arachne15-velocity-v1/qualification/validation-collision/eval-seed-61504.json \
	  "$$staged_app/Contents/Resources/checkpoints/arachne15-velocity-v1/qualification/validation-collision/"; \
	  cp checkpoints/arachne15-goal-v1/deployment-manifest.json \
	  checkpoints/arachne15-goal-v1/metadata.json \
	  checkpoints/arachne15-goal-v1/policy-bundle.json \
	  checkpoints/arachne15-goal-v1/policy.safetensors \
	  checkpoints/arachne15-goal-v1/requalification-manifest.json \
	  checkpoints/arachne15-goal-v1/training-state.json \
	  "$$staged_app/Contents/Resources/checkpoints/arachne15-goal-v1/"; \
	  cp checkpoints/arachne15-goal-v1/qualification/nominal/aggregate.json \
	  checkpoints/arachne15-goal-v1/qualification/nominal/eval-seed-62001.json \
	  checkpoints/arachne15-goal-v1/qualification/nominal/eval-seed-62002.json \
	  checkpoints/arachne15-goal-v1/qualification/nominal/eval-seed-62003.json \
	  checkpoints/arachne15-goal-v1/qualification/nominal/eval-seed-62004.json \
	  "$$staged_app/Contents/Resources/checkpoints/arachne15-goal-v1/qualification/nominal/"; \
	  cp checkpoints/arachne15-goal-v1/qualification/validation-collision/aggregate.json \
	  checkpoints/arachne15-goal-v1/qualification/validation-collision/eval-seed-63001.json \
	  checkpoints/arachne15-goal-v1/qualification/validation-collision/eval-seed-63002.json \
	  checkpoints/arachne15-goal-v1/qualification/validation-collision/eval-seed-63003.json \
	  checkpoints/arachne15-goal-v1/qualification/validation-collision/eval-seed-63004.json \
	  "$$staged_app/Contents/Resources/checkpoints/arachne15-goal-v1/qualification/validation-collision/"; \
	  cp -R .xcbuild/Build/Products/Release/mlx-swift_Cmlx.bundle \
	    "$$staged_app/Contents/Resources/"; \
	  test -f "$$staged_app/Contents/Resources/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"; \
	  python3 Tools/verify_policy_evidence.py \
	    --app-checkpoints "$$staged_app/Contents/Resources/checkpoints"; \
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
	  '</dict></plist>' > "$$staged_app/Contents/Info.plist"; \
	  test -x "$$staged_app/Contents/MacOS/AVBDApp"; \
	  test -x "$$staged_app/Contents/MacOS/avbd"; \
	  test -d "$$staged_app/Contents/Resources/$(PHYSICS_RESOURCE_BUNDLE)"; \
	  test -d "$$staged_app/Contents/Resources/$(DEMOS_RESOURCE_BUNDLE)"; \
	  test -d "$$staged_app/Contents/Resources/$(ROBOTICS_RESOURCE_BUNDLE)"; \
	  test -s "$$staged_app/Contents/Resources/THIRD_PARTY_NOTICES.md"; \
	  plutil -lint "$$staged_app/Contents/Info.plist" >/dev/null; \
	  codesign --force --deep --sign - "$$staged_app"; \
	  codesign --verify --deep --strict "$$staged_app"; \
	  "$$staged_app/Contents/MacOS/avbd" list-rl >/dev/null; \
	  "$$staged_app/Contents/MacOS/avbd" verify-policy-rl \
	  humanoid-isaac-flat-v0 \
	  --checkpoint "$$staged_app/Contents/Resources/checkpoints/humanoid-isaac-flat-v2" \
	  --frames 1 --json >/dev/null; \
	  "$$staged_app/Contents/MacOS/avbd" verify-policy-rl \
	  arachne15-velocity-v0 \
	  --checkpoint "$$staged_app/Contents/Resources/checkpoints/arachne15-velocity-v1" \
	  --frames 1 --json >/dev/null; \
	  "$$staged_app/Contents/MacOS/avbd" verify-policy-rl \
	  arachne15-goal-v0 \
	  --checkpoint "$$staged_app/Contents/Resources/checkpoints/arachne15-goal-v1" \
	  --frames 1 --json >/dev/null; \
	  $(ATOMIC_APP_PUBLISH) "$$staged_app" AVBD.app; \
	  echo "built AVBD.app (ML-enabled)"

# Compile the reusable MLX runtime for a physical iOS device. Xcode must have
# the matching iOS platform installed; no signing is required for this library.
ios-ml:
	cd Development && xcodebuild -scheme MLXRL -configuration Release \
	  -destination 'generic/platform=iOS' -derivedDataPath ../.xcbuild-ios \
	  -clonedSourcePackagesDirPath ../.xcbuild/SourcePackages \
	  -disableAutomaticPackageResolution \
	  -onlyUsePackageVersionsFromResolvedFile build -quiet

# Read-only CI guard for the imported runtime snapshot. The CAD, hardware,
# generator, and device-qualification project is intentionally external.
verify-arachne-assets:
	python3 Tools/verify_arachne15_assets.py
	PYTHONPYCACHEPREFIX=/tmp/avbd-arachne-pycache \
	  python3 -m unittest Tools.tests.test_verify_arachne15_assets

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
	  arachne15-velocity-v0 \
	  --checkpoint checkpoints/arachne15-velocity-v1 \
	  --frames 500 --json
	.xcbuild/Build/Products/Release/avbd verify-policy-rl \
	  arachne15-goal-v0 \
	  --checkpoint checkpoints/arachne15-goal-v1 \
	  --frames 500 --json

# Rebuild every accepted result from its raw, immutable evidence without MLX.
# The verifier discovers accepted entries from policy-release-index.json and fails
# closed when a future evidence shape has not been added to the contract.
verify-policy-evidence:
	python3 Tools/verify_policy_evidence.py
