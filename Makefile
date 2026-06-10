# AVBD Metal build helpers

.PHONY: build test app cli bench clean

build:
	swift build -c release

test:
	swift test

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

# App with working MLX policy mode (xcodebuild; SwiftPM cannot build MLX shaders)
app-ml:
	xcodebuild -scheme AVBDApp -configuration Release -destination 'platform=macOS' -derivedDataPath .xcbuild build -quiet
	rm -rf AVBD.app
	mkdir -p AVBD.app/Contents/MacOS AVBD.app/Contents/Resources
	cp .xcbuild/Build/Products/Release/AVBDApp AVBD.app/Contents/MacOS/AVBDApp
	cp -R .xcbuild/Build/Products/Release/avbd-metal_AVBDCore.bundle AVBD.app/Contents/Resources/
	-cp .xcbuild/Build/Products/Release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib AVBD.app/Contents/Resources/ 2>/dev/null
	-cp -R .xcbuild/Build/Products/Release/mlx-swift_Cmlx.bundle AVBD.app/Contents/Resources/ 2>/dev/null
	cp AVBD.app/Contents/Info.plist.tmp AVBD.app/Contents/Info.plist 2>/dev/null || printf '%s\n' \
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
