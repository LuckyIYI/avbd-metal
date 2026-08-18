import SimCore
import simd

/// Published dimensions used to lay out the Gaudí funicular demo. The
/// Sagrada Família information booklets describe the temple in 7.5 m
/// modules: 90 m interior length, a 45 m five-nave body, a 60 m transept,
/// and a 15 m central nave.
public enum GaudiFunicularBlueprint {
    public static let moduleMeters: Float = 7.5
    public static let interiorLengthMeters: Float = 90
    public static let naveWidthMeters: Float = 45
    public static let transeptWidthMeters: Float = 60
    public static let centralNaveWidthMeters: Float = 15
    public static let modelScale: Float = 0.12
    public static let verticalScale: Float = 0.06
    public static let sideNaveHeightMeters: Float = 30
    public static let centralNaveHeightMeters: Float = 45
    public static let crossingHeightMeters: Float = 60
    public static let apseHeightMeters: Float = 75
    public static let jesusTowerHeightMeters: Float = 172.5
    public static let maryTowerHeightMeters: Float = 138
    public static let evangelistTowerHeightMeters: Float = 135

    public static var module: Float { moduleMeters * modelScale }
    public static var interiorLength: Float { interiorLengthMeters * modelScale }
    public static var naveWidth: Float { naveWidthMeters * modelScale }
    public static var transeptWidth: Float { transeptWidthMeters * modelScale }
}

extension Demos {
    /// A live three-dimensional polyfunicular workshop.
    ///
    /// Historical provenance matters here: Gaudí's surviving photographic
    /// hanging-model evidence and the modern museum reconstruction are for
    /// the church at Colònia Güell. For the Sagrada Família he described
    /// using the corresponding graphical funicular method. This scene uses
    /// the documented Sagrada Família dimensional module and load hierarchy,
    /// but the physical apparatus follows the Colònia Güell string-and-
    /// birdshot technique.
    ///
    /// Each ochre capsule is a low-mass, ball-jointed string segment. The
    /// tan bags are concentrated loads proportional to tributary bay area;
    /// the weighted tower tips carry the additional tower reactions. Gravity
    /// puts every member into tension. Orbit underneath (or mentally reflect
    /// the model) and the same force lines become masonry paths in compression.
    public static func gaudiFunicular(segmentsPerCable: Int = 5,
                                      loadScale: Float = 1,
                                      slack: Float = 1.12,
                                      linearDamping: Float = 1.8,
                                      angularDamping: Float = 2.6) -> PhysicsScene {
        var s = PhysicsScene(name: "gaudifunicular")
        s.settings.iterations = 40
        s.settings.alpha = 0.999
        s.settings.betaLin = 40_000
        s.settings.betaAng = 200
        s.settings.gamma = 1.0
        s.settings.lambdaMax = 20_000
        s.settings.gravity = -9.81
        s.settings.dt = 1.0 / 120.0
        s.settings.rigidLinearDamping = max(0, linearDamping)
        s.settings.rigidAngularDamping = max(0, angularDamping)
        // Fine model twine needs a millimetric contact skin. The meter-scale
        // default would make separated strings repel across a visible gap.
        s.settings.collisionMargin = 0.001

        let module = GaudiFunicularBlueprint.module
        let boardZ: Float = 12.0
        let cableRadius: Float = 0.006
        let nodeRadius: Float = 0.030
        let cableSegments = max(4, min(9, segmentsPerCable))
        let loadScale = max(0.05, loadScale)
        let slack = min(max(slack, 1.001), 1.4)

        let wood = F3(0.30, 0.14, 0.045)
        let iron = F3(0.17, 0.19, 0.20)
        let twine = F3(0.72, 0.38, 0.10)
        let towerTwine = F3(0.93, 0.57, 0.10)
        let knot = F3(0.90, 0.60, 0.19)
        let canvas = F3(0.55, 0.34, 0.16)

        @discardableResult
        func addVisualBody(size: F3, density: Float, position: F3,
                           rotation: Quat = Quat(real: 1, imag: .zero),
                           shape: BodyShape = .box, color: F3,
                           collisionEnabled: Bool = false,
                           friction: Float = 0.35) -> Int {
            let body = s.addBody(size: size, density: density, friction: 0,
                                 position: position, rotation: rotation,
                                 shape: shape, collisionEnabled: false)
            _ = s.addCollider(body: body, size: size, friction: friction,
                              localPosition: .zero, shape: shape,
                              collisionEnabled: collisionEnabled, isRendered: true,
                              renderColor: color)
            return body
        }

        func densityForSphere(mass: Float, radius: Float) -> Float {
            mass / (4.0 / 3.0 * .pi * radius * radius * radius)
        }

        @discardableResult
        func addNode(at position: F3, mass: Float, color: F3 = F3.zero) -> Int {
            addVisualBody(size: F3(repeating: 2 * nodeRadius),
                          density: densityForSphere(mass: mass, radius: nodeRadius),
                          position: position, shape: .sphere,
                          color: color == .zero ? knot : color)
        }

        func alignZ(to vector: F3) -> Quat {
            let d = normalize(vector)
            let z = F3(0, 0, 1)
            let cosine = min(max(dot(z, d), -1), 1)
            if cosine > 0.99999 { return Quat(real: 1, imag: .zero) }
            if cosine < -0.99999 { return Quat(angle: .pi, axis: F3(1, 0, 0)) }
            return Quat(angle: acos(cosine), axis: normalize(cross(z, d)))
        }

        // Complete cord branches incident to each knot. Cords tied into one
        // knot are a single topological bundle: letting discrete capsules in
        // sibling branches collide can force apart a connection that cannot
        // separate. Unrelated cords still use exact capsule contact, and a
        // cord keeps non-adjacent self-collision.
        var cablesAtKnot: [Int: [[Int]]] = [:]

        /// Add a visible, self-colliding ball-jointed cable. Adjacent links
        /// are excluded automatically by their joint; links from separate
        /// cords use exact capsule-capsule contact.
        func addCable(from bodyA: Int, localA: F3 = .zero, worldA: F3,
                      to bodyB: Int, localB: F3 = .zero, worldB: F3,
                      segments: Int, slackRatio: Float,
                      radius: Float = cableRadius,
                      color: F3? = nil) {
            let count = max(1, segments)
            let chord = worldB - worldA
            let chordLength = max(length(chord), 1e-4)
            // For a shallow parabolic cable, L/d ~= 1 + 8/3 (sag/d)^2.
            // This seeds the selected physical slack without prescribing the
            // final force line: the AVBD solve and authored loads find that.
            let sag = chordLength * sqrt(max(0, slackRatio - 1) * 3 / 8)
            var points: [F3] = []
            for i in 0...count {
                let t = Float(i) / Float(count)
                points.append(worldA + chord * t
                              + F3(0, 0, -4 * sag * t * (1 - t)))
            }

            var links: [Int] = []
            var spans: [Float] = []
            for i in 0..<count {
                let delta = points[i + 1] - points[i]
                let span = length(delta)
                let cylinderLength = max(0.01, span - 2 * radius)
                // Cotton thread is deliberately very light relative to the
                // shot bags, while retaining finite inertia for interaction.
                let link = addVisualBody(size: F3(cylinderLength, radius, 0),
                                         // Radius is smaller and resolution
                                         // higher than v1; density preserves
                                         // comparable total cord mass.
                                         density: 56,
                                         position: (points[i] + points[i + 1]) / 2,
                                         rotation: alignZ(to: delta),
                                         shape: .capsule, color: color ?? twine,
                                         collisionEnabled: true,
                                         friction: 0.28)
                links.append(link)
                spans.append(span)
            }

            s.addJoint(SceneJoint(bodyA: bodyA, bodyB: links[0],
                                  rA: localA,
                                  rB: F3(0, 0, -spans[0] / 2)))
            if links.count > 1 {
                for i in 1..<links.count {
                    s.addJoint(SceneJoint(bodyA: links[i - 1], bodyB: links[i],
                                          rA: F3(0, 0, spans[i - 1] / 2),
                                          rB: F3(0, 0, -spans[i] / 2)))
                }
            }
            s.addJoint(SceneJoint(bodyA: links[links.count - 1], bodyB: bodyB,
                                  rA: F3(0, 0, spans[spans.count - 1] / 2),
                                  rB: localB))
            cablesAtKnot[bodyA, default: []].append(links)
            cablesAtKnot[bodyB, default: []].append(links)
        }

        // The overhead board is the plan at foundation level. Thin beams
        // expose the modular grid without hiding the strings from above.
        let planLength = GaudiFunicularBlueprint.interiorLength
        let naveWidth = GaudiFunicularBlueprint.naveWidth
        let transeptWidth = GaudiFunicularBlueprint.transeptWidth
        for ix in 0...12 {
            let x = -planLength / 2 + Float(ix) * module
            let width = (7...9).contains(ix) ? transeptWidth : naveWidth
            _ = addVisualBody(size: F3(0.055, width + 0.18, 0.07), density: 0,
                              position: F3(x, 0, boardZ + 0.20), color: wood)
        }
        let naveY: [Float] = [-3, -2, -1, 1, 2, 3].map { Float($0) * module }
        for y in naveY {
            _ = addVisualBody(size: F3(planLength + 0.18, 0.055, 0.07), density: 0,
                              position: F3(0, y, boardZ + 0.20), color: wood)
        }
        for y in [-4, 4].map({ Float($0) * module }) {
            _ = addVisualBody(size: F3(2 * module + 0.18, 0.055, 0.07), density: 0,
                              position: F3(2 * module, y, boardZ + 0.20), color: wood)
        }

        struct Station {
            var anchor: Int
            var capital: Int
            var anchorPosition: F3
            var capitalPosition: F3
        }
        var stations: [[Station]] = []
        for ix in 0...12 {
            var row: [Station] = []
            let x = -planLength / 2 + Float(ix) * module
            for y in naveY {
                let anchorPosition = F3(x, y, boardZ)
                let anchor = addVisualBody(size: F3(repeating: 0.12), density: 0,
                                           position: anchorPosition,
                                           shape: .sphere, color: iron)
                // Gaudí's Sagrada Família columns are regular in longitudinal
                // section but lean transversely into the load path.
                let isInner = abs(y) < 1.5 * module
                let supportedHeight = isInner
                    ? GaudiFunicularBlueprint.centralNaveHeightMeters
                    : GaudiFunicularBlueprint.sideNaveHeightMeters
                let capitalDrop = supportedHeight
                    * GaudiFunicularBlueprint.verticalScale * 0.62
                let capitalPosition = F3(x, y * (isInner ? 0.72 : 0.80),
                                         boardZ - capitalDrop)
                let capital = addNode(at: capitalPosition, mass: 0.012)
                addCable(from: anchor, worldA: anchorPosition,
                         to: capital, worldB: capitalPosition,
                         segments: cableSegments, slackRatio: 1.015)
                row.append(Station(anchor: anchor, capital: capital,
                                   anchorPosition: anchorPosition,
                                   capitalPosition: capitalPosition))
            }
            stations.append(row)
        }

        func addLoadCell(_ corners: [Station], xIndex: Int, yIndex: Int,
                         tributaryModules: Float,
                         heightMeters: Float,
                         includeBag: Bool = true) {
            let center = corners.reduce(F3.zero) { $0 + $1.capitalPosition }
                / Float(corners.count)
            let vaultPosition = F3(center.x, center.y,
                                   boardZ - heightMeters
                                       * GaudiFunicularBlueprint.verticalScale)
            let vault = addNode(at: vaultPosition, mass: 0.018)
            for corner in corners {
                addCable(from: corner.capital, worldA: corner.capitalPosition,
                         to: vault, worldB: vaultPosition,
                         segments: cableSegments, slackRatio: slack)
            }

            // The two central crossing reactions are carried by the Jesus
            // tower's own weighted bundle below. A second roof bag here would
            // occupy the tower cone and create a geometrically impossible
            // box/cord intersection in the hanging apparatus.
            guard includeBag else { return }

            // One bag represents the tributary masonry/roof load of the bay.
            // The central 15 m nave is two modules wide; crossing bags add the
            // four main central-tower reactions.
            let load = 0.105 * loadScale * tributaryModules
            let bagWidth = 0.14 + 0.025 * sqrt(load / 0.16)
            let bagHeight = 0.19 + 0.045 * sqrt(load / 0.16)
            let bagSize = F3(bagWidth, bagWidth * 0.78, bagHeight)
            let bagPosition = vaultPosition + F3(0, 0, -0.38 - bagHeight / 2)
            let bagDensity = load / (bagSize.x * bagSize.y * bagSize.z)
            let bag = addVisualBody(size: bagSize, density: bagDensity,
                                    position: bagPosition, color: canvas,
                                    collisionEnabled: true, friction: 0.48)
            let tieTop = bagPosition + F3(0, 0, bagHeight / 2)
            addCable(from: vault, worldA: vaultPosition,
                     to: bag, localB: F3(0, 0, bagHeight / 2), worldB: tieTop,
                     segments: 1, slackRatio: 1.001, radius: cableRadius * 0.72)
            _ = xIndex
            _ = yIndex
        }

        // Five nave bands: four 7.5 m aisles and one 15 m central nave.
        for ix in 0..<12 {
            for iy in 0..<5 {
                let corners = [stations[ix][iy], stations[ix + 1][iy],
                               stations[ix][iy + 1], stations[ix + 1][iy + 1]]
                let tributary: Float = iy == 2 ? 2 : 1
                let isCrossing = (7...8).contains(ix) && (1...3).contains(iy)
                let isApse = ix >= 10 && iy == 2
                let height: Float
                if isApse {
                    height = GaudiFunicularBlueprint.apseHeightMeters
                } else if isCrossing {
                    height = GaudiFunicularBlueprint.crossingHeightMeters
                } else if iy == 2 {
                    height = GaudiFunicularBlueprint.centralNaveHeightMeters
                } else {
                    height = GaudiFunicularBlueprint.sideNaveHeightMeters
                }
                addLoadCell(corners, xIndex: ix, yIndex: iy,
                            tributaryModules: tributary, heightMeters: height,
                            includeBag: !(isCrossing && iy == 2))
            }
        }

        // Extend the crossing from the 45 m nave body to its documented
        // 60 m transept width: two extra bays on each side, x = 0.9...2.7 m
        // in model-module coordinates.
        for side in [-1, 1] {
            var outer: [Station] = []
            let y = Float(side) * transeptWidth / 2
            for ix in 7...9 {
                let x = -planLength / 2 + Float(ix) * module
                let anchorPosition = F3(x, y, boardZ)
                let anchor = addVisualBody(size: F3(repeating: 0.12), density: 0,
                                           position: anchorPosition,
                                           shape: .sphere, color: iron)
                let capitalPosition = F3(x, y * 0.82,
                                         boardZ - 0.62
                                             * GaudiFunicularBlueprint.sideNaveHeightMeters
                                             * GaudiFunicularBlueprint.verticalScale)
                let capital = addNode(at: capitalPosition, mass: 0.012)
                addCable(from: anchor, worldA: anchorPosition,
                         to: capital, worldB: capitalPosition,
                         segments: cableSegments, slackRatio: 1.015)
                outer.append(Station(anchor: anchor, capital: capital,
                                     anchorPosition: anchorPosition,
                                     capitalPosition: capitalPosition))
            }
            let innerY = side < 0 ? 0 : naveY.count - 1
            for localX in 0..<2 {
                let ix = 7 + localX
                let corners = side < 0
                    ? [outer[localX], outer[localX + 1],
                       stations[ix][innerY], stations[ix + 1][innerY]]
                    : [stations[ix][innerY], stations[ix + 1][innerY],
                       outer[localX], outer[localX + 1]]
                addLoadCell(corners, xIndex: ix, yIndex: side,
                            tributaryModules: 1.0,
                            heightMeters: GaudiFunicularBlueprint.crossingHeightMeters)
            }
        }

        // The tower group is what makes the inverted force model read as the
        // Sagrada Família rather than a level roof net. Each tower tip is a
        // weighted suspension point with its own foundation-pin and base
        // rings. Each foundation string drops directly from its matching pin,
        // so neighboring towers cannot start mutually threaded through the
        // roof net. The conical lower bundle carries the tower load to its tip.
        // Heights are the Basilica's authored values, compressed by one
        // consistent vertical model scale.
        func addTower(at xy: F3, heightMeters: Float,
                      baseHeightMeters: Float, baseRadius: Float,
                      load: Float, supports: Int = 4) {
            let tipPosition = F3(xy.x, xy.y,
                                 boardZ - heightMeters
                                     * GaudiFunicularBlueprint.verticalScale)
            let tip = addNode(at: tipPosition, mass: 0.024, color: towerTwine)
            let baseZ = boardZ - baseHeightMeters
                * GaudiFunicularBlueprint.verticalScale
            for supportIndex in 0..<supports {
                let angle = 2 * Float.pi * Float(supportIndex) / Float(supports)
                    + 0.25 * Float.pi
                let radial = F3(cos(angle), sin(angle), 0) * baseRadius
                let anchorPosition = F3(xy.x + radial.x, xy.y + radial.y,
                                        boardZ)
                let anchor = addVisualBody(size: F3(repeating: 0.10), density: 0,
                                           position: anchorPosition,
                                           shape: .sphere, color: iron)
                let basePosition = F3(xy.x + radial.x, xy.y + radial.y, baseZ)
                let base = addNode(at: basePosition, mass: 0.020,
                                   color: towerTwine)
                addCable(from: anchor, worldA: anchorPosition,
                         to: base, worldB: basePosition,
                         segments: cableSegments, slackRatio: 1.015,
                         radius: cableRadius, color: twine)
                addCable(from: base, worldA: basePosition,
                         to: tip, worldB: tipPosition,
                         segments: min(10, cableSegments + 2), slackRatio: 1.008,
                         radius: cableRadius * 1.12, color: towerTwine)
            }
            let actualLoad = load * loadScale
            let bagWidth: Float = 0.20 + 0.025 * sqrt(actualLoad)
            let bagHeight: Float = 0.28 + 0.035 * sqrt(actualLoad)
            let bagSize = F3(bagWidth, bagWidth * 0.80, bagHeight)
            let bagPosition = tipPosition + F3(0, 0, -0.34 - bagHeight / 2)
            let density = actualLoad / (bagSize.x * bagSize.y * bagSize.z)
            let bag = addVisualBody(size: bagSize, density: density,
                                    position: bagPosition, color: canvas,
                                    collisionEnabled: true, friction: 0.48)
            addCable(from: tip, worldA: tipPosition,
                     to: bag, localB: F3(0, 0, bagHeight / 2),
                     worldB: bagPosition + F3(0, 0, bagHeight / 2),
                     segments: 1, slackRatio: 1.001,
                     radius: cableRadius * 0.78, color: towerTwine)
        }

        let crossingX = 2 * module
        // Tower of Jesus, then the four Evangelists around it.
        addTower(at: F3(crossingX, 0, 0),
                 heightMeters: GaudiFunicularBlueprint.jesusTowerHeightMeters,
                 baseHeightMeters: GaudiFunicularBlueprint.crossingHeightMeters,
                 baseRadius: 0.74,
                 load: 2.8, supports: 8)
        for dx in [-0.62, 0.62] as [Float] {
            for dy in [-0.64, 0.64] as [Float] {
                addTower(at: F3(crossingX + dx, dy, 0),
                         heightMeters: GaudiFunicularBlueprint.evangelistTowerHeightMeters,
                         baseHeightMeters: GaudiFunicularBlueprint.centralNaveHeightMeters,
                         baseRadius: 0.22,
                         load: 1.35, supports: 5)
            }
        }
        // Tower of Mary rises over the apse.
        addTower(at: F3(4.55, 0, 0),
                 heightMeters: GaudiFunicularBlueprint.maryTowerHeightMeters,
                 baseHeightMeters: GaudiFunicularBlueprint.apseHeightMeters,
                 baseRadius: 0.42,
                 load: 1.55, supports: 6)

        // Twelve Apostle bell towers: four at each monumental façade. Their
        // stepped 98.5...120 m heights reproduce the recognizable clusters.
        let apostleHeights: [Float] = [98.5, 105, 112.5, 120]
        let gloryY: [Float] = [-1.65, -0.62, 0.62, 1.65]
        for i in 0..<4 {
            addTower(at: F3(-5.05 + 0.10 * Float(i % 2), gloryY[i], 0),
                     heightMeters: apostleHeights[i],
                     baseHeightMeters: GaudiFunicularBlueprint.sideNaveHeightMeters,
                     baseRadius: 0.16, load: 0.82, supports: 4)
        }
        let façadeX: [Float] = [0.82, 1.48, 2.12, 2.78]
        for side in [-1, 1] {
            for i in 0..<4 {
                addTower(at: F3(façadeX[i], Float(side) * 3.25, 0),
                         heightMeters: apostleHeights[i],
                         baseHeightMeters: GaudiFunicularBlueprint.sideNaveHeightMeters,
                         baseRadius: 0.16, load: 0.82, supports: 4)
            }
        }

        // Sibling cords tied into one knot are one collision-topology bundle.
        // Exclude only cross-cord pairs in that bundle; non-adjacent capsules
        // within each cord and every unrelated cord/bag pair remain active.
        for branches in cablesAtKnot.values where branches.count > 1 {
            for i in 0..<(branches.count - 1) {
                for j in (i + 1)..<branches.count {
                    for linkA in branches[i] {
                        for linkB in branches[j] {
                            s.addCollisionExclusion(bodyA: linkA, bodyB: linkB)
                        }
                    }
                }
            }
        }

        s.settings.cameraDistance = 18.5
        s.settings.cameraTargetZ = 6.1
        s.settings.cameraAzimuth = -0.98
        s.settings.cameraElevation = -0.03
        return s
    }
}
