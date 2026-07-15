// Arachne-15: parametric eight-legged carrier for an iPhone 15 Pro.
// Units: millimetres.  The assembly uses conservative envelopes for purchased
// parts; printable parts are selected with -D 'PART="..."'.

$fn = 48;
PART = is_undef(PART) ? "assembly" : PART;
SHOW_KEEP_OUTS = is_undef(SHOW_KEEP_OUTS) ? true : SHOW_KEEP_OUTS;

// Apple iPhone 15 Pro, Apple dimensional drawing / tech specs.
phone_x = 146.60;
phone_y = 70.60;
phone_body_z = 8.25;
phone_corner_r = 12.0;
phone_mass_g = 187;
phone_clearance = 0.60;          // per side, bare phone
camera_extra_z = 5.46;           // conservative rear keep-out below body

// ROBOTIS XC330-M288-T official envelope.
servo_w = 20.0;
servo_h = 34.0;
servo_d = 29.0;                 // 26 mm body; 29 mm audited STL envelope
servo_mass_g = 23;
servo_stall_torque_nm_5v = 0.93;
frame_mount_spacing = 16.0;     // FPX330-S102 two-hole mounting pair
horn_hub_clearance = 8.2;       // clears official 8 mm hub opening

// Chassis and top-inserted portrait dock.  Robot forward is +X.  The phone's
// rear cameras face +X and its screen faces -X.
chassis_x = 178;
chassis_y = 112;
chassis_z = 5;
chassis_corner_r = 18;
chassis_window_x = 136;
chassis_window_y = 58;

dock_x = 24;
dock_y = 84;
dock_base_z = 8;
dock_wall = 3;
dock_guide_h = 94;
dock_phone_bottom_z = 2;
phone_cavity_thickness = phone_body_z + 2 * phone_clearance;
phone_cavity_width = phone_y + 2 * phone_clearance;

// Nominal flat-ground stance.
body_floor_z = 76;
hip_axis_z = 98;
hip_xs = [-66, -22, 22, 66];
hip_y = 63;
coxa_length = 50;
tibia_length = 105;
tibia_pitch_deg = 65;
foot_radius = 4.5;

// Power/electronics envelopes, intentionally not powered from the iPhone.
battery_size = [105, 35, 18];     // typical 2S 2200 mAh pack envelope
bridge_size = [55, 25, 12];       // ESP32 + TTL bridge + distribution PCB
bec_size = [55, 30, 14];          // 5 V / 20 A-class BEC envelope

assert(phone_cavity_thickness >= phone_body_z + 1.0,
       "phone thickness cavity needs at least 0.5 mm clearance per side");
assert(phone_cavity_width >= phone_y + 1.0,
       "phone cavity needs at least 0.5 mm clearance per side");
assert(chassis_x >= dock_x + 40 && chassis_y >= dock_y + 20,
       "dock leaves insufficient structural rail around phone");

module rounded_prism(size = [10, 10, 2], r = 2) {
    x = size[0]; y = size[1]; z = size[2];
    hull() {
        for (px = [-x / 2 + r, x / 2 - r])
            for (py = [-y / 2 + r, y / 2 - r])
                translate([px, py, 0]) cylinder(r = r, h = z);
    }
}

module hole_pattern_pcd12(h = 10, diameter = 2.2) {
    for (a = [0 : 90 : 270])
        translate([6 * cos(a), 6 * sin(a), -0.5])
            cylinder(d = diameter, h = h + 1);
}

module chassis() {
    difference() {
        union() {
            rounded_prism([chassis_x, chassis_y, chassis_z], chassis_corner_r);
            // Eight local load spreaders for the hip brackets.
            for (x = hip_xs)
                for (side = [-1, 1])
                    translate([x, side * 49, 0])
                        rounded_prism([28, 18, chassis_z + 2], 5);
        }
        // Open centre: camera clearance, cooling, and lower electronics access.
        translate([0, 0, -0.5])
            rounded_prism([chassis_window_x, chassis_window_y,
                           chassis_z + 3], 12);
        // Two M2.2 bracket clearances per hip. Final hardware uses FPX330 frames.
        for (x = hip_xs)
            for (side = [-1, 1])
                for (dx = [-frame_mount_spacing / 2, frame_mount_spacing / 2])
                    translate([x + dx, side * 49, -0.5])
                        cylinder(d = 2.2, h = chassis_z + 3);
        // Portrait-dock M3 attachment holes land on the centre side rails.
        for (x = [-9, 9])
            for (y = [-38, 38])
                translate([x, y, -0.5]) cylinder(d = 3.2, h = chassis_z + 2);
    }
}

module phone_tray() {
    difference() {
        union() {
            // Bottom bridge transfers the phone load directly into the chassis.
            rounded_prism([dock_x, dock_y, dock_base_z], 4);
            // Edge guides leave both broad faces open for camera, cooling, and UI.
            for (side = [-1, 1])
                translate([0, side * (phone_cavity_width / 2 + dock_wall / 2), 0])
                    rounded_prism([dock_x, dock_wall, dock_guide_h], 1.2);
            // Screen-side lower buttress; the camera-facing side remains open.
            translate([-(phone_cavity_thickness / 2 + dock_wall / 2), 0, 0])
                rounded_prism([dock_wall, phone_cavity_width + 2 * dock_wall, 30], 1.2);
        }
        // Bare-phone slot, open upward. TPU tape is applied after printing.
        translate([0, 0, dock_phone_bottom_z + 5])
            cube([phone_cavity_thickness, phone_cavity_width, 10], center = true);
        // Four M3 chassis screws outside the phone-thickness slot.
        for (x = [-9, 9])
            for (y = [-38, 38])
                translate([x, y, -0.5]) cylinder(d = 3.2, h = dock_base_z + 2);
        // Two low strap paths retain the phone without entering the camera field.
        for (z = [38, 74])
            translate([0, 0, z])
                cube([dock_x + 2, 5, 3], center = true);
    }
}

module retainer_clip() {
    // Optional TPU-backed wedge for either vertical edge guide.  A silicone
    // safety strap through the dock slots remains mandatory during walking.
    difference() {
        union() {
            rounded_prism([14, 9, 3], 1.5);
            translate([0, 3, -4]) cube([14, 3, 8], center = true);
            translate([0, -3, 1.5]) rounded_prism([14, 3, 2.5], 1);
        }
        translate([0, 0, -5]) cylinder(d = 2.2, h = 12);
    }
}

module coxa_link() {
    difference() {
        union() {
            hull() {
                cylinder(r = 9, h = 4);
                translate([coxa_length, 0, 0]) cylinder(r = 9, h = 4);
            }
            // Shallow centre rib resists yaw-plane bending.
            translate([10, -2, 4]) cube([coxa_length - 20, 4, 3]);
        }
        translate([0, 0, -0.5]) cylinder(d = horn_hub_clearance, h = 8);
        hole_pattern_pcd12(h = 8);
        translate([coxa_length, 0, -0.5]) {
            cylinder(d = horn_hub_clearance, h = 8);
            hole_pattern_pcd12(h = 8);
        }
        // Weight-relief slots leave two continuous outer load paths.
        translate([coxa_length / 2, 0, -0.5])
            rounded_prism([coxa_length - 24, 7, 6], 3);
    }
}

module tibia_link() {
    difference() {
        union() {
            hull() {
                cylinder(r = 8, h = 6);
                translate([tibia_length, 0, 0]) cylinder(r = 7, h = 6);
            }
            translate([12, -2, 6]) cube([tibia_length - 26, 4, 3]);
        }
        translate([0, 0, -0.5]) {
            cylinder(d = horn_hub_clearance, h = 10);
            hole_pattern_pcd12(h = 10);
        }
        translate([tibia_length, 0, -0.5]) cylinder(d = 3.2, h = 10);
        translate([tibia_length / 2, 0, -0.5])
            rounded_prism([tibia_length - 28, 7, 8], 3);
    }
}

module battery_cradle() {
    bx = battery_size[0] + 6;
    by = battery_size[1] + 5;
    difference() {
        union() {
            for (side = [-1, 1])
                translate([0, side * by / 2, 0])
                    rounded_prism([bx, 5, 4], 2);
            for (x = [-bx / 2 + 5, bx / 2 - 5])
                translate([x, 0, 0]) rounded_prism([10, by, 4], 3);
            for (x = [-bx / 2 + 7, bx / 2 - 7])
                for (side = [-1, 1])
                    translate([x, side * (by / 2 + 5), 0])
                        rounded_prism([14, 10, 4], 3);
        }
        for (x = [-bx / 2 + 7, bx / 2 - 7])
            for (side = [-1, 1])
                translate([x, side * (by / 2 + 5), -0.5])
                    cylinder(d = 3.2, h = 6);
    }
}

module foot_pad() {
    difference() {
        scale([1.25, 1, 0.55]) sphere(r = 7);
        translate([0, 0, 2.5]) cylinder(d = 3.2, h = 12);
    }
}

module xc330_visual(vertical = true) {
    color([0.12, 0.15, 0.19])
        if (vertical) {
            translate([-servo_w / 2, -servo_d / 2, -servo_h])
                cube([servo_w, servo_d, servo_h]);
            color([0.75, 0.76, 0.78]) cylinder(d = 10, h = 3);
        } else {
            translate([-servo_d / 2, -servo_w / 2, -servo_h / 2])
                cube([servo_d, servo_w, servo_h]);
            color([0.75, 0.76, 0.78])
                rotate([90, 0, 0]) cylinder(d = 10, h = 3);
        }
}

module beam_between(p1, p2, radius, colour = [0.15, 0.35, 0.42]) {
    v = p2 - p1;
    len = norm(v);
    axis = [-v[1], v[0], 0];
    angle = acos(v[2] / len);
    color(colour)
        translate(p1)
            rotate(a = angle, v = norm(axis) < 0.001 ? [1, 0, 0] : axis)
                cylinder(r = radius, h = len);
}

function outward_angle(x, side) =
    side * (x < -44 ? 140 : x < 0 ? 110 : x < 44 ? 70 : 40);

module leg_assembly(x, side, index) {
    a = outward_angle(x, side);
    hip = [x, side * hip_y, hip_axis_z];
    knee = hip + [coxa_length * cos(a), coxa_length * sin(a), 0];
    foot = knee + [tibia_length * cos(tibia_pitch_deg) * cos(a),
                   tibia_length * cos(tibia_pitch_deg) * sin(a),
                   -tibia_length * sin(tibia_pitch_deg)];

    translate(hip) xc330_visual(true);
    beam_between(hip, knee, 5.5, index % 2 == 0 ? [0.13, 0.38, 0.46]
                                                      : [0.18, 0.45, 0.35]);
    translate(knee) rotate([0, 0, a + 90]) xc330_visual(false);
    beam_between(knee, foot, 5.2, index % 2 == 0 ? [0.26, 0.57, 0.63]
                                                        : [0.31, 0.64, 0.48]);
    color([0.08, 0.09, 0.10]) translate(foot) foot_pad();
}

module orient_phone(phone_bottom) {
    // Local Apple drawing axes: long X, short Y, thickness Z.  World axes:
    // thickness -> +X, short side -> -Y, long side -> +Z.
    multmatrix([[0, 0, 1, -phone_body_z / 2],
                [0, -1, 0, 0],
                [1, 0, 0, phone_bottom],
                [0, 0, 0, 1]]) children();
}

module phone_visual() {
    phone_bottom = body_floor_z + chassis_z + dock_phone_bottom_z;
    color([0.28, 0.31, 0.34, 0.93])
        orient_phone(phone_bottom)
            rounded_prism([phone_x, phone_y, phone_body_z], phone_corner_r);
    // Rear camera keep-out projects toward robot-forward (+X).
    if (SHOW_KEEP_OUTS)
        color([0.85, 0.15, 0.18, 0.35])
            orient_phone(phone_bottom)
                translate([phone_x / 2 - 21, phone_y / 2 - 18, phone_body_z])
                    rounded_prism([42, 36, camera_extra_z], 7);
}

module camera_forward_visual() {
    camera_z = body_floor_z + chassis_z + dock_phone_bottom_z + phone_x - 21;
    camera_y = -phone_y / 2 + 18;
    color([0.15, 0.55, 1.0, 0.65]) {
        beam_between([phone_body_z / 2 + camera_extra_z, camera_y, camera_z],
                     [72, camera_y, camera_z], 1.2, [0.15, 0.55, 1.0]);
        translate([72, camera_y, camera_z]) rotate([0, 90, 0])
            cylinder(r1 = 5, r2 = 0, h = 13);
    }
}

module electronics_visual() {
    color([0.20, 0.65, 0.28, 0.8])
        translate([24, 0, body_floor_z - 20])
            rounded_prism(bridge_size, 3);
    color([0.18, 0.35, 0.75, 0.8])
        translate([-34, 0, body_floor_z - 22])
            rounded_prism(bec_size, 3);
    color([0.35, 0.25, 0.12, 0.9])
        translate([0, 0, body_floor_z - 29])
            rounded_prism(battery_size, 4);
}

module assembly() {
    color([0.13, 0.14, 0.16]) translate([0, 0, body_floor_z]) chassis();
    color([0.92, 0.52, 0.10])
        translate([0, 0, body_floor_z + chassis_z]) phone_tray();
    phone_visual();
    camera_forward_visual();
    electronics_visual();
    color([0.18, 0.18, 0.20])
        translate([0, 0, body_floor_z - 28]) battery_cradle();

    for (side = [-1, 1])
        for (i = [0 : len(hip_xs) - 1])
            leg_assembly(hip_xs[i], side, i + (side > 0 ? 0 : 4));

    // Two removable edge wedges; the walking build also uses a silicone strap.
    for (side = [-1, 1])
        color([0.95, 0.62, 0.16])
            translate([-8, side * (phone_cavity_width / 2 + dock_wall),
                       body_floor_z + chassis_z + 78])
                rotate([90, 0, side > 0 ? 0 : 180]) retainer_clip();

    // Ground datum.
    color([0.7, 0.72, 0.75, 0.25])
        translate([0, 0, -1]) cube([430, 350, 1], center = true);
}

echo(str("Arachne-15 vertical dock cavity: ", phone_cavity_thickness,
         " thick x ", phone_cavity_width, " wide mm; bare phone: ", phone_x,
         " x ", phone_y, " x ", phone_body_z, " mm; rear camera faces +X"));
echo(str("Nominal footprint approx 430 x 350 mm; chassis ground clearance ",
         body_floor_z, " mm"));

if (PART == "assembly") assembly();
else if (PART == "chassis") chassis();
else if (PART == "phone_tray") phone_tray();
else if (PART == "retainer_clip") retainer_clip();
else if (PART == "coxa_link") coxa_link();
else if (PART == "tibia_link") tibia_link();
else if (PART == "battery_cradle") battery_cradle();
else if (PART == "foot_pad") foot_pad();
else assert(false, str("Unknown PART: ", PART));
