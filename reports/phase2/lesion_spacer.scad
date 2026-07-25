// =====================================================================
//  lesion_spacer.scad
//  North India Dermatology Photo Data Collection Study (MNKAI-III)
//  Parametric clip-on standoff for smartphone clinical skin photography.
//
//  PURPOSE
//    Holds the phone's MAIN rear camera at a fixed, repeatable distance
//    from the skin, so that scale, focus and framing are the same across
//    operators, days and phones.
//
//  PRINTING
//    The model is split into two parts so that BOTH print WITHOUT SUPPORTS:
//      part = "cone"      -> the standoff. Print large (skin) end DOWN on
//                            the bed. All outer walls lean inwards by only
//                            ~10-20 degrees from vertical, which is
//                            self-supporting on any FDM printer.
//      part = "clip"      -> the phone band + camera collar. Prints flat,
//                            screen-side face DOWN, socket ring UP. The back
//                            plate bridges the phone pocket between the two
//                            side walls - a normal, fully supported bridge.
//      part = "testring"  -> a 10 mm tall slice of the sleeve only. PRINT THIS
//                            FIRST and check it slides onto the phone before
//                            committing filament to the full part.
//      part = "assembly"  -> preview only (do NOT slice this).
//    Use a BRIM on both parts.
//
//  REQUIRES OpenSCAD 2019.05 or newer (uses rotate_extrude(angle=...)).
//  UNITS: millimetres.
//  See MOUNT_SPEC.md (same folder) for measuring, fitting, calibration,
//  cleaning and the cost table.
// =====================================================================


// =====================================================================
//  1. WHAT TO PRINT
// =====================================================================

part = "cone";      // "cone" | "clip" | "testring" | "assembly"


// =====================================================================
//  2. THE ONE NUMBER THAT DEFINES THE STUDY GEOMETRY
// =====================================================================

// Distance from the FRONT OF THE CAMERA LENS to the SKIN, in mm.
// Default 120 mm. Adjustable range 100-150 mm.
// Do NOT go below ~100 mm: phone main cameras generally cannot focus
// closer than about 10-15 cm. If images come back soft at 100 mm on your
// phone, increase this to 130-150 mm and re-run the calibration.
// Whatever value you choose, USE THE SAME VALUE FOR EVERY UNIT AT BOTH
// SITES and record it in the study log.
working_distance = 120;


// =====================================================================
//  3. PHONE MEASUREMENTS  (take these with vernier calipers, phone CASE OFF)
// =====================================================================

// Width of the phone across the short edge (the edge nearest the camera).
// Measure the actual body, not the screen.
phone_width = 76.0;

// Thickness of the phone body, measured away from the camera bump
// (i.e. on the flat part of the back).
phone_thickness = 8.5;

// Lens centre position, measured ON THE BACK OF THE PHONE, phone held
// with the camera at the TOP, looking at the back:
//   camera_offset_x = distance from the LEFT edge to the CENTRE of the
//                     MAIN lens (the big one, usually the widest barrel).
//   camera_offset_y = distance from the TOP edge down to the CENTRE of
//                     the MAIN lens.
// Identify the MAIN lens first (see MOUNT_SPEC.md section "Fitting to a
// specific phone"). Never centre this on the ultrawide or the small
// 2 MP macro lens.
camera_offset_x = 14.0;
camera_offset_y = 16.0;

// How far the front glass of the MAIN lens sits ABOVE the flat back of
// the phone (the camera bump height at that lens). Measure with the
// calipers' depth probe. Enter 0 for a perfectly flush camera.
lens_bump = 2.5;

// The raised camera island (the whole block that carries the lenses).
// A pocket is cut in the clip plate so the plate still sits flat on the
// phone back. Measure the island's overall size and how far it stands
// proud. Set island_pocket = false for a flush-backed phone.
island_pocket  = true;
island_w       = 34.0;   // island size across the phone width (x)
island_l       = 34.0;   // island size along the phone length (y)
island_depth   = 3.0;    // how far the island stands proud of the back
island_ctr_x   = 14.0;   // island centre, from LEFT edge  (same datum as camera_offset_x)
island_ctr_y   = 18.0;   // island centre, from TOP edge   (same datum as camera_offset_y)


// =====================================================================
//  4. OPTICAL / FRAMING PARAMETERS
// =====================================================================

// Clear hole through the clip plate over the lens. Must be larger than
// the lens barrel or you will see a dark ring at the edge of the frame.
// Measure the outer diameter of the MAIN lens barrel and add ~6 mm.
aperture_diameter = 26.0;

// Diameter of the opening that sits ON THE SKIN. This is what the camera
// actually sees: everything outside this circle is blocked by the cone.
// It therefore fixes the field of view, which is useful - but it also
// means a small lesion will occupy a small part of the frame. See
// MOUNT_SPEC.md, "Limitations".
skin_aperture_diameter = 90.0;


// =====================================================================
//  5. STRUCTURE / PRINT PARAMETERS
// =====================================================================

wall           = 2.4;    // wall thickness (= 3 perimeters at 0.4 mm nozzle)
fit            = 0.4;    // clearance around the phone body (looser = 0.6)
fit_joint      = 0.25;   // clearance at the cone-to-clip push fit
                         //   too tight -> won't seat; too loose -> wobbles

// Soft contact rim at the skin end: a WIDE FLAT flange, not a knife edge.
// Wide and flat spreads the load so the spacer does not press or blanch
// the skin (blanching changes lesion colour and ruins the image).
// A self-adhesive foam or silicone strip is stuck to this face - see
// MOUNT_SPEC.md.
rim_thickness  = 3.0;    // flange thickness (z)
rim_width      = 6.0;    // how far the flange sticks out radially
rim_chamfer    = 4.0;    // height of the chamfer blending flange into cone

// Cut-away side windows: let ambient light onto the lesion and let the
// operator see what is being photographed.
n_windows      = 3;      // number of windows (3 leaves 3 sturdy ribs)
rib_deg        = 26;     // angular width of each rib, degrees
window_bottom  = 22;     // window starts this far above the skin plane
window_top_gap = 26;     // window ends this far below the phone end

// Card / ruler slot at the SKIN end. The colour-reference card and the
// adhesive ruler slide in through these channels and lie flat ON THE
// SKIN, in the same plane as the lesion.
card_slot_w    = 26.0;   // slot width  - set to your card width + 2 mm
card_slot_t    = 1.6;    // slot height - set to card thickness + 0.8 mm
card_slot_angles = [0, 180];   // one channel for the card, one for the ruler
card_guide_h   = 4.0;    // little guide walls on the flange either side

// Push-fit joint between cone and clip.
socket_h       = 8.0;    // depth of the socket ring on the clip

$fn = 96;


// =====================================================================
//  6. DERIVED VALUES  (do not edit)
// =====================================================================

clip_plate_t = wall + fit;                       // effective plate thickness
standoff_h   = working_distance - clip_plate_t + lens_bump;

skin_r    = skin_aperture_diameter / 2;
spigot_id = aperture_diameter;
spigot_od = aperture_diameter + 2 * wall;
spigot_r  = spigot_od / 2;

cam_x = -phone_width / 2 + camera_offset_x;      // lens centre, clip frame
cam_y = camera_offset_y;
isl_x = -phone_width / 2 + island_ctr_x;
isl_y = island_ctr_y;

clip_band_h = max(38, camera_offset_y + aperture_diameter / 2 + 3 * wall + 8,
                      island_ctr_y + island_l / 2 + 2 * wall + 4);

echo(str("Standoff cone height  = ", standoff_h, " mm"));
echo(str("Working distance      = ", working_distance, " mm (lens to skin)"));
echo(str("Skin opening          = ", skin_aperture_diameter, " mm across"));
echo(str("Clip band length      = ", clip_band_h, " mm along the phone"));


// =====================================================================
//  7. MODULES
// =====================================================================

// --- hollow tapered standoff shell -----------------------------------
module cone_shell() {
    difference() {
        cylinder(h = standoff_h, r1 = skin_r + wall, r2 = spigot_r);
        translate([0, 0, -1])
            cylinder(h = standoff_h + 2, r1 = skin_r, r2 = spigot_id / 2);
    }
}

// --- wide flat contact flange at the skin end ------------------------
module contact_flange() {
    difference() {
        union() {
            cylinder(h = rim_thickness, r = skin_r + wall + rim_width);
            translate([0, 0, rim_thickness])
                cylinder(h = rim_chamfer,
                         r1 = skin_r + wall + rim_width,
                         r2 = skin_r + wall);
        }
        translate([0, 0, -1])
            cylinder(h = rim_thickness + rim_chamfer + 2, r = skin_r);
    }
}

// --- pie-slice cutter used to make the side windows ------------------
module window_cutter(ang, z1, z2, rmax) {
    rotate_extrude(angle = ang, $fn = 160)
        translate([0, z1]) square([rmax, z2 - z1]);
}

module side_windows() {
    step = 360 / n_windows;
    for (i = [0 : n_windows - 1])
        rotate([0, 0, i * step + rib_deg / 2])
            window_cutter(step - rib_deg,
                          window_bottom,
                          standoff_h - window_top_gap,
                          skin_r + wall + rim_width + 10);
}

// --- channels for the colour card and the ruler ----------------------
module card_channels() {
    for (a = card_slot_angles)
        rotate([0, 0, a])
            translate([-card_slot_w / 2, 0, -0.01])
                cube([card_slot_w,
                      skin_r + wall + rim_width + 10,
                      card_slot_t + 0.01]);
}

// --- small guide walls so the card goes in straight ------------------
module card_guides() {
    for (a = card_slot_angles)
        rotate([0, 0, a])
            for (s = [-1, 1])
                translate([s * (card_slot_w / 2 + wall / 2) - wall / 2,
                           skin_r + wall - 1,
                           0])
                    cube([wall, rim_width + 1, card_slot_t + card_guide_h]);
}

// --- the standoff, complete ------------------------------------------
module cone_part() {
    difference() {
        union() {
            cone_shell();
            contact_flange();
            card_guides();
        }
        side_windows();
        card_channels();
    }
}

// --- phone band clip + camera collar ---------------------------------
// A U-section sleeve that slides down over the TOP EDGE of the phone,
// carrying a back plate with the optical hole and the socket ring.
//
// Local frame: phone back surface = z 0, phone body fills z -phone_thickness..0
//              x = across the phone, y = along the phone from the top edge.
//
// y0 is normally 0 (the phone's top edge). If the lens sits very close to
// the top edge, the sleeve is automatically extended past it so the socket
// ring is always fully supported by plate material underneath.
//
// NOTE: on phones whose main lens is very close to a SIDE edge, the collar
// will project a millimetre or two beyond the side of the phone. That is
// expected and harmless - it prints fine and does not affect the optics.
module clip_part(band_h = clip_band_h) {
    y0  = min(0, cam_y - (spigot_r + fit_joint + wall) - wall);
    len = band_h - y0;
    difference() {
        union() {
            // outer sleeve
            translate([-(phone_width / 2 + wall + fit),
                       y0,
                       -(phone_thickness + wall + fit)])
                cube([phone_width + 2 * (wall + fit),
                      len,
                      phone_thickness + 2 * (wall + fit)]);
            // socket ring that the cone spigot pushes into
            translate([cam_x, cam_y, 0])
                cylinder(h = clip_plate_t + socket_h,
                         r = spigot_r + fit_joint + wall);
        }
        // phone body pocket, open at both ends so the phone slides in
        translate([-(phone_width / 2 + fit), y0 - 1, -(phone_thickness + fit)])
            cube([phone_width + 2 * fit, len + 2, phone_thickness + fit]);
        // camera island pocket, so the plate still sits flat on the back
        if (island_pocket)
            translate([isl_x - island_w / 2 - fit,
                       isl_y - island_l / 2 - fit,
                       -0.01])
                cube([island_w + 2 * fit, island_l + 2 * fit,
                      island_depth + fit]);
        // clear optical hole through the plate
        translate([cam_x, cam_y, -(phone_thickness + 10)])
            cylinder(h = phone_thickness + clip_plate_t + socket_h + 20,
                     d = aperture_diameter);
        // socket bore
        translate([cam_x, cam_y, clip_plate_t])
            cylinder(h = socket_h + 1, r = spigot_r + fit_joint);
        // screen-side window: keeps the viewfinder visible. A 6 mm border
        // is left all round so the band stays stiff.
        translate([-(phone_width / 2 - 6), y0 + 6,
                   -(phone_thickness + wall + fit + 1)])
            cube([phone_width - 12, max(len - 12, 4), wall + 1.02]);
    }
}

// --- 10 mm test ring: checks the phone fit before the full print ------
module test_ring() {
    difference() {
        translate([-(phone_width / 2 + wall + fit), 0,
                   -(phone_thickness + wall + fit)])
            cube([phone_width + 2 * (wall + fit), 10,
                  phone_thickness + 2 * (wall + fit)]);
        translate([-(phone_width / 2 + fit), -1, -(phone_thickness + fit)])
            cube([phone_width + 2 * fit, 12, phone_thickness + fit]);
    }
}


// =====================================================================
//  8. OUTPUT
// =====================================================================

if (part == "cone")
    // Skin end is already at z = 0: slice as exported, large end on the bed.
    cone_part();

else if (part == "clip")
    // Screen-side face on the bed, socket ring pointing UP.
    // The back plate bridges the phone pocket - a short, fully supported
    // bridge between the two side walls. No supports needed.
    translate([0, 0, phone_thickness + wall + fit]) clip_part();

else if (part == "testring")
    translate([0, 0, phone_thickness + wall + fit]) test_ring();

else if (part == "assembly") {
    // Preview only. The clip is moved so its socket sits on the cone axis.
    color("LightGrey") cone_part();
    color("SteelBlue")
        translate([-cam_x, -cam_y, standoff_h - clip_plate_t]) clip_part();
}
