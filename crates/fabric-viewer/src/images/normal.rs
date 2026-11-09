//! Normal-map math in the camera frame: x right, y up, z towards the camera.
//! Encoded RGB is `n * 0.5 + 0.5`; G > 128 tilts towards image top.

/// Decode `c / 255 * 2 - 1` and normalize. Preserve 1-LSB tilts; only near-zero
/// magnitudes, including the `(128, 128, 128)` sentinel, fall back to +Z.
pub fn decode_normal(r: u8, g: u8, b: u8) -> [f32; 3] {
    let v = [
        r as f32 / 255.0 * 2.0 - 1.0,
        g as f32 / 255.0 * 2.0 - 1.0,
        b as f32 / 255.0 * 2.0 - 1.0,
    ];
    normalize3(v)
}

/// Transparent texels decode to +Z; otherwise use [`decode_normal`].
/// Channels are raw normal data, not sRGB.
#[cfg(test)]
pub fn decode_normal_pixel(px: &[u8; 4]) -> [f32; 3] {
    if px[3] == 0 {
        [0.0, 0.0, 1.0]
    } else {
        decode_normal(px[0], px[1], px[2])
    }
}

/// Treat near-zero magnitude as missing data. The neutral sentinel has magnitude
/// ≈0.0068; valid encoded normals are near 1.
const NEAR_ZERO_LEN: f32 = 0.01;

/// Normalize a vector; zero/near-zero input falls back to +Z so callers can
/// never hand a zero-length direction to lighting code.
pub fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > NEAR_ZERO_LEN {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Unit light direction. Azimuth is counter-clockwise from image right;
/// elevation is above the image plane (90° towards the camera).
pub fn light_direction(azimuth_deg: f32, elevation_deg: f32) -> [f32; 3] {
    let az = azimuth_deg.to_radians();
    let el = elevation_deg.to_radians();
    normalize3([el.cos() * az.cos(), el.cos() * az.sin(), el.sin()])
}

/// Lambertian diffuse term for unit normal and light vectors.
pub fn lambert(normal: [f32; 3], light_dir: [f32; 3]) -> f32 {
    (normal[0] * light_dir[0] + normal[1] * light_dir[1] + normal[2] * light_dir[2]).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol && (a[2] - b[2]).abs() < tol
    }

    #[test]
    fn normal_decode_is_exact_no_dead_zone() {
        // (0,0,0) decodes exactly to (-1,-1,-1)/sqrt(3), not to a cardinal axis.
        let black = decode_normal(0, 0, 0);
        let expected = [-1.0_f32 / 3.0_f32.sqrt(); 3];
        assert!(approx(black, expected, 1e-5), "got {black:?}");

        // (255,255,255) decodes exactly to (+1,+1,+1)/sqrt(3).
        let white = decode_normal(255, 255, 255);
        let expected = [1.0_f32 / 3.0_f32.sqrt(); 3];
        assert!(approx(white, expected, 1e-5), "got {white:?}");

        // +X-dominant tilt.
        let x = decode_normal(255, 127, 127);
        assert!(
            x[0] > 0.99 && x[1].abs() < 0.01 && x[2].abs() < 0.01,
            "got {x:?}"
        );

        // -Y-dominant tilt.
        let y = decode_normal(127, 0, 127);
        assert!(
            y[1] < -0.99 && y[0].abs() < 0.01 && y[2].abs() < 0.01,
            "got {y:?}"
        );
    }

    #[test]
    fn neutral_surrounding_keeps_small_tilt() {
        // Preserve 1-LSB tilts near neutral rather than snapping to an axis.
        for c in 126..=129u8 {
            let n = decode_normal(c, c, 255);
            assert!(
                n[0].abs() > 1e-6,
                "{c}: per-component tilt must be preserved, got {n:?}"
            );
            assert!(
                approx(n, [0.0, 0.0, 1.0], 0.02),
                "{c}: must stay near +Z, got {n:?}"
            );
        }
    }

    #[test]
    fn near_zero_magnitude_falls_back_to_plus_z() {
        assert_eq!(normalize3([0.0, 0.0, 0.0]), [0.0, 0.0, 1.0]);
        assert_eq!(normalize3([1e-12, 0.0, 0.0]), [0.0, 0.0, 1.0]);
        // The neutral-gray "no data" sentinel (128,128,128) has magnitude
        // ~0.0068 < NEAR_ZERO_LEN -> +Z.
        let decoded = decode_normal(128, 128, 128);
        assert_eq!(decoded, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn light_direction_never_zero() {
        for (az, el) in [(0.0, 90.0), (45.0, 45.0), (180.0, 0.0), (90.0, -30.0)] {
            let d = light_direction(az, el);
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "{az},{el} -> {d:?}");
        }
        assert!(approx(light_direction(0.0, 90.0), [0.0, 0.0, 1.0], 1e-5));
    }

    #[test]
    fn azimuth_90_lights_a_normal_pointing_image_up() {
        // G > 128 means the normal tilts toward the top of the image (+y).
        let up = decode_normal(128, 230, 200);
        let down = decode_normal(128, 26, 200);
        assert!(up[1] > 0.0 && down[1] < 0.0);
        let l = light_direction(90.0, 30.0);
        assert!(lambert(up, l) > lambert([0.0, 0.0, 1.0], l));
        assert!(lambert(up, l) > lambert(down, l));
        assert_eq!(lambert(down, light_direction(90.0, 5.0)), 0.0);
        // Azimuth 0 lights a normal tilted to image-right (R > 128).
        let right = decode_normal(230, 128, 200);
        assert!(
            lambert(right, light_direction(0.0, 30.0)) > lambert(up, light_direction(0.0, 30.0))
        );
    }

    #[test]
    fn lambert_known_cases() {
        assert!((lambert([0.0, 0.0, 1.0], light_direction(0.0, 90.0)) - 1.0).abs() < 1e-5);
        let l = light_direction(0.0, 45.0);
        assert!((lambert([0.0, 0.0, 1.0], l) - 2.0_f32.sqrt() / 2.0).abs() < 1e-5);
        assert!((lambert([0.0, 0.0, 1.0], light_direction(0.0, 0.0))).abs() < 1e-6);
        assert_eq!(lambert([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]), 0.0);
    }

    #[test]
    fn normals_pointing_away_are_clamped() {
        // A surface tilted 90 deg away from the light gets zero, not negative.
        let n = [1.0, 0.0, 0.0];
        let l = light_direction(0.0, 90.0); // straight down
        assert_eq!(lambert(n, l), 0.0);
    }

    #[test]
    fn alpha_aware_decode_transparent_is_plus_z() {
        // Fully transparent texels are "no data" -> +Z, regardless of RGB.
        assert_eq!(decode_normal_pixel(&[255, 255, 255, 0]), [0.0, 0.0, 1.0]);
        assert_eq!(decode_normal_pixel(&[0, 0, 0, 0]), [0.0, 0.0, 1.0]);
        // The neutral-gray no-data sentinel also falls back to +Z.
        assert_eq!(decode_normal_pixel(&[128, 128, 128, 255]), [0.0, 0.0, 1.0]);
        let opaque = decode_normal_pixel(&[255, 127, 127, 255]);
        assert!(opaque[0] > 0.99, "got {opaque:?}");
    }
}
