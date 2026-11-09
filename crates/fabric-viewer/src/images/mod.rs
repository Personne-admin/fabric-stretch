//! Image cache, role lookup and relighting helpers.

pub mod cache;
pub mod normal;
pub mod relight;
pub mod roles;

pub use cache::{ImageCache, ImageStatus};
pub use roles::{frame_roles, PsLayers};

/// Preferred role, then diffuse → rgb → albedo → first sorted role.
/// Returns `(role, filename)` or `None` for a frame without images.
pub fn resolve_role<'a, I>(roles: I, preferred: Option<&str>) -> Option<(&'a str, &'a str)>
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let roles: Vec<(&'a str, &'a str)> = roles.into_iter().collect();
    if let Some(preferred) = preferred {
        if let Some(hit) = roles.iter().find(|(role, _)| *role == preferred) {
            return Some(*hit);
        }
    }
    for fallback in ["diffuse", "rgb", "albedo"] {
        if let Some(hit) = roles.iter().find(|(role, _)| *role == fallback) {
            return Some(*hit);
        }
    }
    roles.into_iter().min_by(|a, b| a.0.cmp(b.0))
}

/// Find the role that carries a normal map ("normal", "normal_map", "nrm",
/// or any role containing "normal"), if the frame has one.
pub fn find_normal_role<'a, I>(roles: I) -> Option<&'a str>
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    roles.into_iter().map(|(role, _)| role).find(|role| {
        let lower = role.to_ascii_lowercase();
        lower == "normal" || lower == "nrm" || lower == "normal_map" || lower.contains("normal")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roles() -> Vec<(&'static str, &'static str)> {
        vec![
            ("albedo", "alb.png"),
            ("diffuse", "dif.png"),
            ("ps_1", "ps1.png"),
            ("rgb", "rgb.png"),
        ]
    }

    #[test]
    fn role_fallback_prefers_requested_role() {
        let r = resolve_role(roles().iter().copied(), Some("ps_1")).unwrap();
        assert_eq!(r, ("ps_1", "ps1.png"));
    }

    #[test]
    fn role_fallback_falls_through_when_preferred_missing() {
        let r = resolve_role(roles().iter().copied(), Some("missing")).unwrap();
        assert_eq!(r, ("diffuse", "dif.png"));
    }

    #[test]
    fn role_fallback_defaults_to_diffuse_then_rgb_then_albedo() {
        let no_diffuse = roles().into_iter().filter(|(r, _)| *r != "diffuse");
        let r = resolve_role(no_diffuse, None).unwrap();
        assert_eq!(r, ("rgb", "rgb.png"));

        let only_albedo = [("albedo", "alb.png")];
        let r = resolve_role(only_albedo.iter().copied(), None).unwrap();
        assert_eq!(r, ("albedo", "alb.png"));
    }

    #[test]
    fn role_fallback_sorted_first() {
        let r = resolve_role(vec![("zzz", "z.png"), ("aaa", "a.png")], None).unwrap();
        assert_eq!(r, ("aaa", "a.png"));
    }

    #[test]
    fn role_fallback_empty_roles_is_none() {
        assert_eq!(resolve_role(std::iter::empty::<(&str, &str)>(), None), None);
    }

    #[test]
    fn normal_role_detection() {
        let roles = roles();
        assert_eq!(
            find_normal_role(roles.iter().copied()),
            None,
            "no normal role present"
        );
        let with_normal = [("normal", "n.png"), ("diffuse", "d.png")];
        assert_eq!(
            find_normal_role(with_normal.iter().copied()),
            Some("normal")
        );
        let with_nrm = [("nrm", "n.png")];
        assert_eq!(find_normal_role(with_nrm.iter().copied()), Some("nrm"));
    }
}
