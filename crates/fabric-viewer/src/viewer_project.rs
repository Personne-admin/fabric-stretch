//! Portable viewer-workspace project format.

use crate::press_tab::PersistedPressView;
use crate::selection::FrameSelection;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const FORMAT_VERSION: u32 = 4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct DatasetState {
    pub source_path: PathBuf,
    pub display_name: String,
    pub enabled: bool,
    pub hidden_runs: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub run_visible: Vec<bool>,
}

impl Default for DatasetState {
    fn default() -> Self {
        Self {
            source_path: PathBuf::new(),
            display_name: String::new(),
            enabled: true,
            hidden_runs: Vec::new(),
            run_visible: Vec::new(),
        }
    }
}

impl DatasetState {
    pub fn is_run_visible(&self, run_name: &str, index: usize) -> bool {
        if !self.hidden_runs.is_empty() || self.run_visible.is_empty() {
            let leaf = run_name.rsplit('/').next().unwrap_or(run_name);
            !self
                .hidden_runs
                .iter()
                .any(|hidden| hidden == run_name || (hidden == leaf && !hidden.contains('/')))
        } else {
            self.run_visible.get(index).copied().unwrap_or(true)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ViewerProjectState {
    pub format_version: u32,
    pub datasets: Vec<DatasetState>,
    pub press_view: PersistedPressView,
    pub press_views: Vec<PersistedPressView>,
    pub selection: FrameSelection,
}

impl Default for ViewerProjectState {
    fn default() -> Self {
        Self {
            format_version: FORMAT_VERSION,
            datasets: Vec::new(),
            press_view: PersistedPressView::default(),
            press_views: Vec::new(),
            selection: FrameSelection::default(),
        }
    }
}

impl ViewerProjectState {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let mut state: Self = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if state.format_version > FORMAT_VERSION {
            return Err(format!(
                "viewer project format {} is newer than supported {}",
                state.format_version, FORMAT_VERSION
            ));
        }

        if state.press_views.is_empty() && state.format_version <= 2 {
            state.press_views = vec![state.press_view.clone()];
        }
        state.format_version = FORMAT_VERSION;
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults_missing_fields_and_round_trips() {
        let s = ViewerProjectState::from_json(
            r#"{"format_version":1,"datasets":[{"source_path":"/a","display_name":"A"}]}"#,
        )
        .unwrap();
        assert!(s.datasets[0].enabled);
        assert_eq!(s.press_view, Default::default());
        let again = ViewerProjectState::from_json(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(again.datasets.len(), 1);
    }
    #[test]
    fn rejects_future_format() {
        assert!(ViewerProjectState::from_json(r#"{"format_version":99}"#).is_err());
    }

    #[test]
    fn the_removed_3d_view_state_in_old_files_is_ignored() {
        let state = ViewerProjectState::from_json(
            r#"{"format_version":3,"datasets":[{"source_path":"/a","display_name":"A"}],
                "view_3d":{"state":{"camera":{"azimuth_deg":123.0},"cylinder":{"radius":2.0}}},
                "press_view":{"scene":{"surface":"Cylinder"}}}"#,
        )
        .unwrap();
        assert_eq!(state.datasets.len(), 1);
        assert_eq!(state.format_version, FORMAT_VERSION);
        let text = serde_json::to_string(&state).unwrap();
        assert!(!text.contains("view_3d"), "3D state is no longer written");
    }

    #[test]
    fn keeps_multiple_datasets_and_subset_visibility() {
        let mut state = ViewerProjectState {
            datasets: vec![
                DatasetState {
                    source_path: "/a".into(),
                    display_name: "A".into(),
                    ..DatasetState::default()
                },
                DatasetState {
                    source_path: "/b".into(),
                    display_name: "B".into(),
                    enabled: false,
                    hidden_runs: vec!["warp_loading".into(), "warp_release".into()],
                    ..DatasetState::default()
                },
                DatasetState {
                    source_path: "/c".into(),
                    display_name: "C".into(),
                    hidden_runs: vec!["weft_release".into()],
                    ..DatasetState::default()
                },
            ],
            ..ViewerProjectState::default()
        };
        state.press_view.beta = Some(0.3);
        state.press_view.light.elevation_deg = 20.0;
        let restored =
            ViewerProjectState::from_json(&serde_json::to_string(&state).unwrap()).unwrap();
        assert_eq!(restored.datasets.len(), 3);
        assert!(
            restored.datasets[0].enabled
                && !restored.datasets[1].enabled
                && restored.datasets[2].enabled
        );
        assert_eq!(restored.datasets[2].hidden_runs, vec!["weft_release"]);
        assert!(restored.datasets[2].is_run_visible("weft_loading", 0));
        assert!(!restored.datasets[2].is_run_visible("weft_release", 1));
        assert_eq!(restored.press_view.beta, Some(0.3));
        assert_eq!(restored.press_view.light.elevation_deg, 20.0);
    }

    #[test]
    fn visibility_is_keyed_by_run_name_not_position() {
        let state = DatasetState {
            hidden_runs: vec!["weft_release".into()],
            ..DatasetState::default()
        };
        assert!(state.is_run_visible("warp_loading", 0));
        assert!(state.is_run_visible("weft_loading", 1));
        assert!(!state.is_run_visible("weft_release", 0));
        assert!(!state.is_run_visible("weft_release", 7));
    }

    #[test]
    fn nested_runs_sharing_a_directory_name_are_told_apart() {
        let state = DatasetState {
            hidden_runs: vec!["b/warp_loading".into()],
            ..DatasetState::default()
        };
        assert!(state.is_run_visible("a/warp_loading", 0));
        assert!(!state.is_run_visible("b/warp_loading", 1));

        let legacy = DatasetState {
            hidden_runs: vec!["warp_loading".into()],
            ..DatasetState::default()
        };
        assert!(!legacy.is_run_visible("a/warp_loading", 0));
        assert!(!legacy.is_run_visible("warp_loading", 0));
    }

    #[test]
    fn format_three_positional_visibility_still_loads() {
        let state = ViewerProjectState::from_json(
            r#"{"format_version":3,"datasets":[{"source_path":"/a","display_name":"A",
                "run_visible":[true,false,true]}]}"#,
        )
        .unwrap();
        let d = &state.datasets[0];
        assert!(d.is_run_visible("anything", 0));
        assert!(!d.is_run_visible("anything", 1));
        assert!(d.is_run_visible("anything", 2));
        assert!(
            d.is_run_visible("anything", 9),
            "unknown positions are visible"
        );
    }

    #[test]
    fn format_two_single_press_view_is_restored_as_one_view() {
        let state = ViewerProjectState::from_json(
            r#"{"format_version":2,"press_view":{"force_n":6.5,"orientation":"weft","branch":"release"}}"#,
        )
        .unwrap();
        assert_eq!(state.press_views.len(), 1);
        assert_eq!(state.press_views[0].force_n, Some(6.5));
        assert_eq!(state.press_views[0].orientation, "weft");
        assert!(
            state.press_views[0].tiling,
            "older files predate the tiling switch: default on"
        );
    }

    #[test]
    fn several_press_views_round_trip_through_the_project_file() {
        let left = PersistedPressView {
            project: Some("/data/fabric_002".into()),
            force_n: Some(5.0),
            ..PersistedPressView::default()
        };
        let right = PersistedPressView {
            project: Some("/data/mm003".into()),
            orientation: "weft".into(),
            tiling: false,
            beta: Some(0.3),
            ..PersistedPressView::default()
        };
        let state = ViewerProjectState {
            press_views: vec![left.clone(), right.clone()],
            ..ViewerProjectState::default()
        };
        let text = serde_json::to_string_pretty(&state).unwrap();
        let restored = ViewerProjectState::from_json(&text).unwrap();
        assert_eq!(restored.press_views, vec![left, right]);
        assert_eq!(restored.format_version, FORMAT_VERSION);
    }
}
