use std::time::{Duration, Instant};

use glam::{Affine3A, Quat, Vec3, Vec3A, vec3a};
use ovr_overlay::{
    chaperone_setup::ChaperoneSetupManager,
    compositor::CompositorManager,
    sys::{EChaperoneConfigFile, ETrackingUniverseOrigin, HmdMatrix34_t},
};
use wgui::log::LogErr;

use crate::{
    backend::{playspace_common, task::PlayspaceTask},
    state::{AppState, PlayspaceState, load_playspace_state, save_playspace_state},
    windowing::manager::OverlayWindowManager,
};

use super::{helpers::Affine3AConvert, overlay::OpenVrOverlayData};

struct MoverData<T> {
    pose: Affine3A,
    hand: usize,
    hand_pose: T,
}

// How often to check whether SteamVR or its driver replaced the zero pose while nothing is being moved
const OFFSET_CHECK_INTERVAL: Duration = Duration::from_millis(500);

pub(super) struct PlayspaceMover {
    universe: ETrackingUniverseOrigin,
    drag: Option<MoverData<Vec3A>>,
    rotate: Option<MoverData<Quat>>,
    playspace_state: PlayspaceState,
    // The zero pose SteamVR set before WayVR moved the space, and the one WayVR committed on top of it.
    // Inside-out drivers rewrite the zero pose, for example when tracking relocalizes; the offset between
    // these two is then applied again to the new pose.
    external_pose: Option<Affine3A>,
    committed_pose: Option<Affine3A>,
    committed_bounds: Vec<[f32; 3]>,
    next_offset_check: Instant,
}

impl PlayspaceMover {
    pub fn new() -> Self {
        Self {
            universe: ETrackingUniverseOrigin::TrackingUniverseRawAndUncalibrated,
            drag: None,
            rotate: None,
            playspace_state: load_playspace_state().unwrap_or_default(),
            external_pose: None,
            committed_pose: None,
            committed_bounds: Vec::new(),
            next_offset_check: Instant::now(),
        }
    }

    pub fn handle_task(
        &mut self,
        app: &mut AppState,
        chaperone_mgr: &mut ChaperoneSetupManager,
        overlays: &mut OverlayWindowManager<OpenVrOverlayData>,
        task: PlayspaceTask,
    ) {
        match task {
            PlayspaceTask::FixFloor => {
                self.fix_floor(chaperone_mgr, app, overlays);
            }
            PlayspaceTask::Reset => {
                self.reset_offset(chaperone_mgr, app, overlays);
            }
            PlayspaceTask::Recenter => {
                self.recenter(chaperone_mgr, app, overlays);
            }
            PlayspaceTask::SaveCenter => {
                self.save_center(chaperone_mgr);
            }
            PlayspaceTask::ResetCenter => {
                self.playspace_state.openvr_space_center = Affine3A::IDENTITY;
            }
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn update(
        &mut self,
        chaperone_mgr: &mut ChaperoneSetupManager,
        overlays: &mut OverlayWindowManager<OpenVrOverlayData>,
        app: &mut AppState,
    ) {
        let universe = self.universe.clone();

        if self.committed_pose.is_some()
            && self.drag.is_none()
            && self.rotate.is_none()
            && Instant::now() >= self.next_offset_check
        {
            self.next_offset_check = Instant::now() + OFFSET_CHECK_INTERVAL;
            self.current_pose(chaperone_mgr);
        }

        if let Some(data) = self.rotate.as_mut() {
            let pointer = &app.input_state.pointers[data.hand];
            if !pointer.now.space_rotate {
                self.rotate = None;
                log::info!("End space rotate");
                return;
            }

            let new_hand =
                Quat::from_affine3(&(data.pose * app.input_state.pointers[data.hand].raw_pose));

            let dq = new_hand * data.hand_pose.conjugate();
            let rel_y = f32::atan2(
                2.0 * dq.y.mul_add(dq.w, dq.x * dq.z),
                2.0f32.mul_add(dq.w.mul_add(dq.w, dq.x * dq.x), -1.0),
            );

            let mut space_transform = Affine3A::from_rotation_y(rel_y);
            let offset = (space_transform.transform_vector3a(app.input_state.hmd.translation)
                - app.input_state.hmd.translation)
                * -1.0;
            space_transform.translation = offset;

            let before_pose = data.pose;
            data.pose *= space_transform;
            if !app.session.config.space_drag_affects_world {
                playspace_common::shift_world(overlays, &mut app.anchor, &before_pose, &data.pose);
            }
            data.hand_pose = new_hand;

            if self.universe == ETrackingUniverseOrigin::TrackingUniverseStanding {
                apply_chaperone_transform(space_transform.inverse(), chaperone_mgr);
            }
            set_working_copy(&universe, chaperone_mgr, &data.pose);
            chaperone_mgr.commit_working_copy(EChaperoneConfigFile::EChaperoneConfigFile_Live);
            let after = data.pose;
            self.note_commit(before_pose, after, chaperone_mgr);
        } else {
            for (i, pointer) in app.input_state.pointers.iter().enumerate() {
                if pointer.now.space_rotate {
                    let Some(mat) = self.current_pose(chaperone_mgr) else {
                        log::warn!("Can't space rotate - failed to get zero pose");
                        return;
                    };
                    let hand_pose = Quat::from_affine3(&(mat * pointer.raw_pose));
                    self.rotate = Some(MoverData {
                        pose: mat,
                        hand: i,
                        hand_pose,
                    });
                    self.drag = None;
                    log::info!("Start space rotate");
                    return;
                }
            }
        }

        if let Some(data) = self.drag.as_mut() {
            let pointer = &app.input_state.pointers[data.hand];
            if !pointer.now.space_drag {
                self.drag = None;
                log::info!("End space drag");
                return;
            }

            let new_hand = data
                .pose
                .transform_point3a(app.input_state.pointers[data.hand].raw_pose.translation);

            let relative_pos = if app.session.config.space_drag_unlocked {
                new_hand - data.hand_pose
            } else {
                vec3a(0., new_hand.y - data.hand_pose.y, 0.)
            } * app.session.config.space_drag_multiplier;

            if relative_pos.length_squared() > 1000.0 {
                log::warn!("Space drag too fast, ignoring");
                return;
            }

            let overlay_offset = data.pose.inverse().transform_vector3a(relative_pos) * -1.0;
            let before_pose = data.pose;
            data.pose.translation += relative_pos;
            if !app.session.config.space_drag_affects_world {
                playspace_common::shift_world(overlays, &mut app.anchor, &before_pose, &data.pose);
            }
            data.hand_pose = new_hand;

            if self.universe == ETrackingUniverseOrigin::TrackingUniverseStanding {
                apply_chaperone_offset(overlay_offset, chaperone_mgr);
            }
            set_working_copy(&universe, chaperone_mgr, &data.pose);
            chaperone_mgr.commit_working_copy(EChaperoneConfigFile::EChaperoneConfigFile_Live);
            let after = data.pose;
            self.note_commit(before_pose, after, chaperone_mgr);
        } else {
            for (i, pointer) in app.input_state.pointers.iter().enumerate() {
                if pointer.now.space_drag {
                    let Some(mat) = self.current_pose(chaperone_mgr) else {
                        log::warn!("Can't space drag - failed to get zero pose");
                        return;
                    };
                    let hand_pos = mat.transform_point3a(pointer.raw_pose.translation);
                    self.drag = Some(MoverData {
                        pose: mat,
                        hand: i,
                        hand_pose: hand_pos,
                    });
                    self.rotate = None;
                    log::info!("Start space drag");
                    return;
                }
            }
        }

        for pointer in &app.input_state.pointers {
            if pointer.now.space_reset && !pointer.before.space_reset {
                self.reset_offset(chaperone_mgr, app, overlays);
                log::info!("Space reset");
                return;
            }
        }
    }

    pub fn reset_offset(
        &mut self,
        chaperone_mgr: &mut ChaperoneSetupManager,
        app: &mut AppState,
        overlays: &mut OverlayWindowManager<OpenVrOverlayData>,
    ) {
        let Some(before) = get_working_copy(&self.universe, chaperone_mgr) else {
            log::warn!("Can't reset offset - failed to get zero pose");
            return;
        };

        let mut xform = self.playspace_state.openvr_space_center;
        if self.universe == ETrackingUniverseOrigin::TrackingUniverseSeated {
            xform.translation.y -= 1.7;
        }

        set_working_copy(&self.universe, chaperone_mgr, &xform);
        chaperone_mgr.commit_working_copy(EChaperoneConfigFile::EChaperoneConfigFile_Live);
        self.forget_offset();

        if !app.session.config.space_drag_affects_world {
            playspace_common::shift_world(overlays, &mut app.anchor, &before, &xform);
        }

        if self.drag.is_some() {
            log::info!("Space drag interrupted by manual reset");
            self.drag = None;
        }
        if self.rotate.is_some() {
            log::info!("Space rotate interrupted by manual reset");
            self.rotate = None;
        }
    }

    pub fn fix_floor(
        &mut self,
        chaperone_mgr: &mut ChaperoneSetupManager,
        app: &mut AppState,
        overlays: &mut OverlayWindowManager<OpenVrOverlayData>,
    ) {
        let input = &app.input_state;
        let anchor = &mut app.anchor;
        let y1 = input.pointers[0].pose.translation.y;
        let y2 = input.pointers[1].pose.translation.y;
        let Some(mut mat) = self.current_pose(chaperone_mgr) else {
            log::warn!("Can't fix floor - failed to get zero pose");
            return;
        };
        let offset = y1.min(y2) - 0.03;
        let before = mat;
        mat.translation.y += offset;
        self.playspace_state.openvr_space_center.translation.y = mat.translation.y;

        set_working_copy(&self.universe, chaperone_mgr, &mat);
        chaperone_mgr.commit_working_copy(EChaperoneConfigFile::EChaperoneConfigFile_Live);
        self.note_commit(before, mat, chaperone_mgr);

        if !app.session.config.space_drag_affects_world {
            playspace_common::shift_world(overlays, anchor, &before, &mat);
        }

        if self.drag.is_some() {
            log::info!("Space drag interrupted by fix floor");
            self.drag = None;
        }
        if self.rotate.is_some() {
            log::info!("Space rotate interrupted by fix floor");
            self.rotate = None;
        }
    }

    pub fn recenter(
        &mut self,
        chaperone_mgr: &mut ChaperoneSetupManager,
        app: &mut AppState,
        overlays: &mut OverlayWindowManager<OpenVrOverlayData>,
    ) {
        if self.drag.is_some() {
            log::info!("Space drag interrupted by recenter");
            self.drag = None;
        }
        if self.rotate.is_some() {
            log::info!("Space rotate interrupted by recenter");
            self.rotate = None;
        }

        let input = &app.input_state;
        let anchor = &mut app.anchor;

        let Some(mat) = self.current_pose(chaperone_mgr) else {
            log::warn!("Can't recenter - failed to get zero pose");
            return;
        };

        let before = mat;

        let cur_rot: Quat = Quat::from_affine3(&mat);
        let cur_pos: Vec3 = Vec3::from(mat.translation);

        let mut stage_offset = Affine3A::from_rotation_translation(cur_rot, cur_pos);

        let horiz_hmd_pos = Vec3::new(input.hmd.translation.x, 0.0, input.hmd.translation.z);

        let fwd = input.hmd.transform_vector3a(Vec3A::NEG_Z);
        let horiz_len_sq = fwd.x.mul_add(fwd.x, fwd.z * fwd.z);

        let hmd_yaw = if horiz_len_sq > f32::EPSILON {
            let yaw = (-fwd.x).atan2(-fwd.z);
            Quat::from_rotation_y(yaw)
        } else {
            Quat::IDENTITY
        };

        let recenter_offset = Affine3A::from_rotation_translation(hmd_yaw, horiz_hmd_pos);

        stage_offset *= recenter_offset;

        let (_, new_rot, new_pos) = stage_offset.to_scale_rotation_translation();

        let new_mat = if horiz_len_sq > f32::EPSILON {
            Affine3A::from_rotation_translation(new_rot, new_pos)
        } else {
            let mut m = mat;
            m.translation = new_pos.into();
            m
        };

        set_working_copy(&self.universe, chaperone_mgr, &new_mat);
        chaperone_mgr.commit_working_copy(EChaperoneConfigFile::EChaperoneConfigFile_Live);
        self.note_commit(before, new_mat, chaperone_mgr);

        if !app.session.config.space_drag_affects_world {
            playspace_common::shift_world(overlays, anchor, &before, &new_mat);
        }
    }

    pub fn playspace_changed(
        &mut self,
        compositor_mgr: &mut CompositorManager,
        chaperone_mgr: &mut ChaperoneSetupManager,
    ) {
        let new_universe = compositor_mgr.get_tracking_space();
        if new_universe != self.universe {
            log::info!(
                "Playspace changed: {} -> {}",
                universe_str(&self.universe),
                universe_str(&new_universe)
            );
            self.universe = new_universe;
            self.forget_offset();
        } else if self.committed_pose.is_some() && self.drag.is_none() && self.rotate.is_none() {
            // A drag in progress re-applies the offset when it starts again
            self.current_pose(chaperone_mgr);
        }
    }

    /// The current zero pose. If SteamVR or its driver replaced the pose WayVR committed, WayVR's offset is
    /// applied again on top of the new one first.
    fn current_pose(&mut self, chaperone_mgr: &mut ChaperoneSetupManager) -> Option<Affine3A> {
        let live = get_working_copy(&self.universe, chaperone_mgr)?;
        let (Some(external), Some(committed)) = (self.external_pose, self.committed_pose) else {
            return Some(live);
        };
        if poses_match(&live, &committed) {
            return Some(live);
        }

        let offset = external.inverse() * committed;
        let restored = live * offset;
        // Bounds that still match WayVR's were left alone and are already moved
        if self.universe == ETrackingUniverseOrigin::TrackingUniverseStanding
            && !bounds_match(&bounds_points(chaperone_mgr), &self.committed_bounds)
        {
            apply_chaperone_transform(offset.inverse(), chaperone_mgr);
        }
        set_working_copy(&self.universe, chaperone_mgr, &restored);
        chaperone_mgr.commit_working_copy(EChaperoneConfigFile::EChaperoneConfigFile_Live);
        log::info!("Zero pose was replaced externally, applied the space offset again");

        self.external_pose = None;
        self.note_commit(live, restored, chaperone_mgr);
        Some(restored)
    }

    fn note_commit(
        &mut self,
        before: Affine3A,
        after: Affine3A,
        chaperone_mgr: &mut ChaperoneSetupManager,
    ) {
        self.external_pose.get_or_insert(before);
        self.committed_pose = Some(after);
        if self.universe == ETrackingUniverseOrigin::TrackingUniverseStanding {
            self.committed_bounds = bounds_points(chaperone_mgr);
        }
    }

    fn forget_offset(&mut self) {
        self.external_pose = None;
        self.committed_pose = None;
        self.committed_bounds.clear();
    }

    pub fn get_universe(&self) -> ETrackingUniverseOrigin {
        self.universe.clone()
    }

    pub fn save_center(&mut self, chaperone_mgr: &mut ChaperoneSetupManager) {
        if self.drag.is_some() {
            log::info!("Space drag interrupted by save center");
            self.drag = None;
        }
        if self.rotate.is_some() {
            log::info!("Space rotate interrupted by save center");
            self.rotate = None;
        }

        let Some(mat) = get_working_copy(&self.universe, chaperone_mgr) else {
            log::warn!("Can't save center - failed to get zero pose");
            return;
        };

        self.playspace_state.openvr_space_center = mat;
        let _ =
            save_playspace_state(&self.playspace_state).log_err("Could not save playspace state");
    }
}

const fn universe_str(universe: &ETrackingUniverseOrigin) -> &'static str {
    match universe {
        ETrackingUniverseOrigin::TrackingUniverseSeated => "Seated",
        ETrackingUniverseOrigin::TrackingUniverseStanding => "Standing",
        ETrackingUniverseOrigin::TrackingUniverseRawAndUncalibrated => "Raw",
    }
}

fn get_working_copy(
    universe: &ETrackingUniverseOrigin,
    chaperone_mgr: &mut ChaperoneSetupManager,
) -> Option<Affine3A> {
    chaperone_mgr.revert_working_copy();
    let mat = match universe {
        ETrackingUniverseOrigin::TrackingUniverseStanding => {
            chaperone_mgr.get_working_standing_zero_pose_to_raw_tracking_pose()
        }
        _ => chaperone_mgr.get_working_seated_zero_pose_to_raw_tracking_pose(),
    };
    mat.map(|m| m.to_affine())
}

fn set_working_copy(
    universe: &ETrackingUniverseOrigin,
    chaperone_mgr: &mut ChaperoneSetupManager,
    mat: &Affine3A,
) {
    let mat = HmdMatrix34_t::from_affine(mat);
    match universe {
        ETrackingUniverseOrigin::TrackingUniverseStanding => {
            chaperone_mgr.set_working_standing_zero_pose_to_raw_tracking_pose(&mat);
        }
        _ => chaperone_mgr.set_working_seated_zero_pose_to_raw_tracking_pose(&mat),
    }
}

fn apply_chaperone_offset(offset: Vec3A, chaperone_mgr: &mut ChaperoneSetupManager) {
    let mut quads = chaperone_mgr.get_live_collision_bounds_info();
    for quad in &mut quads {
        quad.vCorners.iter_mut().for_each(|corner| {
            corner.v[0] += offset.x;
            corner.v[2] += offset.z;
        });
    }
    chaperone_mgr.set_working_collision_bounds_info(quads.as_mut_slice());
}

fn apply_chaperone_transform(transform: Affine3A, chaperone_mgr: &mut ChaperoneSetupManager) {
    let mut quads = chaperone_mgr.get_live_collision_bounds_info();
    for quad in &mut quads {
        quad.vCorners.iter_mut().for_each(|corner| {
            let coord = transform.transform_point3a(Vec3A::from_slice(&corner.v));
            corner.v[0] = coord.x;
            corner.v[2] = coord.z;
        });
    }
    chaperone_mgr.set_working_collision_bounds_info(quads.as_mut_slice());
}

fn poses_match(a: &Affine3A, b: &Affine3A) -> bool {
    a.to_cols_array()
        .iter()
        .zip(b.to_cols_array())
        .all(|(x, y)| (x - y).abs() < 1e-4)
}

fn bounds_match(a: &[[f32; 3]], b: &[[f32; 3]]) -> bool {
    a.len() == b.len()
        && a.iter()
            .flatten()
            .zip(b.iter().flatten())
            .all(|(x, y)| (x - y).abs() < 1e-4)
}

fn bounds_points(chaperone_mgr: &mut ChaperoneSetupManager) -> Vec<[f32; 3]> {
    chaperone_mgr
        .get_live_collision_bounds_info()
        .iter()
        .flat_map(|quad| quad.vCorners.iter().map(|corner| corner.v))
        .collect()
}
