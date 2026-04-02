from trackers import Tracker
from utils import read_video, save_video
import numpy as np

from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def run_analysis(input_video_path, output_video_path):
    # Read video
    video_frames = read_video(input_video_path)

    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False
    )

    tracker.add_position_to_tracks(tracks)

    # Camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=False
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement
    )

    # View transform
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Ball interpolation
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed & distance
    speed_estimator = SpeedAndDistance_Estimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    # Team assignment
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Ball possession
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', [])

        if not ball_bbox:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
            continue

        assigned_player = player_assigner.assign_ball_to_player(
            player_track, ball_bbox
        )

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(
                tracks['players'][frame_num][assigned_player]['team']
            )
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)

    team_ball_control = np.array(team_ball_control)

    # Draw output
    output_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control
    )

    output_frames = camera_movement_estimator.draw_camera_movement(
        output_frames, camera_movement
    )

    speed_estimator.draw_speed_and_distance(output_frames, tracks)

    # Save result
    save_video(output_frames, output_video_path)
