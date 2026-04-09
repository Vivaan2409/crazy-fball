import os
import hashlib
import time
import threading
import subprocess
import shutil
import uuid
import gc
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from pathlib import Path
import tempfile

# Import your existing modules
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'football-analysis-secret-key-2024')
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'static/uploads')
app.config['PROCESSED_FOLDER'] = os.environ.get('PROCESSED_FOLDER', 'static/processed')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['FFMPEG_PATH'] = os.environ.get('FFMPEG_PATH', 'ffmpeg')

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'], 'stubs']:
    os.makedirs(folder, exist_ok=True)

# Enhanced save_video function for web compatibility
def save_video_web_compatible(frames, output_path, fps=24):
    """Save video with web-compatible encoding"""
    if not frames:
        raise ValueError("No frames to save")
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Create a temporary file first
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f'temp_{uuid.uuid4()}.mp4')
    
    # Try different codecs in order of preference for web compatibility
    codec_attempts = [
        ('libx264', cv2.VideoWriter_fourcc(*'mp4v')),  # H.264 via mp4v
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),     # XVID
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),     # Motion JPEG
    ]
    
    writer = None
    used_codec = None
    
    for codec_name, fourcc in codec_attempts:
        try:
            writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            if writer.isOpened():
                used_codec = codec_name
                print(f"Using codec: {codec_name}")
                break
        except Exception as e:
            print(f"Codec {codec_name} failed: {e}")
            continue
    
    if writer is None:
        # Last resort: use default
        writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        used_codec = 'mp4v (fallback)'
    
    # Write frames with memory management
    for i, frame in enumerate(frames):
        writer.write(frame)
        if i % 50 == 0:
            gc.collect()
    
    writer.release()
    
    # Convert to web-optimized format using ffmpeg if available
    try:
        ffmpeg_cmd = [
            app.config['FFMPEG_PATH'], '-i', temp_path,
            '-c:v', 'libx264',          # H.264 codec (web standard)
            '-preset', 'fast',          # Faster encoding
            '-crf', '23',               # Quality (23 is good)
            '-pix_fmt', 'yuv420p',      # Compatibility with all browsers
            '-movflags', '+faststart',  # Optimize for web streaming
            '-y',                       # Overwrite output
            output_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"Video successfully converted to web format: {output_path}")
        else:
            print(f"FFmpeg conversion failed, using OpenCV output: {result.stderr}")
            # Copy the OpenCV output as fallback
            shutil.copy2(temp_path, output_path)
    
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"FFmpeg not available or failed, using OpenCV output: {e}")
        # Copy the OpenCV output
        shutil.copy2(temp_path, output_path)
    
    # Clean up temporary file
    try:
        os.remove(temp_path)
    except:
        pass
    
    # Clear memory
    gc.collect()
    
    return used_codec

def convert_video_for_web(input_path, output_path):
    """Convert any video to web-compatible MP4 with H.264"""
    if not os.path.exists(input_path):
        return False
    
    try:
        ffmpeg_cmd = [
            app.config['FFMPEG_PATH'], '-i', input_path,
            '-c:v', 'libx264',          # H.264 video codec
            '-preset', 'medium',        # Balance speed/quality
            '-crf', '22',               # Quality (lower = better)
            '-c:a', 'aac',              # AAC audio codec
            '-b:a', '128k',             # Audio bitrate
            '-pix_fmt', 'yuv420p',      # Pixel format for compatibility
            '-movflags', '+faststart',  # Web optimization
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure even dimensions
            '-y',                       # Overwrite output
            output_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print(f"Video converted successfully: {output_path}")
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("Video conversion timeout")
        return False
    except FileNotFoundError:
        print("FFmpeg not found. Please install ffmpeg for better video compatibility.")
        # Try to copy as-is
        try:
            shutil.copy2(input_path, output_path)
            return True
        except:
            return False
    except Exception as e:
        print(f"Conversion error: {e}")
        return False

class VideoProcessor:
    def __init__(self):
        self.processing_status = {}
        self.results = {}
    
    def generate_video_hash(self, video_path):
        """Generate a unique hash for the video file"""
        hasher = hashlib.sha256()
        try:
            with open(video_path, 'rb') as f:
                buf = f.read(65536)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = f.read(65536)
            return hasher.hexdigest()
        except Exception as e:
            print(f"Error generating hash: {e}")
            return str(uuid.uuid4())
    
    def allowed_file(self, filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    
    def read_video_with_memory_management(self, video_path, max_frames=None, scale_factor=0.5):
        """Read video with memory optimization"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if max_frames and max_frames < total_frames:
            total_frames = max_frames
        
        print(f"Reading video: {video_path}")
        print(f"FPS: {fps}, Total frames to read: {total_frames}")
        print(f"Scaling factor: {scale_factor}")
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Scale down frame to save memory
            if scale_factor != 1.0:
                h, w = frame.shape[:2]
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                frame = cv2.resize(frame, (new_w, new_h))
            
            frames.append(frame)
            frame_count += 1
            
            # Clear memory every 50 frames
            if frame_count % 50 == 0:
                gc.collect()
                print(f"Read {frame_count}/{total_frames} frames")
        
        cap.release()
        print(f"Loaded {len(frames)} frames")
        
        if frames:
            h, w = frames[0].shape[:2]
            print(f"Frame dimensions: {w}x{h}")
        
        return frames
    
    def process_video(self, video_id, video_path, use_stubs=False):
        """Process video with memory optimization"""
        try:
            self.processing_status[video_id] = {
                'status': 'processing',
                'progress': 0,
                'message': 'Starting video processing...',
                'start_time': time.time()
            }
            
            # Generate unique hash for this video
            video_hash = self.generate_video_hash(video_path)
            
            # Check video size and adjust parameters
            video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            scale_factor = 1.0
            
            # Auto-adjust scale factor based on video size
            if video_size_mb > 200:  # Large video
                scale_factor = 0.5  # Scale to 50%
                print(f"Large video detected ({video_size_mb:.1f} MB). Scaling to {scale_factor*100}%")
            elif video_size_mb > 100:  # Medium video
                scale_factor = 0.75  # Scale to 75%
                print(f"Medium video detected ({video_size_mb:.1f} MB). Scaling to {scale_factor*100}%")
            
            # Read Video with memory optimization
            self.processing_status[video_id]['message'] = 'Reading video frames with memory optimization...'
            self.processing_status[video_id]['progress'] = 5
            
            video_frames = self.read_video_with_memory_management(video_path, scale_factor=scale_factor)
            
            if len(video_frames) == 0:
                raise ValueError("No frames loaded from the video.")
            
            self.processing_status[video_id]['progress'] = 10
            self.processing_status[video_id]['message'] = f'Loaded {len(video_frames)} frames'
            
            # Initialize Tracker
            tracker = Tracker('models/best.pt')
            
            # Generate stub paths based on video hash to prevent mismatch
            if use_stubs:
                track_stub_path = f'stubs/track_stubs_{video_hash}.pkl'
                camera_stub_path = f'stubs/camera_movement_stub_{video_hash}.pkl'
            else:
                track_stub_path = None
                camera_stub_path = None
            
            # Process in chunks to save memory
            chunk_size = 50
            all_tracks = {
                "players": [],
                "referees": [],
                "ball": []
            }
            
            # Initialize empty tracks for all frames
            for _ in range(len(video_frames)):
                all_tracks["players"].append({})
                all_tracks["referees"].append({})
                all_tracks["ball"].append({})
            
            # Process video in chunks
            for chunk_start in range(0, len(video_frames), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(video_frames))
                chunk_frames = video_frames[chunk_start:chunk_end]
                
                # Update progress
                progress = 10 + (chunk_end / len(video_frames)) * 20
                self.processing_status[video_id]['progress'] = progress
                self.processing_status[video_id]['message'] = f'Tracking objects in chunk {chunk_start//chunk_size + 1}/{(len(video_frames)+chunk_size-1)//chunk_size}'
                
                # Get object tracks for this chunk
                chunk_tracks = tracker.get_object_tracks(chunk_frames, read_from_stub=False, stub_path=None)
                
                # Add positions
                tracker.add_position_to_tracks(chunk_tracks)
                
                # Merge chunk tracks into all_tracks
                for i in range(len(chunk_frames)):
                    frame_idx = chunk_start + i
                    
                    if "players" in chunk_tracks and i < len(chunk_tracks["players"]):
                        all_tracks["players"][frame_idx] = chunk_tracks["players"][i]
                    if "referees" in chunk_tracks and i < len(chunk_tracks["referees"]):
                        all_tracks["referees"][frame_idx] = chunk_tracks["referees"][i]
                    if "ball" in chunk_tracks and i < len(chunk_tracks["ball"]):
                        all_tracks["ball"][frame_idx] = chunk_tracks["ball"][i]
                
                # Clear memory
                del chunk_frames, chunk_tracks
                gc.collect()
            
            tracks = all_tracks
            
            # Camera movement estimator
            self.processing_status[video_id]['message'] = 'Estimating camera movement...'
            camera_movement_estimator = CameraMovementEstimator(video_frames[0])
            
            # Use a subset of frames for camera movement to save memory
            if len(video_frames) > 100:
                sample_indices = np.linspace(0, len(video_frames)-1, 100, dtype=int)
                sample_frames = [video_frames[i] for i in sample_indices]
            else:
                sample_frames = video_frames
            
            camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
                sample_frames,
                read_from_stub=use_stubs,
                stub_path=camera_stub_path
            )
            
            # Interpolate for all frames
            if len(camera_movement_per_frame) < len(video_frames):
                # Extend last movement to all frames
                last_movement = camera_movement_per_frame[-1] if camera_movement_per_frame else [0, 0]
                camera_movement_per_frame = camera_movement_per_frame + [last_movement] * (len(video_frames) - len(camera_movement_per_frame))
            
            camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
            self.processing_status[video_id]['progress'] = 40
            
            # View Transformer
            view_transformer = ViewTransformer()
            view_transformer.add_transformed_position_to_tracks(tracks)
            
            # Interpolate Ball Positions
            tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
            
            # Speed and distance estimator
            self.processing_status[video_id]['message'] = 'Calculating speed and distance...'
            speed_and_distance_estimator = SpeedAndDistance_Estimator()
            speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
            self.processing_status[video_id]['progress'] = 50
            
            # Assign Player Teams
            self.processing_status[video_id]['message'] = 'Assigning teams...'
            team_assigner = TeamAssigner()
            
            # Find a good frame for team color detection
            selected_frame_idx = None
            min_players_required = 4
            
            for i in range(len(tracks['players'])):
                if len(tracks['players'][i]) >= min_players_required:
                    selected_frame_idx = i
                    break
            
            if selected_frame_idx is None:
                # Fallback to frame with max players
                max_players = 0
                for i in range(len(tracks['players'])):
                    num = len(tracks['players'][i])
                    if num > max_players:
                        max_players = num
                        selected_frame_idx = i
            
            if selected_frame_idx is not None:
                team_assigner.assign_team_color(
                    video_frames[selected_frame_idx],
                    tracks['players'][selected_frame_idx]
                )
                
                # Assign team to every player in chunks
                for chunk_start in range(0, len(tracks['players']), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(tracks['players']))
                    
                    for frame_num in range(chunk_start, chunk_end):
                        player_track = tracks['players'][frame_num]
                        for player_id, track in player_track.items():
                            if 'bbox' not in track or not track['bbox']:
                                continue
                            team = team_assigner.get_player_team(
                                video_frames[frame_num],
                                track['bbox'],
                                player_id
                            )
                            tracks['players'][frame_num][player_id]['team'] = team
                            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors.get(team, (0, 0, 255))
                    
                    # Clear memory
                    gc.collect()
            
            self.processing_status[video_id]['progress'] = 60
            
            # Assign Ball Acquisition
            self.processing_status[video_id]['message'] = 'Analyzing ball control...'
            player_assigner = PlayerBallAssigner()
            team_ball_control = []
            
            for frame_num, player_track in enumerate(tracks['players']):
                if 1 in tracks['ball'][frame_num]:
                    ball_bbox = tracks['ball'][frame_num][1].get('bbox', [])
                else:
                    ball_bbox = []
                
                if not ball_bbox or len(ball_bbox) == 0:
                    team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
                    continue
                
                assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
                
                if assigned_player != -1 and assigned_player in tracks['players'][frame_num]:
                    tracks['players'][frame_num][assigned_player]['has_ball'] = True
                    team_ball_control.append(tracks['players'][frame_num][assigned_player].get('team', 1))
                else:
                    team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
                
                # Clear memory every 100 frames
                if frame_num % 100 == 0:
                    gc.collect()
            
            team_ball_control = np.array(team_ball_control)
            self.processing_status[video_id]['progress'] = 70
            
            # Draw annotations in chunks
            self.processing_status[video_id]['message'] = 'Generating output video...'
            output_video_frames = []
            
            for chunk_start in range(0, len(video_frames), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(video_frames))
                chunk_frames = video_frames[chunk_start:chunk_end]
                
                # Update progress
                progress = 70 + ((chunk_end / len(video_frames)) * 25)
                self.processing_status[video_id]['progress'] = progress
                self.processing_status[video_id]['message'] = f'Drawing annotations for chunk {chunk_start//chunk_size + 1}/{(len(video_frames)+chunk_size-1)//chunk_size}'
                
                chunk_tracks = {
                    "players": tracks["players"][chunk_start:chunk_end],
                    "referees": tracks["referees"][chunk_start:chunk_end],
                    "ball": tracks["ball"][chunk_start:chunk_end]
                }
                
                chunk_camera_movement = camera_movement_per_frame[chunk_start:chunk_end]
                chunk_team_ball_control = team_ball_control[chunk_start:chunk_end] if len(team_ball_control) > chunk_start else team_ball_control[-1:]
                
                # Draw annotations for this chunk
                chunk_output = tracker.draw_annotations(chunk_frames, chunk_tracks, chunk_team_ball_control)
                
                # Draw Camera movement
                chunk_output = camera_movement_estimator.draw_camera_movement(chunk_output, chunk_camera_movement)
                
                # Draw Speed and Distance
                chunk_output = speed_and_distance_estimator.draw_speed_and_distance(chunk_output, chunk_tracks)
                
                output_video_frames.extend(chunk_output)
                
                # Clear memory
                del chunk_frames, chunk_output, chunk_tracks
                gc.collect()
            
            self.processing_status[video_id]['progress'] = 95
            
            # Save output video with web compatibility
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], f'{video_id}.mp4')
            
            # Use the enhanced save function
            used_codec = save_video_web_compatible(output_video_frames, output_path, fps=24)
            
            # Convert to web format if ffmpeg is available
            web_output_path = os.path.join(app.config['PROCESSED_FOLDER'], f'{video_id}_web.mp4')
            if convert_video_for_web(output_path, web_output_path):
                # Use the web-optimized version
                os.replace(web_output_path, output_path)
            
            # Generate thumbnail from middle frame
            thumbnail_path = os.path.join(app.config['PROCESSED_FOLDER'], f'{video_id}_thumb.jpg')
            middle_frame_idx = len(video_frames) // 2
            cv2.imwrite(thumbnail_path, video_frames[middle_frame_idx])
            
            # Calculate processing time
            processing_time = time.time() - self.processing_status[video_id]['start_time']
            
            self.processing_status[video_id]['progress'] = 100
            self.processing_status[video_id]['status'] = 'completed'
            self.processing_status[video_id]['message'] = f'Processing complete! (Took {processing_time:.1f}s)'
            self.processing_status[video_id]['codec'] = used_codec
            
            # Store results
            self.results[video_id] = {
                'output_path': output_path,
                'thumbnail': thumbnail_path,
                'original_filename': os.path.basename(video_path),
                'video_hash': video_hash,
                'team_stats': {
                    'team1_control': int(np.sum(team_ball_control == 1)),
                    'team2_control': int(np.sum(team_ball_control == 2)),
                    'total_frames': len(team_ball_control),
                    'processing_time': processing_time
                },
                'video_info': {
                    'frames': len(video_frames),
                    'output_frames': len(output_video_frames),
                    'codec': used_codec,
                    'scale_factor': scale_factor
                }
            }
            
            print(f"Video {video_id} processed successfully in {processing_time:.1f} seconds")
            
            # Final memory cleanup
            del video_frames, output_video_frames, tracks
            gc.collect()
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error processing video {video_id}: {e}")
            print(f"Error details: {error_details}")
            
            self.processing_status[video_id] = {
                'status': 'error',
                'message': f'Error: {str(e)}',
                'error_details': error_details
            }
            
            # Clean up on error
            if video_id in self.results:
                del self.results[video_id]

processor = VideoProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    use_stubs = request.form.get('use_stubs', 'false').lower() == 'true'
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not processor.allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Supported: MP4, AVI, MOV, MKV, WEBM'}), 400
    
    # Generate unique ID for this video
    video_id = str(uuid.uuid4())
    
    # Save uploaded file
    original_filename = secure_filename(file.filename)
    filename = f"{video_id}_{original_filename}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(video_path)
        file_size_mb = os.path.getsize(video_path) / (1024*1024)
        print(f"Video saved: {video_path} ({file_size_mb:.1f} MB)")
        
        # Warn if video is large
        if file_size_mb > 200:
            print(f"Warning: Large video file ({file_size_mb:.1f} MB). Processing may be slower.")
        
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
    
    # Start processing in background thread
    thread = threading.Thread(
        target=processor.process_video,
        args=(video_id, video_path, use_stubs)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'video_id': video_id,
        'message': 'Video uploaded and processing started',
        'original_name': original_filename,
        'file_size_mb': f"{file_size_mb:.1f}",
        'upload_path': video_path
    })

@app.route('/status/<video_id>')
def get_status(video_id):
    if video_id in processor.processing_status:
        status = processor.processing_status[video_id].copy()
        # Calculate estimated time remaining if processing
        if status['status'] == 'processing' and 'start_time' in status:
            elapsed = time.time() - status['start_time']
            if status['progress'] > 10:  # Wait for some progress
                estimated_total = elapsed / (status['progress'] / 100)
                remaining = max(0, estimated_total - elapsed)
                status['estimated_remaining'] = f"{remaining:.0f}s"
        return jsonify(status)
    elif video_id in processor.results:
        result = processor.results[video_id]
        return jsonify({
            'status': 'completed',
            'progress': 100,
            'message': 'Processing complete',
            'download_url': f'/download/{video_id}',
            'preview_url': f'/preview/{video_id}',
            'direct_video_url': f'/video/{video_id}',
            'stats': result['team_stats'],
            'video_info': result['video_info'],
            'original_filename': result['original_filename']
        })
    else:
        return jsonify({
            'status': 'not_found',
            'message': 'Video ID not found'
        }), 404

@app.route('/preview/<video_id>')
def preview_video(video_id):
    if video_id not in processor.results:
        return render_template('error.html', 
                             message='Video not found or not processed yet.',
                             video_id=video_id), 404
    
    video_path = processor.results[video_id]['output_path']
    if not os.path.exists(video_path):
        return render_template('error.html',
                             message='Processed video file not found.',
                             video_id=video_id), 404
    
    relative_path = f'processed/{video_id}.mp4'
    static_url = f'/static/{relative_path}'
    direct_url = f'/video/{video_id}'
    
    # Get video info
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duration = frame_count / fps if fps > 0 else 0
        video_info = {
            'fps': f'{fps:.1f}',
            'frames': frame_count,
            'resolution': f'{width}x{height}',
            'duration': f'{duration:.1f}s'
        }
    except:
        video_info = {}
    
    return render_template('preview.html', 
                         video_id=video_id,
                         video_url=static_url,
                         direct_video_url=direct_url,
                         video_info=video_info)

@app.route('/video/<video_id>')
def serve_video(video_id):
    """Serve video file directly with proper headers for streaming"""
    if video_id not in processor.results:
        return jsonify({'error': 'Video not found'}), 404
    
    video_path = processor.results[video_id]['output_path']
    
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404
    
    # Determine content type based on extension
    if video_path.endswith('.mp4'):
        mimetype = 'video/mp4'
    elif video_path.endswith('.avi'):
        mimetype = 'video/x-msvideo'
    elif video_path.endswith('.mov'):
        mimetype = 'video/quicktime'
    elif video_path.endswith('.webm'):
        mimetype = 'video/webm'
    else:
        mimetype = 'application/octet-stream'
    
    # Get file size
    file_size = os.path.getsize(video_path)
    
    # Check for range header (for video seeking)
    range_header = request.headers.get('Range', None)
    
    if range_header:
        # Parse range header
        import re
        match = re.search(r'bytes=(\d+)-(\d*)', range_header)
        
        if match:
            first_byte = int(match.group(1))
            last_byte = match.group(2)
            
            if last_byte:
                last_byte = int(last_byte)
            else:
                last_byte = file_size - 1
            
            if first_byte >= file_size:
                return Response('Range Not Satisfiable', status=416)
            
            length = last_byte - first_byte + 1
            
            # Read the file in chunks
            def generate():
                with open(video_path, 'rb') as f:
                    f.seek(first_byte)
                    remaining = length
                    chunk_size = 65536  # 64KB chunks
                    
                    while remaining > 0:
                        chunk = f.read(min(chunk_size, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk
            
            rv = Response(generate(), 
                          status=206,  # Partial Content
                          mimetype=mimetype,
                          direct_passthrough=True)
            rv.headers.add('Content-Range', f'bytes {first_byte}-{last_byte}/{file_size}')
            rv.headers.add('Content-Length', str(length))
            rv.headers.add('Accept-Ranges', 'bytes')
            rv.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
            rv.headers.add('Pragma', 'no-cache')
            rv.headers.add('Expires', '0')
            return rv
    
    # If no range header, serve full file
    response = send_file(video_path, mimetype=mimetype)
    response.headers.add('Accept-Ranges', 'bytes')
    response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
    response.headers.add('Pragma', 'no-cache')
    response.headers.add('Expires', '0')
    response.headers.add('Content-Length', str(file_size))
    return response

@app.route('/download/<video_id>')
def download_video(video_id):
    if video_id not in processor.results:
        return jsonify({'error': 'Video not found'}), 404
    
    video_path = processor.results[video_id]['output_path']
    original_name = processor.results[video_id]['original_filename']
    
    # Create a download-friendly name
    name, ext = os.path.splitext(original_name)
    output_name = f"analyzed_{name}{ext if ext else '.mp4'}"
    
    return send_file(
        video_path,
        as_attachment=True,
        download_name=output_name,
        mimetype='video/mp4'
    )

@app.route('/thumbnail/<video_id>')
def get_thumbnail(video_id):
    if video_id not in processor.results:
        return jsonify({'error': 'Thumbnail not found'}), 404
    
    thumbnail_path = processor.results[video_id]['thumbnail']
    
    if not os.path.exists(thumbnail_path):
        # Generate a default thumbnail
        default_thumb = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(default_thumb, "Football", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', default_thumb)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    
    return send_file(thumbnail_path, mimetype='image/jpeg')

@app.route('/test_video/<video_id>')
def test_video(video_id):
    """Diagnostic page to test video playback"""
    if video_id not in processor.results:
        return "Video not found", 404
    
    video_path = processor.results[video_id]['output_path']
    
    if not os.path.exists(video_path):
        return "File not found", 404
    
    # Get file info
    file_size = os.path.getsize(video_path)
    file_size_mb = file_size / (1024*1024)
    
    # Try to get video info with OpenCV
    video_info = {}
    try:
        cap = cv2.VideoCapture(video_path)
        video_info['fps'] = cap.get(cv2.CAP_PROP_FPS)
        video_info['frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        video_info['duration'] = video_info['frames'] / video_info['fps'] if video_info['fps'] > 0 else 0
    except:
        video_info = {'error': 'Could not read video info'}
    
    return render_template('diagnostic.html',
                         video_id=video_id,
                         video_path=video_path,
                         file_size_mb=f"{file_size_mb:.2f}",
                         video_info=video_info)

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up old processed videos"""
    try:
        import glob
        from datetime import datetime, timedelta
        
        # Keep files from last 48 hours
        cutoff_time = time.time() - (48 * 3600)
        
        cleaned = 0
        for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER']]:
            for file_path in glob.glob(os.path.join(folder, '*')):
                try:
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
                        cleaned += 1
                except:
                    pass
        
        # Clean old stubs (keep for 1 week)
        stub_cutoff = time.time() - (7 * 24 * 3600)
        for stub_file in glob.glob('stubs/*.pkl'):
            try:
                if os.path.getmtime(stub_file) < stub_cutoff:
                    os.remove(stub_file)
                    cleaned += 1
            except:
                pass
        
        # Clear memory cache
        old_videos = []
        current_time = time.time()
        for video_id in list(processor.results.keys()):
            result = processor.results[video_id]
            if 'output_path' in result and os.path.exists(result['output_path']):
                file_time = os.path.getmtime(result['output_path'])
                if file_time < cutoff_time:
                    old_videos.append(video_id)
        
        for video_id in old_videos:
            if video_id in processor.results:
                del processor.results[video_id]
            if video_id in processor.processing_status:
                del processor.processing_status[video_id]
        
        return jsonify({
            'message': f'Cleanup completed. Removed {cleaned} old files.',
            'cleaned': cleaned,
            'remaining_results': len(processor.results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recent')
def get_recent_videos():
    """Get list of recently processed videos"""
    recent = []
    all_results = list(processor.results.items())
    
    # Sort by modification time of output file
    all_results.sort(key=lambda x: os.path.getmtime(x[1]['output_path']) 
                     if os.path.exists(x[1]['output_path']) else 0, 
                     reverse=True)
    
    for video_id, result in all_results[:10]:  # Last 10
        if os.path.exists(result['output_path']):
            recent.append({
                'id': video_id,
                'name': result['original_filename'],
                'thumbnail': f'/thumbnail/{video_id}',
                'preview_url': f'/preview/{video_id}',
                'download_url': f'/download/{video_id}',
                'size': f"{os.path.getsize(result['output_path']) / (1024*1024):.1f}MB",
                'processed': time.ctime(os.path.getmtime(result['output_path']))
            })
    
    return jsonify(recent)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'processing_jobs': len(processor.processing_status),
        'completed_results': len(processor.results),
        'upload_folder_size': f"{sum(os.path.getsize(f) for f in Path(app.config['UPLOAD_FOLDER']).rglob('*') if f.is_file()) / (1024*1024):.1f}MB",
        'processed_folder_size': f"{sum(os.path.getsize(f) for f in Path(app.config['PROCESSED_FOLDER']).rglob('*') if f.is_file()) / (1024*1024):.1f}MB"
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    for folder in ['static', 'static/uploads', 'static/processed', 'stubs', 'templates']:
        os.makedirs(folder, exist_ok=True)
    
    # Check for ffmpeg
    try:
        result = subprocess.run([app.config['FFMPEG_PATH'], '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg found: Video conversion enabled")
        else:
            print("⚠ FFmpeg not found or not working. Video compatibility may be limited.")
            print("  Install ffmpeg for better results: https://ffmpeg.org/download.html")
    except:
        print("⚠ FFmpeg not found. Install for better video compatibility.")
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    
    print("\n" + "="*50)
    print("⚽ FootballVision AI")
    print("="*50)
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Processed folder: {app.config['PROCESSED_FOLDER']}")
    print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024)}MB")
    print(f"Memory optimization: Enabled")
    print(f"Server starting on: http://localhost:{port}")
    print("="*50 + "\n")
    
    app.run(debug=debug, host='0.0.0.0', port=port, threaded=True)