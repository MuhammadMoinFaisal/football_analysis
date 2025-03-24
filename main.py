#Import All the Required Libraries
from utils import read_video, save_video
from trackers import Tracker
def main():
    #Read Video
    video_frames = read_video("input_videos/video.mp4")

    #Initialize Tracker
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path='tracker_stubs/player_detection.pkl')

    #Draw Output
    #Draw Object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    #Save Video
    save_video(output_video_frames, 'output_videos/output.avi')


if __name__ == "__main__":
    main()