import cv2
import os

def dataset_creation(base_folder):
    """
    Processes video files in participant subfolders, extracts frames, and stores them in organized train, val, test folders.
    Args:
    - base_folder (str): The base directory containing participant folders. Each participant folder should contain
                         subfolders for TSV (Hot, Warm, etc.), each with a .MP4 file.
    """
    if not os.path.exists(base_folder):
        raise FileNotFoundError(f"The base folder {base_folder} does not exist.")

    participants_list = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    if not participants_list:
        print(f"No participant folders found in {base_folder}.")
        return
    TSVs = ['Hot', 'Warm', 'Slightly Warm', 'Neutral', 'Slightly Cool', 'Cool', 'Cold']

    for participant in participants_list:
        participant_folder = os.path.join(base_folder, participant)
        print(f"Processing participant: {participant}")

        for dataset_type in ['train', 'val', 'test']:
            dataset_folder = os.path.join(participant_folder, dataset_type)
            os.makedirs(dataset_folder, exist_ok=True)

            for TSV in TSVs:
                TSV_folder = os.path.join(dataset_folder, TSV)
                os.makedirs(TSV_folder, exist_ok=True)

        for TSV in TSVs:
            TSV_folder_path = os.path.join(participant_folder, TSV)
            if not os.path.isdir(TSV_folder_path):
                print(f"Condition folder {TSV_folder_path} not found. Skipping.")
                continue
            video_files = [f for f in os.listdir(TSV_folder_path) if f.lower().endswith('.mp4')]
            if not video_files:
                print(f"No .MP4 files found in {TSV_folder_path}. Skipping.")
                continue
            video_file_path = os.path.join(TSV_folder_path, video_files[0])
            video_reader = cv2.VideoCapture(video_file_path)
            if not video_reader.isOpened():
                print(f"Could not open video file {video_file_path}. Skipping.")
                continue
            video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video_reader.get(cv2.CAP_PROP_FPS)
            print(f"Processing video: {video_file_path}")
            print(f"Total frames: {video_frames_count}, FPS: {fps}")
            train_end = int(0.7 * video_frames_count)
            val_end = int(0.8 * video_frames_count)
            frame_count = 0
            for frame_counter in range(video_frames_count):
                success, frame = video_reader.read()
                if not success:
                    print(f"Failed to read frame {frame_counter}. Continuing.")
                    continue
                if frame_counter <= train_end:
                    folder_type = 'train'
                elif frame_counter <= val_end:
                    folder_type = 'val'
                else:
                    folder_type = 'test'
                frame_folder = os.path.join(participant_folder, folder_type, TSV)
                frame_name = f"{TSV}_{participant}_{frame_count}.jpg"
                frame_path = os.path.join(frame_folder, frame_name)
                cv2.imwrite(frame_path, frame)
                frame_count += 1
            video_reader.release()
            print(f"Extracted {frame_count} frames for {TSV} condition of participant {participant}.")
    print("Dataset creation completed.")

# Example usage
base_folder_path = '/path/to/base/folder'  # Replace with your actual path
dataset_creation(base_folder_path)