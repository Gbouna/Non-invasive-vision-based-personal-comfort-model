# main.py
import os
import argparse
from train.train import Trainer

def main():
    """
    Parse arguments and loop through the base directory to train the model for each participant.
    """
    parser = argparse.ArgumentParser(description="Train CNN model on participant data")
    parser.add_argument('--base_dir', required=True, type=str, help='Base directory containing participant folders')
    parser.add_argument('--num_classes', required=True, type=int, choices=[3, 7], help='Number of classes (3 or 7)')
    args = parser.parse_args()
    base_folder = args.base_dir
    num_classes = args.num_classes
    participants = [p for p in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, p))]

    # Create Trainer instance
    trainer = Trainer(base_dir=base_folder, num_classes=num_classes)
    for participant in participants:
        print(f"Training model for participant: {participant}")
        trainer.train_model_for_participant(participant)
        print(f"Finished training for participant: {participant}")

if __name__ == '__main__':
    main()
