import os
from dotenv import load_dotenv
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# This code is meant to convert only the .wav and .ogg files into mel-specrogram images
# Load environment variables from the .env file
load_dotenv()

def create_spectrogram(audio_path, output_path, n_mels=128, fmax=8000):
    """
    Loads an audio file, computes its mel-spectrogram, and saves it as an image.
    """
    # Preserves the original sample rate)
    y, sr = librosa.load(audio_path, sr=None)
    
    # Compute the mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Plot the mel-spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    
    # Save the spectrogram image and close the figure to free up memory
    plt.savefig(output_path, dpi=300)
    plt.close()

def process_folder(input_folder, output_folder):
    """
    Processes all WAV and OGG files in input_folder and saves their mel-spectrogram
    images into output_folder.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Loops over each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.wav', '.ogg')):
            audio_path = os.path.join(input_folder, filename)
            base, _ = os.path.splitext(filename)
            output_file = base + ".png"
            output_path = os.path.join(output_folder, output_file)
            
            print(f"Processing {filename} -> {output_file}")
            create_spectrogram(audio_path, output_path)

if __name__ == "__main__":
    # Gets folders from environment variables
    input_folder = os.environ.get('INPUT_FOLDER')
    output_folder = os.environ.get('OUTPUT_FOLDER')
    
    # Check if the environment variables were loaded properly
    if not input_folder or not output_folder:
        raise ValueError("Please ensure INPUT_FOLDER and OUTPUT_FOLDER are set in your .env file.")
    
    process_folder(input_folder, output_folder)
    print("Spectrogram images saved to:", output_folder)
