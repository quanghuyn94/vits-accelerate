import argparse
import text
from libs.utils import load_filepaths_and_text
import whisper
import glob
import os
import tqdm
from pydub import AudioSegment

if os.name == "nt":
    paths = os.getenv("PATH").split(";")
    espeak_path = ""
    
    for path in paths:
        # print(path)
        if 'espeak' in path.lower():
            espeak_path = path
            break

    assert espeak_path != "", "Espeak not install."
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = os.path.join(espeak_path, "libespeak-ng.dll")

def convert_to_mono(input_path, output_path, target_sample_rate):
    # Load audio file
    sound = AudioSegment.from_file(input_path)

    # Convert to mono
    mono_sound = sound.set_channels(1)

    # Resample to the target sample rate
    resampled_sound = mono_sound.set_frame_rate(target_sample_rate)

    # Export the mono audio to the output path
    resampled_sound.export(output_path, format="wav")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_extension", default="cleaned")
    parser.add_argument("--text_index", default=1, type=int)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--src_lang", required=True)
    parser.add_argument("--save_to", required=True)
    parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])
    parser.add_argument("--sample_rate", default=22050)
    args = parser.parse_args()
    
    audio_paths = glob.glob(args.data_dir + "/**/*.wav", recursive=True)
    model = whisper.load_model('base')

    # for filelist in args.audio_paths:
    #   print("START:", filelist)
    #   filepaths_and_text = load_filepaths_and_text(filelist)
    #   for i in range(len(filepaths_and_text)):
    #     original_text = filepaths_and_text[i][args.text_index]
    #     cleaned_text = text._clean_text(original_text, args.text_cleaners)
    #     filepaths_and_text[i][args.text_index] = cleaned_text

    #   new_filelist = filelist + "." + args.out_extension
    #   with open(new_filelist, "w", encoding="utf-8") as f:
    #     f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])
    for i, audio_path in tqdm.tqdm(enumerate(audio_paths), total=len(audio_paths)):
        # options = whisper.DecodingOptions(language=args.src_lang)
        base_path = os.path.splitext(audio_path)
        file_name = os.path.split(base_path[0])[1]

        save_to = os.path.join(args.save_to, file_name)

        result = model.transcribe(audio_path, language=args.src_lang)
        with open(save_to + ".txt", 'w', encoding='utf-8') as f:
            f.write(result['text'])

        cleaned_text = text._clean_text(result["text"], args.text_cleaners)
        
        with open(save_to + ".txt." + args.out_extension, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        convert_to_mono(audio_path, save_to +".wav", args.sample_rate)