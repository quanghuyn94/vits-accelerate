import argparse
import text
from libs.utils import load_filepaths_and_text
import whisper
import glob
import os
import tqdm


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_extension", default="cleaned")
    parser.add_argument("--text_index", default=1, type=int)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--src_lang", required=True)
    parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])

    args = parser.parse_args()
    
    audio_paths = glob.glob(args.data_dir + "/*.wav")
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

        result = model.transcribe(audio_path, language=args.src_lang)
        with open(base_path[0] + ".txt", 'w', encoding='utf-8') as f:
            f.write(result['text'])

        cleaned_text = text._clean_text(result["text"], args.text_cleaners)
        
        with open(base_path[0] + ".txt." + args.out_extension, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)