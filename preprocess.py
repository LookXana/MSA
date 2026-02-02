import argparse
import json
import glob
import logging
import os
import pickle
import random

import pandas as pd
import soundfile as sf
import tqdm
import numpy as np
import torch
from moviepy.editor import VideoFileClip
from sklearn.model_selection import train_test_split

#1️⃣ 随机种子（保证可复现）
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
#2️⃣ 情感标签映射
LABEL_MAP = {
    "ang": 0,
    "hap": 1,
    "sad": 2,
    "neu": 3,
}

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 工具函数：视频 → 音频
# export_mp4_to_audio
def export_mp4_to_audio(
    mp4_file: str,
    wav_file: str,
    verbose: bool = False,
):
    """Convert mp4 file to wav file

    Args:
        mp4_file (str): Path to mp4 input file
        wav_file (str): Path to wav output file
        verbose (bool, optional): Whether to print ffmpeg output. Defaults to False.
    """
    try:
        video = VideoFileClip(mp4_file)
    except:
        logging.warning(f"Failed to load {mp4_file}")
        return 0
    audio = video.audio
    audio.write_audiofile(wav_file, verbose=verbose)
    return 1

#IEMOCAP 的处理逻辑（最复杂，重点）
#1️⃣ IEMOCAP 目录结构假设
#SessionX/
# ├── sentences/wav/        # 音频
# ├── dialog/transcriptions # 文本
# └── dialog/EmoEvaluation  # 情感标注

def preprocess_IEMOCAP(args):
    data_root = args.data_root
    ignore_length = args.ignore_length
#3️⃣ 遍历 Session 1–5
    session_id = list(range(1, 6))
#2️⃣ 标签扩展（exc → hap）
    samples = []
    labels = []
    iemocap2label = LABEL_MAP
    iemocap2label.update({"exc": 1})

    for sess_id in tqdm.tqdm(session_id):
        sess_path = os.path.join(data_root, "Session{}".format(sess_id))
        sess_autio_root = os.path.join(sess_path, "sentences/wav")
        sess_text_root = os.path.join(sess_path, "dialog/transcriptions")
        sess_label_root = os.path.join(sess_path, "dialog/EmoEvaluation")
        label_paths = glob.glob(os.path.join(sess_label_root, "*.txt"))
        for l_path in label_paths:
            l_name = os.path.basename(l_path)
            transcripts_path = os.path.join(sess_text_root, l_name)
            with open(transcripts_path, "r") as f:
                transcripts = f.readlines()
                #4️⃣ 读取文本转录文件
                transcripts = {
                    t.split(":")[0]: t.split(":")[1].strip() for t in transcripts
                }
            with open(l_path, "r") as f:
                label = f.read().split("\n")
                for l in label:
                    #5️⃣ 解析情感标注文件（核心）
                    #6️⃣ 音频读取 + 过滤短样本
                    if str(l).startswith("["):
                        data = l[1:].split()
                        wav_folder = data[3][:-5]
                        wav_name = data[3] + ".wav"
                        emo = data[4]
                        wav_path = os.path.join(sess_autio_root, wav_folder, wav_name)
                        wav_data, _ = sf.read(wav_path, dtype="int16")
                        # Ignore samples with length < ignore_length
                        if len(wav_data) < ignore_length:
                            logging.warning(
                                f"Ignoring sample {wav_path} with length {len(wav_data)}"
                            )
                            continue
                        emo = iemocap2label.get(emo, None)
                        text_query = data[3] + " [{:08.4f}-{:08.4f}]".format(
                            float(data[0]), float(data[2][:-1])
                        )
                        if emo is not None:
                            text = transcripts.get(text_query, None)
                            if text is None:
                                #7️⃣ 对齐文本（最 tricky 的地方）
                                text_query = data[3] + " [{:08.4f}-{:08.4f}]".format(
                                    float(data[0]), (float(data[2][:-1]) + 0.0001)
                                )
                                text = transcripts.get(text_query, None)
                                if text is None:
                                    text_query = data[
                                        3
                                    ] + " [{:08.4f}-{:08.4f}]".format(
                                        float(data[0]) + 0.0001, float(data[2][:-1])
                                    )
                                    text = transcripts.get(text_query, None)
                                    if text is None:
                                        print(transcripts.keys())
                                        print(text_query)
                                        raise Exception
                            #8️⃣ 最终样本格式     
                            samples.append((wav_path, text, emo))
                            labels.append(emo)

    # Shuffle and split
    temp = list(zip(samples, labels))
    random.Random(args.seed).shuffle(temp)
    samples, labels = zip(*temp)
    train, test_samples, train_labels, _ = train_test_split(
        samples, labels, test_size=0.1, random_state=args.seed
    )
    train_samples, val_samples, _, _ = train_test_split(
        train, train_labels, test_size=0.1, random_state=args.seed
    )
    # Save data
    os.makedirs(args.dataset + "_preprocessed", exist_ok=True)
    with open(os.path.join(args.dataset + "_preprocessed", "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(args.dataset + "_preprocessed", "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(args.dataset + "_preprocessed", "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(f"Train samples: {len(train_samples)}")
    logging.info(f"Val samples: {len(val_samples)}")
    logging.info(f"Test samples: {len(test_samples)}")
    logging.info(f"Saved to {args.dataset + '_preprocessed'}")
    logging.info("Preprocessing finished successfully")
#ESD 数据集处理（最简单）
#SpeakerX/
# ├── Angry/
# ├── Happy/
# ├── Neutral/
# ├── Sad/
# └── SpeakerX.txt

def preprocess_ESD(args):
    esd2label = {
        "Angry": "ang",
        "Happy": "hap",
        "Neutral": "neu",
        "Sad": "sad",
    }
#标注文件格式
    directory = glob.glob(args.data_root + "/*")
    samples = []
    labels = []

    # Loop through all folders
    for dir in tqdm.tqdm(directory):
        # Read label file
        label_path = os.path.join(dir, dir.split("/")[-1] + ".txt")
        with open(label_path, "r") as f:
            label = f.read().strip().splitlines()
        # Extract samples from label file
        for l in label:
            filename, transcript, emotion = l.split("\t")
            target = esd2label.get(emotion, None)
            if target is not None:
                #核心逻辑，不需要时间戳、不需要对齐，非常干净
                samples.append(
                    (
                        os.path.abspath(os.path.join(dir, emotion, filename + ".wav")),
                        transcript,
                        LABEL_MAP[target],
                    )
                )
                # Labels are use for splitting
                labels.append(LABEL_MAP[target])

    # Shuffle and split
    temp = list(zip(samples, labels))
    random.Random(args.seed).shuffle(temp)
    samples, labels = zip(*temp)
    train, test_samples, train_labels, _ = train_test_split(
        samples, labels, test_size=0.2, random_state=args.seed
    )
    train_samples, val_samples, _, _ = train_test_split(
        train, train_labels, test_size=0.1, random_state=args.seed
    )

    # Save data
    os.makedirs(args.dataset + "_preprocessed", exist_ok=True)
    with open(os.path.join(args.dataset + "_preprocessed", "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(args.dataset + "_preprocessed", "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(args.dataset + "_preprocessed", "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(f"Train samples: {len(train_samples)}")
    logging.info(f"Train samples: {len(val_samples)}")
    logging.info(f"Test samples: {len(test_samples)}")
    logging.info(f"Saved to {args.dataset + '_preprocessed'}")
    logging.info("Preprocessing finished successfully")

#MELD 数据集处理（视频为主）
def preprocess_MELD(args):
    meld2label = {
        "anger": "ang",
        "joy": "hap",
        "neutral": "neu",
        "sadness": "sad",
    }

    train_csv = os.path.join(args.data_root, "train_sent_emo.csv")
    val_csv = os.path.join(args.data_root, "dev_sent_emo.csv")
    test_csv = os.path.join(args.data_root, "test_sent_emo.csv")

    train_dataframe = pd.read_csv(train_csv)
    val_dataframe = pd.read_csv(val_csv)
    test_dataframe = pd.read_csv(test_csv)
    #3️⃣ 支持 --all_classes
    if args.all_classes:
        meld2label = {}
        LABEL_MAP = {}
        labels = []
        for _, row in test_dataframe.iterrows():
            labels.append(row.Emotion)
        labels = list(set(labels))
        for i, label_name in enumerate(labels):
            meld2label[label_name] = i
        for i in range(len(labels)):
            LABEL_MAP[i] = i
      # IEMOCAP / ESD  
        # Save labels
        os.makedirs(args.dataset + "_preprocessed", exist_ok=True)
        with open(os.path.join(args.dataset + "_preprocessed", "classes.json"), "w") as f:
            json.dump(meld2label, f)

    def _preprocess_data(data_path, dataframe):
        samples = []
        # Loop through all folders
        for _, row in tqdm.tqdm(dataframe.iterrows()):
            # Read label file
            label = meld2label.get(row.Emotion, None)
            if label is None:
                continue
            transcript = str(row.Utterance)
            video_path = os.path.abspath(
                os.path.join(
                    data_path, f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}.mp4"
                )
            )
            audio_path = os.path.abspath(
                os.path.join(
                    data_path, f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}.wav"
                )
            )
            # Convert video to audio
            try:
                videoclip = VideoFileClip(video_path)
                videoclip.audio.write_audiofile(audio_path, verbose=False)
                samples.append((audio_path, transcript, LABEL_MAP[label]))
            except:
                logging.warn(f"Can not preprocess video data: {video_path}")

        return samples

    train_samples = _preprocess_data(
        os.path.join(args.data_root, "train_splits"), train_dataframe
    )
    val_samples = _preprocess_data(
        os.path.join(args.data_root, "dev_splits_complete"), val_dataframe
    )
    test_samples = _preprocess_data(
        os.path.join(args.data_root, "output_repeated_splits_test"), test_dataframe
    )

    # Save data
    os.makedirs(args.dataset + "_preprocessed", exist_ok=True)
    with open(os.path.join(args.dataset + "_preprocessed", "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(args.dataset + "_preprocessed", "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(args.dataset + "_preprocessed", "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(f"Train samples: {len(train_samples)}")
    logging.info(f"Train samples: {len(val_samples)}")
    logging.info(f"Test samples: {len(test_samples)}")
    logging.info(f"Saved to {args.dataset + '_preprocessed'}")
    logging.info("Preprocessing finished successfully")


def main(args):
    preprocess_fn = {
        "IEMOCAP": preprocess_IEMOCAP,
        "ESD": preprocess_ESD,
        "MELD": preprocess_MELD,
    }

    preprocess_fn[args.dataset](args)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ds", "--dataset", type=str, default="ESD", choices=["IEMOCAP", "ESD", "MELD"]
    )
    parser.add_argument(
        "-dr",
        "--data_root",
        type=str,
        help="Path to folder containing IEMOCAP data",
        required=True,
    )
    parser.add_argument("--all_classes", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ignore_length",
        type=int,
        default=0,
        help="Ignore samples with length < ignore_length",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
#该脚本实现了对 IEMOCAP、ESD 与 MELD 数据集的统一预处理流程，通过对音频、文本和情感标签的精确对齐与过滤，构建标准化的 (audio, text, label) 样本格式，并完成可复现的数据划分，为后续多模态情感识别模型训练提供一致的数据接口。
