import argparse
from util import abs_path
from converter import titles_to_jsonl_num, convert_label
from fever_io import load_doclines, read_jsonl, save_jsonl, get_evidence_sentence_list
from tqdm import tqdm
from jack import readers
from jack.core import QASetting

# make everything deterministic
import random
import numpy as np
import tensorflow as tf
tf.set_random_seed(1)
np.random.seed(1)
random.seed(1)


def read_ir_result(path, prependlinum=False, concatev=False):
    """
    Returns
    instances: list of dictionary
    update instance['predicted_sentences'] with list of evidences (list of str)
    """
    instances = read_jsonl(path)
    t2jnum = titles_to_jsonl_num(
        wikipedia_dir=abs_path("data/wiki-pages/wiki-pages/"),
        doctitles=abs_path("data/doctitles"))
    titles = list()

    # make list of titles
    for instance in instances:
        titles.extend([title for title, _ in instance["predicted_sentences"]])

    # load title2line2sentences
    t2l2s = load_doclines(titles, t2jnum)

    for instance in instances:
        if concatev:
            instance["evidence"] = [" ".join(get_evidence_sentence_list(
                instance["predicted_sentences"], t2l2s, prependlinum=prependlinum))]
        else:
            instance["evidence"] = get_evidence_sentence_list(
                instance["predicted_sentences"], t2l2s, prependlinum=prependlinum)

    return instances


def aggregate_preds(prediction, only_use_topev=False):
    """return the most popular verdict
    """
    vote = dict()
    pred_list = [pred[0].text for pred in prediction]
    scores = [float(pred[0].score) for pred in prediction]

    for rank, pred in enumerate(pred_list):
        if pred not in vote:
            vote[pred] = 1
        else:
            vote[pred] += 1

    supports = convert_label("SUPPORTS")
    refutes = convert_label("REFUTES")
    nei = convert_label("NOT ENOUGH INFO")

    # believe more-likely evidence if both supports and refutes appears in the pred_list
    if supports in vote and refutes in vote:
        for pred in pred_list:
            if pred in [supports, refutes]:
                final_verdict = pred
                break
    elif supports in vote:
        final_verdict = supports
    elif refutes in vote:
        final_verdict = refutes
    else:
        final_verdict = nei

    if only_use_topev:
        final_verdict = pred_list[0]

    return (final_verdict, scores, pred_list)


def reshape(preds_list, preds_length):
    """reshape prediction instances
    >> preds_list = [obj, obj, obj, obj, obj, obj]
    >> preds_length = [3, 1, 2]
    >> reshape(preds_list, preds_length)
    [[obj, obj, obj], [obj], [obj, obj]]
    """
    reshaped = list()
    pointer = 0
    for i, length in enumerate(preds_length):
        preds = preds_list[pointer: pointer + length]
        pointer += length
        reshaped.append(preds)
    return reshaped


def flatten(bumpy_2d_list):
    flattened = list()
    for list_ in bumpy_2d_list:
        flattened.extend(list_)
    return flattened


if __name__ == "__main__":
    parser = argparse.ArgumentParser("read claim/evidence and output verdict")
    parser.add_argument(
        "in_file",
        help="input file path for rte (e.g., dev.sentences.p5.s5.jsonl)")
    parser.add_argument("out_file", help="output prediction file")
    parser.add_argument(
        "--saved_reader", help="path to saved reader directory")
    parser.add_argument(
        "--concatev", action="store_true", help="concat evidences")
    parser.add_argument(
        "--prependlinum", action="store_true", help="prepend linum when perform get_evidence_sentence_list")
    parser.add_argument("--only_use_topev", action="store_true", help="only use top evidence for prediction")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for inference")
    args = parser.parse_args()

    print("loading reader from file:", args.saved_reader)
    dam_reader = readers.reader_from_file(args.saved_reader, dropout=0.0)

    results = list()
    preds_length = list()
    all_settings = list()
    instances = read_ir_result(args.in_file, prependlinum=args.prependlinum, concatev=args.concatev)
    for instance in instances:
        evidence_list = instance["evidence"]
        claim = instance["claim"]
        settings = [QASetting(question=claim, support=[evidence]) for evidence in evidence_list]
        all_settings.append(settings)

    # pointer loops from 0 to less than (or equal to) len(all_settings) with step args.batch_size
    preds_list = list()
    for pointer in tqdm(range(0, len(all_settings), args.batch_size)):
        batch_settings = all_settings[pointer: pointer + args.batch_size]
        n_settings = [len(settings_) for settings_ in batch_settings]
        preds_list.extend(reshape(dam_reader(flatten(batch_settings)), n_settings))

    results = list()
    for instance, preds in zip(instances, preds_list):
        prediction, scores, prediction_list = aggregate_preds(preds, args.only_use_topev)
        results.append({
            "actual": instance["label"],
            "predicted":
            convert_label(prediction, inverse=True),
            "scores":
            scores,
            "prediction_list":
            [convert_label(pred, inverse=True) for pred in prediction_list]
        })
    save_jsonl(results, abs_path(args.out_file))
