# originate https://github.com/BryanPlummer/flickr30k_entities/blob/master/flickr30k_entities_utils.py

import re

import nltk
import tokenizations

# tokenizer
nltk.download("punkt")


def space_ap(
    s,
    re_ap_no_left_space=re.compile(r"([^\s])(\[\/AP#\d+-\d+\s[^\]]+])"),
    re_ap_no_right_space=re.compile(r"(\[\/AP#\d+-\d+\s[^\]]+])([^\s])"),
):
    s = re_ap_no_left_space.sub(r"\1 \2", s)
    s = re_ap_no_right_space.sub(r"\1 \2", s)
    return s


def get_sentence_data(sent_with_ap):
    """example input:

    [/AP#3-1 Cut] burdock into small pieces and [/AP#4-1 soak] in water.
    """
    sent_with_ap = space_ap(sent_with_ap)
    first_word = []
    phrases = []
    ap_ids = []
    segments = []
    current_phrase = []
    add_to_phrase = False
    for token in sent_with_ap.split():
        if add_to_phrase:
            if token[-1] == "]":
                add_to_phrase = False
                token = token[:-1]
                current_phrase.append(token)
                phrases.append(" ".join(current_phrase))
                current_phrase = []
            else:
                current_phrase.append(token)

            segments.append(token)
        else:
            if token[0] == "[":
                add_to_phrase = True
                first_word.append(len(segments))
                ap_id = token.split("#")[-1]
                ap_ids.append(ap_id)
            else:
                segments.append(token)

    # tokenization
    sent = " ".join(segments)
    words = nltk.word_tokenize(sent)

    # segments to words alignment
    seg2wrd = tokenizations.get_alignments(segments, words)[0]

    sentence_data = {"sentence": " ".join(words), "phrases": []}
    for index, phrase, ap_id in zip(first_word, phrases, ap_ids):
        sentence_data["phrases"].append(
            {"index": seg2wrd[index][0], "phrase": phrase, "ap_id": ap_id}
        )

    return sentence_data


if __name__ == "__main__":
    print(
        get_sentence_data("[/AP#3-1 Cut] burdock into small pieces and [/AP#4-1 soak] in water.")
    )
