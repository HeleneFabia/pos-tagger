def split_text_into_lines(text):
    lines = list()
    lines.append(list())
    current_idx = 0
    for string in text:
        if string == "\n":
            lines.append(list())
            current_idx += 1
        else:
            lines[current_idx].append(string)
    return lines


def extract_text_and_target(lines):
    text = list()
    target = list()
    for line in lines:
        words = list()
        pos_tags = list()
        for string in line:
            word, pos, _ = string.split()
            words.append(word)
            pos_tags.append(pos)
        text.append(words)
        target.append(pos_tags)
    return text, target