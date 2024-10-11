def find_word_in_circle(circle, word):
    c_len, w_len = len(circle), len(word)
    if w_len == 0 or c_len == 0:
        return -1
    count_circles = w_len // c_len
    for i in range(len(circle)):
        count_circles = w_len // c_len
        if circle[i] == word[0]:
            if c_len - i >= w_len:
                if circle[i:w_len + i] == word:
                    return i, 1
            else:
                start = circle[i:]
                if len(start) == c_len:
                    count_circles -= 1
                rep = count_circles * circle
                tail = circle[:w_len - c_len * (count_circles + 1) + i]
                if len(tail) == c_len:
                    count_circles -= 1
                    rep = count_circles * circle
                    tail = circle[:w_len - c_len * (count_circles + 1) + i]
                if start + rep + tail == word:
                    return i, 1
                else:
                    count_circles = w_len // c_len
            if i + 1 >= w_len:
                if circle[i:i - w_len:-1] == word:
                    return i, -1
            else:
                start = circle[i::-1]
                if len(start) == c_len:
                    count_circles -= 1
                rep = count_circles * circle[::-1]
                tail = circle[:c_len * (count_circles + 1) - w_len + i:-1]
                if len(tail) == 0 and len(start) != c_len:
                    count_circles -= 1
                    rep = count_circles * circle[::-1]
                    tail = circle[:c_len * (count_circles + 1) - w_len + i:-1]
                if start + rep + tail == word:
                    return i, -1
    return -1
