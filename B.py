def find_max_substring_occurrence(input_string):
    s_len = len(input_string)
    for i in range(s_len):
        if s_len % (i + 1) == 0:
            count_of_substr = s_len // (i + 1)
            if input_string[:i + 1] * count_of_substr == input_string:
                return count_of_substr
