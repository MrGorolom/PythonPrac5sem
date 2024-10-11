def check_first_sentence_is_second(s1, s2):
    bag1 = s1.split()
    bag2 = s2.split()
    for i in range(len(bag2)):
        if bag2.count(bag2[i]) > bag1.count(bag2[i]):
            return False
    return True
