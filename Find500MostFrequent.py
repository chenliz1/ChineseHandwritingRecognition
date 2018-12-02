# find the most frequent 500 chars among 3755 chars
# save as npy file
import numpy as np
import pickle as pkl

if __name__ == "__main__":
    result = np.zeros(3755)

    file_500 = open("500MostFrequentChineseChar", encoding="utf8")
    file_dict = open("char_dict.txt", encoding="utf8")

    data_500 = file_500.read()
    data_dict = file_dict.read()

    chars = data_dict.replace("'","").split(", ")

    for pair in chars:
        # [-1] ->  u'\ufeff'
        char = pair.split(": ")[0][-1]
        idx = pair.split(": ")[1]

        # only 496 out of 500 most frequent is found in HWDB1.1
        if char in data_500:
            result[int(idx)] = 1

    np.save("data/MostFreq500inHWDB1.1", result)

    with open('./data/char_dict', 'rb') as handler:
        charDict = pkl.load(handler)

    index_key_500 = {}
    freq500Words = result

    # save a dictionary for mapping class number to chinese characters for GUI
    for key, value in charDict.items():
        if freq500Words[value] == 1:
            newLabel = np.int(freq500Words[:value+1].sum() - 1)
            index_key_500[newLabel] = key
    with open('./data/index_key_500.pickle', 'wb') as handler:
            pkl.dump(index_key_500, handler, protocol=pkl.HIGHEST_PROTOCOL)

