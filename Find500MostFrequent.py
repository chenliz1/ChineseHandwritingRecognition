# find the most frequent 500 chars among 3755 chars
# save as npy file
import numpy as np

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



