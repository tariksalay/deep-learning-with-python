def longestSubstring(string):
    n = len(string)
    st = 0
    maxlen = 0
    start = 0
    pos = {}
    pos[string[0]] = 0

    for i in range(1, n):
        if string[i] not in pos:
            pos[string[i]] = i
        else:
            if pos[string[i]] >= st:
                currlen = i - st
                if maxlen < currlen:
                    maxlen = currlen
                    start = st
                st = pos[string[i]] + 1
            pos[string[i]] = i

    if maxlen < i - st:
        maxlen = i - st
        start = st

    return string[start: start + maxlen]

if __name__ == "__main__":
    string = input("Enter the string ==>")
    print(longestSubstring(string))