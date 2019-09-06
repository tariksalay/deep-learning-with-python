lb_list = input("Enter the weight list in lbs").split(",")
print(type(lb_list))
kg_list = [(int(lb_list[i]) * 0.45) for i in range(len(lb_list))]
print(kg_list)


def string_alternative():
    oldString = input("Enter your string")
    newString = ""
    for i in range (len(oldString)):
        if (i%2==0):
            newString += oldString[i]
    print(newString)

string_alternative()


myFile = open("word.txt", "r")
myDict = {}
line = myFile.readline()
while line != "":
    mylist = line.split()
    for i in mylist:
        if i not in myDict:
            myDict[i] = 1
        else:
            myDict[i] += 1
    line = myFile.readline()

myFile.close()


with open('word.txt', 'a') as f:
    f.write("\n")
    print(myDict, file=f)