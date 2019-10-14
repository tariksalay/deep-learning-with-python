myLst = [('John',('Physics', 80)) , ('Daniel', ('Science', 90)), ('John', ('Science', 95)), ('Mark',('Maths', 100)), ('Daniel', ('History', 75)), ('Mark', ('Social', 95))]
dct = {}

for i in myLst:
    try:
        #Append new record to the exit name key
        dct[i[0]].append(i[1])
    except KeyError:
        #Add new name and its first record to the dictionary
        dct[i[0]] = [i[1]]
for i in dct:
    #Sort the records of each name key
    dct[i].sort()
print(dct)