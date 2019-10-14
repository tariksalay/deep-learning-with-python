subString = ''
lst = []
#Get input string
string = input('Enter a string ==> ')
#Iritate over the input string
for i in string:
    #Add characters to substring if not repeated
    if (i not in subString):
        subString += i
    #Add the substring to the list and create new substring when repeated
    else:
        lst.append(subString)
        subString = i
#Append the last substring
lst.append(subString)
print('The longest substring without repeating characters:', max(lst))
