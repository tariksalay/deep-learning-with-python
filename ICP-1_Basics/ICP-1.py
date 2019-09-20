mystring = list(input(""))
print(mystring)
for i in range(3):
    mystring.pop()
print(mystring)
print(''.join(mystring[::-1]))

number1 = int(input("Enter number1"))
number2 = int(input("Enter number2"))
print(number1 * number2)

replaceinput = input("")
print(replaceinput.replace("python", "pythons"))