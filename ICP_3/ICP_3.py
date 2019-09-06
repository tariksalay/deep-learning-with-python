class Employee:
    numOfEmp = 0
    totalSalary = 0

    def __init__(self, name, family, salary, department): #constructor
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department
        Employee.numOfEmp += 1
        Employee.totalSalary += salary

    def avgSalary():
        return Employee.totalSalary/Employee.numOfEmp

class FullTimeEmployee(Employee):
    pass

emp1 = Employee("Tarik", "Salay", 100000, "Tech")
emp2 = Employee("Tari", "Sala", 500000, "Tech")
fullemp1 = FullTimeEmployee("Tarik2", "Salay2", 200000, "IT")

print(emp1.salary)
print(fullemp1.department)
print(Employee.avgSalary())

import requests
from bs4 import BeautifulSoup

html = requests.get("https://en.wikipedia.org/wiki/Deep_learning")
bsObj = BeautifulSoup(html.content, "html.parser")
print(bsObj.h1)
link = bsObj.find_all("a")
for x in bsObj.find_all("a"):
    print(x.get("href"))

import numpy as np

myVector = ((np.random.random(15)+1)*10).astype(int)
print(myVector)
myVector = myVector.reshape(3,5)
print(myVector)

myVector3 = np.max(myVector, axis=1, keepdims=True)
myVector2 = np.where(myVector == myVector3, 0, myVector)
print(myVector2)