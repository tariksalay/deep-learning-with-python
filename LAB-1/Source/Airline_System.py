#This class holds information of a flight: date, departure time, arrival time
#   origin, destination, number of stops, flight type, and flight number
class Flight():
    def __init__(self, date, dep_time, arv_time, origin, destination, stop_no, f_type, f_no):
        self.date = date
        self.dep_time = dep_time
        self.arv_time = arv_time
        self.origin = origin
        self.destination = destination
        self.stop_no = stop_no
        self.f_type = f_type
        self.f_no = f_no

    #Display the information
    def getInfo(self):
        print(self.date," - Departure", self.stop_no)
        print(self.origin,"\t=>\t", self.destination)
        print(self.dep_time, "\t  \t", self.arv_time)
        print(self.f_type, self.f_no)

#This class inherits Flight class. It holds information of a person: passport, passenger or employee(p_type)
#   and flight information
class Person(Flight):
    def __init__(self, name, passport, p_type, date, dep_time, arv_time, origin, destination, stop_no, f_type, f_no):
        super(Person, self).__init__(date, dep_time, arv_time, origin, destination, stop_no, f_type, f_no)
        self.name = name
        self.passport = passport
        self.p_type = p_type

#This class inherits Person class. It holds information of passengers and their flight
class Passenger(Person):
    available_flight = {}

    def __init__(self, gender, phone, name, passport, p_type, date, dep_time, arv_time, origin, destination, stop_no, f_type, f_no):
        super(Passenger, self).__init__(name, passport, p_type, date, dep_time, arv_time, origin, destination, stop_no, f_type, f_no)
        self.gender = gender
        self.phone = phone

        #Add new person to the system
        if self.f_no in Passenger.available_flight:
            Passenger.available_flight[self.f_no].append(self.name)
        else:
            Passenger.available_flight[self.f_no] = [self.name]

    #Display flight information of a passenger
    def getInfo(self):
        print('=======================================')
        print('Passenger Travel Details')
        print('Name:', self.name)
        print(self.p_type)
        print('*****************')
        super(Passenger, self).getInfo()

    #Display all flights and passengers
    def getFlightInfo(self):
        print('========================================')
        print('Welcome to ABC Airport')
        print('**********************')
        print('Flight No.\tPassenger Name')
        for i in Passenger.available_flight:
            print(i)
            for j in Passenger.available_flight[i]:
                print('\t\t', j)

#This class inherits Person class. It holds information of employees and their flight
class Employee(Person):
    def __init__(self, e_ID, name, passport, p_type, date, dep_time, arv_time, origin, destination, stop_no, f_type, f_no):
        super(Employee, self).__init__(name, passport, p_type, date, dep_time, arv_time, origin, destination, stop_no, f_type, f_no)
        self.e_ID = e_ID

    #Display flight information of employee
    def getInfo(self):
        print('===========================================')
        print('Hello, Pilot', self.name, '\tID:', self.e_ID)
        print('Here is your flight details')
        print('**********************')
        super(Employee, self).getInfo()

#This class inherits Passenger class. It holds ticket information of a passenger
class Ticket(Passenger):
    def __init__(self, seat, boarding, gender, phone, name, passport, p_type, date, dep_time, arv_time, origin, destination, stop_no, f_type, f_no):
        super(Ticket, self).__init__(gender, phone, name, passport, p_type, date, dep_time, arv_time, origin, destination, stop_no, f_type, f_no)
        self.seat = seat
        self.boarding = boarding

    #Display flight ticket information of a passenger
    def boardingInfo(self):
        print('Hello,', self.name)
        print('Here is your flight information')
        super(Passenger, self).getInfo()
        print('Seat: ', self.seat)
        print('Boarding Group: ', self.boarding)




