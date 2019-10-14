from Airline_System import Employee, Ticket
#Create instances for Passenger class
p1 = Ticket('4A', 1, 'F', '816-234-8888', 'James Smith', 1, 'Adult', '1 October 2019', '6:00AM', '11:13AM', 'KCI', 'BNA', '1 stop', 'American Airline', 101)
p2 = Ticket('4B', 1, 'F', '816-234-8888', 'Micheal Smith', 2, 'Children', '1 October 2019', '6:00AM', '11:13AM', 'KCI', 'BNA', '1 stop', 'American Airline', 101)
p3 = Ticket('5A', 2, 'F', '816-234-8888', 'Robert Smith', 3, 'Adult', '1 October 2019', '6:00AM', '11:13AM', 'KCI', 'BNA', '1 stop', 'Delta', 102)
p4 = Ticket('5B', 3, 'F', '816-234-8888', 'Maria Garcia', 4, 'Adult', '1 October 2019', '6:00AM', '11:13AM', 'KCI', 'BNA', '1 stop', 'Southwest', 103)
p5 = Ticket('6A', 4, 'F', '816-234-8888', 'Maria Hernandez', 5, 'Adult', '1 October 2019', '6:00AM', '11:13AM', 'KCI', 'BNA', '1 stop', 'United Airline', 108)
p6 = Ticket('6B', 5, 'M', '816-234-8888', 'Maria Rodriguez', 6, 'Adult', '1 October 2019', '6:30AM', '7:33AM', 'KCI', 'LAS', 'Nonstop', 'United Airline', 105)
p7 = Ticket('7A', 6, 'M', '816-234-8888', 'Lucy Hernandez', 7, 'Adult', '1 October 2019', '6:30AM', '7:33AM', 'KCI', 'LAS', 'Nonstop', 'Spirit', 106)
p8 = Ticket('7B', 7, 'M', '816-234-8888', 'James Johnson', 8, 'Adult', '1 October 2019', '6:30AM', '7:33AM', 'KCI', 'LAS', 'Nonstop', 'American Airline', 107)
p9 = Ticket('8A', 8, 'M', '816-234-8888', 'Hannah Hernandez', 9, 'Adult', '1 October 2019', '6:30AM', '7:33AM', 'KCI', 'LAS', 'Nonstop', 'United Airline', 104)
p10 = Ticket('8B', 8, 'M', '816-234-8888', 'Noah Hernandez', 10, 'Children', '1 October 2019', '6:30AM', '7:33AM', 'KCI', 'LAS', 'Nonstop', 'United Airline', 104)
#Create instances for Employee class
e1 = Employee('201', 'James Smith', 10, 'Adult', '1 October 2019', '6:30AM', '7:33AM', 'KCI', 'LAS', 'Nonstop', 'United Airline', 104)
#Display passengers' ticket information
p1.boardingInfo()
p2.boardingInfo()
print()
#Display list of passengers on each plane
p1.getFlightInfo()
print()
#Display employee's flight information
e1.getInfo()



