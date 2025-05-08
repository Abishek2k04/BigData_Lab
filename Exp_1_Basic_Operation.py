def string_operation():
    string="example"
    print("Actual String :",string)
    print("Upper case",string.upper())
    print("Lower Case",string.lower())
    print("slice",string[2])
    print("Reverse",string[::-1])
string_operation()

def list_operation():
    list1=[1,2,3,4,5]
    list2=[6,7,8,9,0]
    print("list1",list1)
    list1.extend(list2)
    print("Extend",list1)
    list1.append(35)
    print("Adding",list1)
    list2.remove(9)
    print("removing",list2)
list_operation()

def set_operation():
    set1={1,2,3,4,5,6}
    set2={7,8,9,0,11,23}
    print("Intersection",set1|set2)
    print("Union",set1&set2)
    set1.add(25)
    set2.remove(11)
    print("Adding",set1)
    print("Remove",set2)
set_operation()

def tuples_operation():
    tuples=(10,20,30,40,50)
    print("Tuples",tuples)
    print("Access Second Element :",tuples[1])
    print("Length",len(tuples))
    print("Last Element",tuples[-1])
tuples_operation()
