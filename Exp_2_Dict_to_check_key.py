my_dict={'Name':'John','age':56}

def check_dict_key():
    key=input("Enter Te Key:")
    value=input(f"Ener Value for {key}")
    if key not in my_dict:
        my_dict[key]=value
        print(f"\nThe {key} is add to the {value}")
    else:
        print(f"\nthe {key} is alredy Present on Dictionary")
check_dict_key()
while True:
    another_key = input("\nWant to Add Key? (yes/no): ").lower()
    if another_key == "yes":
        check_dict_key()
    else:
        print("Exist")
        break
print("\nUpdated Dictionary",my_dict)
