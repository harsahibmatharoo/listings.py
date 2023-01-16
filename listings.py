def get_recent_listings(listings):
    # Create an empty dictionary to store listings by address
    listings_by_address = {}
    
    # Iterate through each listing
    for listing in listings:
        # extract the 3 variables from list of house listings 
        name, address, date = listing.split(',')
        # cast  to integer from str
        date = int(date)
        # Check if the address already exists in the dictionary
        if address in listings_by_address:
            # Compare the date of the existing listing to the current listing
            if date > listings_by_address[address][1]:
                # Update the dictionary with the current listing
                listings_by_address[address] = [name, date]
        else:
            # if not present, we add the listing number (name)
            # Add the current listing to the dictionary
            listings_by_address[address] = [name, date]
    # return the values of the dictionary as a list
    return [name+","+address+","+str(date) for address,(name,date) in listings_by_address.items()]

listings = ['L4, 123 kings road, 2022', 'L1, 123 kings road, 2020', 'L2, 20 queen road, 1990', 'L3, 20 queen road, 1992']

print(get_recent_listings(listings))
