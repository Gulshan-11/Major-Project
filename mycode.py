
def ML_models(given_data):
    a=''
    if (given_data[0]<35.0) or (given_data[1]<35800000.0):
        if (given_data[0]<35.0):
            a = "This model is not trained for C/N0 less than 35"
        else:
            a= "This model is not trained for Pseudorange less than 35800kMs"
    elif (given_data[0]>44.0) and (given_data[1]>36300000.0 and given_data[1]<36500000.0) and (given_data[2]> 10.0):
        a= "Line of Sight Signal"
    else:
        a= "Multipath Signal"
    return a

# df1.to_csv("SATB_L5_updated.csv",index=False)

# print(ML_models([48.00, 36480000.00,33.1234]))