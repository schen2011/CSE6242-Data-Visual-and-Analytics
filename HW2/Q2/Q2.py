import csv
dataset = []
district = []
year_07 = []
year_08 = []
year_09 = []
year_10 = []
year_11 = []
new_nepal = []
with open('nepal.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader: #reader will read each row when it has been called.
    	dataset.append(row)
    for index in range(0,len(dataset),5):
    	district.append(dataset[index][0])
    	year_08.append(dataset[index][2])
    	year_09.append(dataset[index+1][2])
    	year_10.append(dataset[index+2][2])
    	year_11.append(dataset[index+3][2])
    	year_07.append(dataset[index+4][2])
    	new_nepal.append([dataset[index][0],dataset[index+4][2],dataset[index][2],dataset[index+1][2],dataset[index+2][2],dataset[index+3][2]])
with open("nepal_new.csv","wb") as f:
	writer = csv.writer(f)
	writer.writerow(["District","2007","2008","2009","2010","2011"])
	writer.writerows("new_nepal.csv")