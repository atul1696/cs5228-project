import csv

def create_submission(testY, filename):
    f = open(filename, 'w')
    writer = csv.writer(f)
    writer.writerow(['Id', 'Predicted'])

    for i, ele in enumerate(testY):
        writer.writerow([i, ele])

    f.close()
    return
