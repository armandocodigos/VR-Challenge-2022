import pickle
import numpy as np
import csv

# Load ML model
filename = 'finalized_model_tree.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Load input file
unseen = []
with open('input_relax.txt', newline='\n') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        unseen = unseen + row

# Use ML model
unseen = [float(x) for x in unseen]
unseen = np.array(unseen).reshape(1, -1)
prediction = loaded_model.predict(unseen)

# Save output file
with open('output.txt', 'w', encoding='UTF8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([prediction[0]])
    