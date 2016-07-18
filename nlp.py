import re
import string
import time
import json
import distance
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cross_validation as cv
from pandas import DataFrame
from sknn import mlp
from sklearn.preprocessing import MultiLabelBinarizer
from collections import OrderedDict
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC
from sknn.backend import lasagne
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

with open('data.json') as data_file:    
    train_data = json.load(data_file)
# print(train_data[0])

all_scripts = []
all_questags = []

# for element in train_data:
for element in train_data:
    all_scripts.append(element['script'])
    all_questags.extend(element['questags'])

all_scripts = np.array(all_scripts)
all_questags = np.array(all_questags)

unique_scripts, scripts_counts = np.unique(all_scripts, return_counts=True)
unique_questags, questags_counts = np.unique(all_questags, return_counts=True)

# Sort the scripts by frequency.
scripts_counts, unique_scripts = zip(*sorted(zip(scripts_counts, unique_scripts)))


# # Plot the frequencies.
# pos = np.arange(len(unique_questags)) + 0.5
# plt.figure(figsize=(20, 10))
# plt.barh(pos, questags_counts, align='center')
# plt.yticks(pos, unique_questags)
# plt.xlabel('Frequency')
# plt.ylabel('scripts')
# plt.title('scripts Frequency Distribution')
# plt.show()

punctuation_to_replace_with_space = re.compile(r"[-,]")
symbols_to_remove = re.compile(r"[!\\/%.']")
digits_to_remove = re.compile(r"\d+")
multiple_spaces_to_trim = re.compile(r" +")


def clean_questags(questags):
    """
    Removes unnecessary noise from the questions/tags text.
    """
    
    # Convert to lowercase.
    questags = questags.lower()
    
    # Replace hyphens and commas with spaces.
    questags = punctuation_to_replace_with_space.sub(" ", questags)
      
    # Remove various unwanted symbols.
    questags = symbols_to_remove.sub("", questags)
    
    # Remove digits.
    questags = digits_to_remove.sub("", questags)
            
    # Remove excess spacing in between words after first cleaning pass.
    questags = multiple_spaces_to_trim.sub(" ", questags)
        
    # Remove excess spacing in between words after second cleaning pass and leading/trailing whitespace.
    questags = multiple_spaces_to_trim.sub(" ", questags)
    questags = questags.strip()
    
    return questags

def clean_element(element):
    """
    Cleans all the questags in a single element.
    """

    # Map the clean function to every questags in the element.
    element['questags'] = map(clean_questags, element['questags'])
    
    # Make sure post-cleaning empty strings are removed.
    element['questags'] = [questag for questag in element['questags'] if len(questag) > 2]
    
    return element


def clean_raw_data(raw_data):
    """
    Cleans all the elements in the dataset by mapping each element to a clean function.
    """
    
    # Map a element-cleaning function to every element in the data.
    clean_data = np.array(map(clean_element, raw_data))
    
    return clean_data

# Execute, time, and cache the cleaning procedure.
start_time = time.time()
train_data = clean_raw_data(train_data)
print("\nTime spent cleaning: {0:.3f} min.".format((time.time() - start_time) / float(60)))

clean_questags = []

for element in train_data:
    clean_questags.extend(element['questags'])

print("Noisy questags consolidated: {}".format(len(unique_questags) - len(np.unique(clean_questags))))

# Get unique questags and their frequencies.
unique_clean_questags, clean_questags_counts = np.unique(clean_questags, return_counts=True)

# Sort the questags by frequency.
clean_questags_counts, unique_clean_questags = zip(*sorted(zip(clean_questags_counts, unique_clean_questags)))

# Only plot the most popular questags.
most_popular_unique_clean_questags = unique_clean_questags[-40:]
most_popular_clean_questags_counts = clean_questags_counts[-40:]

# Plot the frequencies.
# pos = np.arange(len(most_popular_unique_clean_questags)) + 0.5
# plt.figure(figsize=(40, 20))
# plt.barh(pos, most_popular_clean_questags_counts, align='center', color='red')
# plt.yticks(pos, most_popular_unique_clean_questags)
# plt.xlabel('Frequency')
# plt.ylabel('Questags')
# plt.title('Most Popular questags')
# plt.show()



def build_vectorized_element_matrix(data):
    """
    Takes the element and produces an UNCONDENSED matrix where each row represents a scripts
    and each column represents the presence of an questags.
    """
    
    stopchar = '|'
    
    element_matrix_in_strings = [stopchar.join(element['questags']) for element in data]

    vectorizer = CountVectorizer(binary=True, vocabulary=unique_clean_questags, token_pattern='[^|]+')
    
    element_matrix = vectorizer.fit_transform(element_matrix_in_strings).toarray()
    
    return element_matrix



vectorized_matrix = build_vectorized_element_matrix(train_data)


print("Number of elements (represented as rows): {}".format(len(train_data)))
print("Number of unique cleaned questags (represented as columns): {}".format(len(unique_clean_questags)))
print("Dimensions of binary element matrix: {}".format(vectorized_matrix.shape))
print("Matrix representation: \n{}".format(vectorized_matrix))

print("First element in original data:")
for questags in sorted(train_data[0]['questags']):
    print("\t" + questags)

print("First record in binary matrix:")
(rows, cols) = vectorized_matrix.shape
first_record = sorted([unique_clean_questags[i] for i in range(cols) if vectorized_matrix[0, i] == 1])
for questags in first_record:
    print("\t" + questags)

def build_csr_sparse_element_matrix(data):
    """
    Takes the elements and produces a sparse matrix in Condensed Row format 
    where each row represents a element and each column represents an questags.
    
    The idea behind doing this is to improve speed when building and using models.
    """
    
    # Defines an questags-index dictionary to help to build our matrix.
    questags_dict = dict(zip(unique_clean_questags, range(len(unique_clean_questags))))
    
    # Temporarily store the nonzero elements of the sparse matrix.
    row_index_vector = []
    col_index_vector = []
    value_vector = []
    
    # Iterate through the data and construct the necessary index and value vectors.
    for row in range(len(data)):
        for i in range(len(data[row]['questags'])):
            col = questags_dict[data[row]['questags'][i]]
            row_index_vector.append(row)
            col_index_vector.append(col)
            value_vector.append(True)
            
    # Build the sparse matrix using the three vectors populated above.
    sparse_matrix = csr_matrix((value_vector, (row_index_vector, col_index_vector)), dtype=np.bool)
    
    return sparse_matrix

# Transform it.
csr_sparse_matrix = build_csr_sparse_element_matrix(train_data)

# Look at what happened.
print("Number of elements (represented as rows): {}".format(len(train_data)))
print("Number of unique cleaned questags (represented as columns): {}".format(len(unique_clean_questags)))
print("Dimensions of sparse element matrix: {}".format(csr_sparse_matrix.shape))
print("Matrix representation (notice it only keeps track of True instances): \n{}".format(csr_sparse_matrix))

print("First element in original data:")
for questags in sorted(train_data[0]['questags']):
    print("\t" + questags)
    
print("First record in sparse matrix:")
first_row_dense_matrix = np.asarray(csr_sparse_matrix.getrow(0).todense())
for element in first_row_dense_matrix:
    indices = np.where(element)[0]
    first_record = sorted([unique_clean_questags[i] for i in indices])
    for questags in first_record:
        print("\t" + questags)

# Define the questags classifier labels with a more sensible variable name than 'all_questags'.
questags_labels = all_questags

# Metrics for predictors.
n_features = csr_sparse_matrix.shape[1]
n_targets = len(set(all_questags))


def kfold_fit(data_matrix, labels, classifier, n_folds=3):
    """
    Performs manual cross-validation given a matrix of data, labels, 
    and a predictive model.
    """
    
    # Define a K-fold for the given matrix dimensions.
    (num_rows, num_cols) = np.shape(data_matrix)
    kf = cv.KFold(num_rows, n_folds=n_folds)

    # Iterator of folds.
    for train, test in kf:

        # Define the training and test data and labels for this fold.
        data_train = data_matrix[train]
        data_test = data_matrix[test]
        labels_train = labels[train]
        labels_test = labels[test]

    # Fit the classifier.
    classifier.fit(data_train, labels_train)
        
    return classifier


def evaluate(classifier, sample_size=10000):
    """
    Returns the accuracy of a given model.
    
    Default sample is the entire dataset.
    """
    
    # Take a random sample so the evaluation doesn't take forever.
    sample_indices = np.random.choice(len(all_scripts), sample_size)
    
    # Correct labels.
    y_true = all_scripts[sample_indices]
    
    # Predicted labels.
    y_pred = classifier.predict(csr_sparse_matrix[sample_indices])
    
    # Proportion predicted correctly.
    accuracy = metrics.accuracy_score(y_true, y_pred)
    
    return accuracy


start_time = time.time()
knn_classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski'))
knn = kfold_fit(csr_sparse_matrix, questags_labels, knn_classifier)
print("\nTime spent fitting model: {0:.3f} min.".format((time.time() - start_time) / float(60)))
print(knn)


# Execute, time, and cache the training evaluating process.
start_time = time.time()
knn_accuracy = evaluate(knn, sample_size=200)
print("\nTime spent making predictions: {0:.3f} min.".format((time.time() - start_time) / float(60)))
print(knn_accuracy)

start_time = time.time()
dt_classifier = OneVsRestClassifier(DecisionTreeClassifier())
dt = kfold_fit(csr_sparse_matrix, questags_labels, dt_classifier)
print("\nTime spent fitting model: {0:.3f} min.".format((time.time() - start_time) / float(60)))
print(dt)

# Execute, time, and cache the training evaluating process.
start_time = time.time()
dt_accuracy = evaluate(dt)
print("\nTime spent making predictions: {0:.3f} min.".format((time.time() - start_time) / float(60)))
print(dt_accuracy)

# Execute, time, and cache the random forest fitting procedure.
start_time = time.time()
rf_classifier = RandomForestClassifier()
rf = kfold_fit(csr_sparse_matrix, questags_labels, rf_classifier)
print("\nTime spent fitting model: {0:.3f} min.".format((time.time() - start_time) / float(60)))
print(rf)

# Execute, time, and cache the training evaluating process.
start_time = time.time()
rf_accuracy = evaluate(rf)
print("\nTime spent making predictions: {0:.3f} min.".format((time.time() - start_time) / float(60)))
print(rf_accuracy)

start_time = time.time()
svm_classifier = LinearSVC(multi_class='ovr')
svm = kfold_fit(csr_sparse_matrix, questags_labels, svm_classifier)
print("\nTime spent fitting model: {0:.3f} min.".format((time.time() - start_time) / float(60)))
print(svm)

# Execute, time, and cache the training evaluating process.
start_time = time.time()
svm_accuracy = evaluate(svm)
print("\nTime spent making predictions: {0:.3f} min.".format((time.time() - start_time) / float(60)))
print(svm_accuracy)

# Execute, time, and cache the logistic regression fitting procedure.
start_time = time.time()
lr_classifier = LogisticRegression()
lr = kfold_fit(csr_sparse_matrix, questags_labels, lr_classifier)
print("\nTime spent fitting model: {0:.3f} min.".format((time.time() - start_time) / float(60)))
print(lr)

# Execute, time, and cache the training evaluating process.
start_time = time.time()
lr_accuracy = evaluate(lr)
print("\nTime spent making predictions: {0:.3f} min.".format((time.time() - start_time) / float(60)))
print(lr_accuracy)

final_classifier = rf

#######################

##Dealing with Test Data
with open('tesdata.json') as data_file:    
    test_data = json.load(data_file)

# All the questags in the test data.
all_test_questags = []
for element in test_data:
    all_test_questags.extend(element['questags'])
    
# All the unique questags in the test data.
unique_test_questags = set(all_test_questags)

# print("How many are there?")
print(len(unique_test_questags))

known_questags = sorted(unique_clean_questags, key=len)

def find_closest_known_questag(questag):
    """
    Returns the "closest" known questag to the existing questag.
    """
    
    # If the unknown questag is a substring of a known questag,
    # return the shortest known questag with that property.
    for known_questag in known_questags:
        if questag in known_questag:
            return known_questag
        
    # If a known questag is the substring of the unknown questag,
    # return the longest known questag with that property.
    for known_questag in reversed(known_questags):
        if known_questag in questag:
            return known_questag
        
    # Now find the known questag with the maximum Jaccard distance.
    max_jaccard_dist = 0
    closest_known_questag = known_questags[0]
    for known_questag in known_questags:
        dist = distance.jaccard(questag, known_questag)
        if dist > max_jaccard_dist:
            closest_known_questag = known_questag
            max_jaccard_dist = dist
            
    return closest_known_questag


def match_new_questags_and_clean_element(element):
    """
    Cleans all the questags and replaces novel questags in test instances 
    with the "closest" known questags.
    """

    # Make sure unknown questags are mapped to existing known questags from our training data.
    element['questags'] = [find_closest_known_questag(questag) if questag not in known_questags else questag for questag in element['questags']]

    return element


def clean_test_data(test_data):
    """
    Performs data cleaning with the additional step of pairing new questags
    with existing ones for test instances.
    """
    
    # Map a element-cleaning function to every element in the data.
    clean_data = np.array(map(match_new_questags_and_clean_element, test_data))
    
    return clean_data

start_time = time.time()
test_data = clean_test_data(test_data)
print("\nTime spent cleaning: {0:.3f} min.".format((time.time() - start_time) / float(60)))

all_clean_test_questags = []
for element in test_data:
    all_clean_test_questags.extend(element['questags'])
    
# All the unique clean questags in the test data.
unique_clean_test_questags = set(all_clean_test_questags)

# How many are there?
print(len(unique_clean_test_questags))

test_data_matrix = build_csr_sparse_element_matrix(test_data)
print("initial")
print(test_data_matrix[0])
print(test_data_matrix.shape[0])


# Execute, time, and cache the predicting procedure.
start_time = time.time()
prediction_ids = [element['id'] for element in test_data]
# final_classifier.fit(csr_sparse_matrix  ,test_data_matrix)
predicted_scripts = final_classifier.predict(test_data_matrix)
print("\nTime spent making predictions: {0:.3f} min.".format((time.time() - start_time) / float(60)))

#predicted_scripts = [prediction[0] for prediction in predicted_scripts]
#d = DataFrame(data=OrderedDict([('id', prediction_ids), ('script', predicted_scripts)]))
#d.to_csv('final_predictions.csv', index=False)

# Open the prediction file with read privileges.
#final_predictions = open('final_predictions.csv', 'r')

# Read through and output the text.
#for line in final_predictions:
#    print(line)
# Close the file.
#final_predictions.close()

print("Finish.")
