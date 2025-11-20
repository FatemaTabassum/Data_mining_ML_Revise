<!-----

You have some errors, warnings, or alerts. If you are using reckless mode, turn it off to see useful information and inline alerts.
* ERRORs: 0
* WARNINGs: 0
* ALERTS: 1

Conversion time: 1.044 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs™ to Markdown version 2.0β1
* Thu Nov 20 2025 10:36:29 GMT-0800 (PST)
* Source doc: data_mining_naive_bayes-Assignment 
* This document has images: check for >>>>>  gd2md-html alert:  inline image link in generated source and store images to your server. NOTE: Images in exported zip file from Google Docs may not appear in  the same order as they do in your doc. Please check the images!

----->


<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 0; ALERTS: 1.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p><a href="#gdcalert1">alert1</a>

<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>



## Advanced Data Mining


## COP3330



1. Naive Bayes Classifier

The Naive Bayes classifier is a simple probabilistic supervised Machine Learning Algorithm based on Bayes’ theorem. Bayes’ Theorem is stated as:



<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image1.png "image_tooltip")


Here, 

A, B are events.

P(A | B) = Probability of A given B is true. 

P(B | A) = Probability of B given A is true. 

P(A), P(B) = The independent probability of A and B

Bayes theorem gives a principled way of calculating a conditional probability without the joint probability. However it assumes that each input is dependent upon all other variables, which make it difficult to compute in many complex cases. Hence, to use Bayes Theorem as a classifier, it is assumed that each variable is independent of each other. It is referred to as the Naive Bayes Classifier. Naive Bayes is named naive due to the calculations being simplified and the effectiveness of its application in real-world is many. 

Some terms are used for Bayes theorem when using Naive Bayes Classifier-

P(A | B): Posterior Probability

P(A): Prior Probability

P(B | A): Likelihood

P(B): Evidence

Hence, we can restate Bayes Theorem as

Posterior = Likelihood * Prior / Evidence

To model this as a classification problem we express Bayes Theorem as a classification model like below-

P(Class | Data) = (P(data | class) * P(Class)) / P(data)

To simplify the calculation we remove the assumption of dependence and consider each variable is independent. Additionally, we drop the denominator as it is a constant for all calculations.

P(Class | X1, X2, … Xn) = P(X1 | Class) * P (X2 | Class) * … * P(Xn | Class) * P(Class)

In this assignment we consider only discrete attributes, and 2-class labels. 

We can compute prior probabilities and probabilities of discrete attributes as follows-


#### Prior Probability

P(C) = Nc / N ; 

Where, N = Total number of instances in the dataset

Nc = Number of instances of dataset labeled class C


#### Probabilities of discrete attributes

P(Ai, Ck) = |Aik| / Nc

Where,  |Aik| is the number of instances having attribute Ai and belong to class Ck.



2. Implementation Details

I have used Python’ list data structure for saving the datasets. I have also used python’s dictionary data structure to save the count for computing likelihood. While implementing I used the following methods and variables to read raw data, process and save the dataset.

Variables


```
dataset_train = holds training dataset
labels_train = holds training dataset label

dataset_test = holds testing dataset
labels_test = holds test labels

labels_test is used for evaluating the classifier.
```


Methods


```
def create_dataset(self, raw_data_set, max_id = 0)
    This method accepts raw_data_set as a parameter. After processing it returns 2 separate lists, one for class labels, one for input. max_id parameter is used to identify the number of maximum number of attributes. I assumed there would not be more than 20 attributes in the dataset. Class referred as "+1" is considered class_1 and the assigned label is 0. Class referred as "-1" is considered class_2 and the assigned label is 1. This is done for simplicity.

def compute_prior_prob(self, data_size):
	This method is used to compute the prior probability of each class. 
    The probability is saved in self.prior_prob_dict dictionary data- structure. 

def classifier_train(self,):
    This method prepares the classifier from dataset_train and labels_train dataset, and does necessary computation. A dictionary keep_count_dict is used to save all the count. The key of this dictionary holds 3 variables-
Key1 = class label,
Key2 = attribute,
Key3 = attribute_subtype
So,
    keep_count_dict[(key1, key2, key3)] holds the count for computing probabilities.
    For example, if class label = 0, attribute = 1, attribute_subtype = 3, then when the first occurrence of this data instance happens, the dictionary saves it as keep_count_dict[(0, 1, 3)] = 1. The second time it encounters key(0, 1, 3), the count is incremented and the value becomes keep_count_dict[(0, 1, 3)] = 2. Likewise, for all such keys present in the dataset, the counts are computed.

def compute_likelyhood(self, dataset_):
    In this method, dataset_train, and dataset_test are sent as parameters (one after one for) computing likelihood and posterior probability. If any instance is not present in dataset_test, then I assigned a 0 to the corresponding keep_count_dict[key].

    The probability of each attribute of a particular instance is computed by,

    prob = math.log2(int(self.keep_count_dict[key]) / self.class_1_cnt)

    The likelihood of that instance is computed by multiplying all the attributes' probabilities and saving it into class_1_likelihood_sum variable.
    So,
    class_1_likelihood_sum = class_1_likelihood_sum * self.prior_prob_dict[self.class_1_identifier]

    Likewise, class_2_likelihood_sum is calculated.

    if class_1_likelihood_sum >= class_2_likelihood_sum
    Then class_1 is assigned as a label to this particular instance.

def compute_confusion_matrix(self, original_label, computed_labels):
	This method is used to determine confusion matrix.

```



3. Results and Calculations

#### 
    Breast_cancer


    Breast_cancer.train has total 180 instances 


    Breast_cancer.test has total 106 instances 


    ‘+1’ (class=0) has instances 56 in the training set


    ‘-1’ (class=1) has instances 124 in the training set


    Prior_prob(class=0) =  56 / 180 


                   = 0.3111111


	Prior_prob(class=1) = 124 / 180

			        = 6.888889

	Without log2, I get the values of confusion matrix as following

	Train:

	a = 30, b = 26, c = 19, d = 105

	Test:

	a = 16, b = 13, c = 14, d = 63

	With log2, I get the values of confusion matrix as following

	Train:

	a = 26, b = 30, c = 47, d = 77

	Test:

	a = 17, b = 12, c = 21, d = 56

	


#### 	Poker


    poker.train has total 1042 instances


    poker.test has total 678 instances


    ‘+1’ (class=0) has instances 747 in the training set


    ‘-1’ (class=1) has instances 295 in the training set


    Prior_prob(class=0) =  747 / 1042 


                   = 0.71689

	Prior_prob(class=1) = 295 / 1042

			        = 0.2831

	Without log2 I get the values of confusion matrix as following

	Train:

	a = 742, b = 5, c = 285, d = 10

	Test:

	a = 448, b = 11, c = 217, d = 2

	With log2, I get the values of confusion matrix as following

	Train:

	a = 747, b = 0, c = 294, d = 1

	Test:

	a = 459, b = 0, c = 219, d = 0

Here, 

a = TP (True Positive)

b = FN (False Negative)

c = FP (False Positive)

d = TN (True Negative)

Train Accuracy = (a + d) / (a + b + c + d)

	    = (747 + 1) / (747 + 0 + 219 + 1)

                = 0.72

	     = 72%

Test Accuracy = (a + d) / (a + b + c + d)

	    = (459 + 0) / (459 + 0 + 219 + 0)

                = 0.68

	     = 68%

This is the train accuracy and test accuracy of Poker data with log2 used.


    We can see that here, train accuracy > test accuracy and usually train accuracy is bigger than test accuracy.


    Likewise we can compute accuracy for Breast_cancer data as following -


    Train Accuracy = (a + d) / (a + b + c + d)

	    = (26 + 77) / (26 + 30 + 47 + 77)

                = 0.5722

	     = 57.2%

Test Accuracy = (a + d) / (a + b + c + d)

	    = (17 + 56) / (17 + 12 + 21 + 56)

                = 0.689

	    = 68.9%



4. Potential Improvements

    Some attribute values are zero. I observed that sometimes, a value in the test dataset occurs which is not present in the training dataset. Since the attribute value is not present, the attribute count value is 0. If one attribute value is 0, then the whole likelihood will be zero no matter what the other attributes values are, or prior probability is. To resolve this matter, we can use some smoothing techniques - like Laplacian smoothing, m-estimate smoothing although I did not use any smoothing technique. We can segment the data and use Gaussian naive bayes when encountering numerical values. Instead of computing only probabilities, computing log probabilities also help to improve the performance. We can also remove redundant features from the dataset to get a better performance.


	

            

	
