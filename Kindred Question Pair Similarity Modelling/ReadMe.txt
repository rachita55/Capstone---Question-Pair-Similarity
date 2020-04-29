Procedure to find similar questions using Infersent:

Technique Used:

1.Word Embedding

2.Cosine Similarity.

Procedure:
Step 1: Mount the drive and define the path to store it in google drive.

Step 2: Load the dataset in google colab.

Step 3: Encode the dataset using infersent.It will convert text to vector.

Step 4: Use cosine similarity on the vectors.It will give the similarity between each sentences.

Step 5: Type a question.Encode the question.

Step 6: Find the input index of the question.

Step 7: Use argmax to compare the index of the question and dataset.It will give the most similar question the question asked. 

Step 8: Use argsort to compare the index of the question and dataset.Top similar questions will be displayed.

Step 9: Append the question and the similar questions.

Step 10: Encode the summary of above questions.

Step 11: Repeat the procedure 4,7,8 to get the similar question based on both question and summary.

 

