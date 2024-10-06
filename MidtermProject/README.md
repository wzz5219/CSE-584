Directories:

data -> this holds all the text completion data. It has 5 files. Each file name is like this <LLM_name>.csv. I have used 5 differnet vendors LLM model. 
So, I didnt put exact model name here, rather just used vendor name.

#textcompletion -> this folder has python scripts to generate suffixes given "llama_prefixes.csv" file. A few models (bert, LLAMA) are used offline. So, those models needs to be put here. 
               Also, other models use APIs(and necessary py packages) for inference suffixes. So, you need  APIs


Classifier:
I have made 3 classifiers. 
---SimpleDLClassifier. you do not need anything to run this. it has a simple NN model and it uses "data" to train/test the mdoel.
---SimpleDLClassifier_withTokenizer.py.To run this, you need pretrained tokenizer from Hugging Face. In my code, i used cache directory, so, during first run, the tokenizer/model will be loaded in that directory for later use.
---DLClassifier.py. This used both pretrained tokenizer and model(bert-base-uncased). 


Necessary packages: pytorch, transformers, numpy, scikit-learn, pandas, accelerator (you can use pip to install all of these)

Run:

<Classifier>.py <dataset_size> <number_of_training_pass> <test size portion> <0/1>

1st: Classifier that you need to run (any 1 of 3 classifier)
2nd: dataset_size: number of datapoints from each LLM dataset. MAX=3000.
3rd: number_of_training_pass. it can be any 1 or any numbers > 1. but the higher the pass number, the longer time , the training takes 
4th: test portion. (0 < X < 1), 0.2 means 20% of all data will be for test cases. 
5th: 0 or 1. This is an strategy. when 
        (balnced)0-> it will divide each LLM dataset in train-test ratio , and then each LLM's train is merged to final train dataset.
            and each LLM's test is merged to final test dataset. (this creates balanced) data



        (Unbalanced)1-> it will first combine all the dataset together. then it will divide whole dataset into train test based on 4th parameter.

Sample run and output:
python3 SimpleDLClassifier.py 3000 3 0.2 1
./data/mistral.csv
Number of rows in the DataFrame after skipping bad lines: 3000
./data/meta.csv
Number of rows in the DataFrame after skipping bad lines: 3000
./data/gemini.csv
Number of rows in the DataFrame after skipping bad lines: 3000
./data/distilgpt2.csv
Number of rows in the DataFrame after skipping bad lines: 3000
./data/openai.csv
Number of rows in the DataFrame after skipping bad lines: 3000
Training set LLM distribution:
label
0    2425
2    2403
1    2400
4    2399
3    2373
Name: count, dtype: int64
Testing set LLM distribution:
label
3    627
4    601
1    600
2    597
0    575
Name: count, dtype: int64
number of LLMs 5
Running on GPU
Pass 1/3, Train Loss: 1.6177, Accuracy: 26.67%
Pass 2/3, Train Loss: 1.5400, Accuracy: 30.28%
Pass 3/3, Train Loss: 1.5128, Accuracy: 31.27%
Test Accuracy: 31.4667
Precision (weighted): 0.30
Recall (weighted): 0.31
F1 Score (weighted): 0.27
LLM 0 Accuracy: 18.96%
LLM-name: meta Accuracy: 18.96%
LLM 1 Accuracy: 73.83%
LLM-name: gemini Accuracy: 73.83%
LLM 2 Accuracy: 19.10%
LLM-name: openai Accuracy: 19.10%
LLM 3 Accuracy: 4.63%
LLM-name: mistral Accuracy: 4.63%
LLM 4 Accuracy: 41.43%
LLM-name: distilgpt2 Accuracy: 41.43%
Elapsed time in testing : 0.293095 seconds
(/scratch/wahid/myenv) python3 SimpleDLClassifier.py 3000 3 0.2 0
./data/mistral.csv
Number of rows in the DataFrame after skipping bad lines: 3000
./data/meta.csv
Number of rows in the DataFrame after skipping bad lines: 3000
./data/gemini.csv
Number of rows in the DataFrame after skipping bad lines: 3000
./data/distilgpt2.csv
Number of rows in the DataFrame after skipping bad lines: 3000
./data/openai.csv
Number of rows in the DataFrame after skipping bad lines: 3000
Training set LLM distribution:
label
3    2400
0    2400
1    2400
4    2400
2    2400
Name: count, dtype: int64
Testing set LLM distribution:
label
3    600
0    600
1    600
4    600
2    600
Name: count, dtype: int64
number of LLMs 5
Running on GPU
Pass 1/3, Train Loss: 1.6178, Accuracy: 21.24%
Pass 2/3, Train Loss: 1.5401, Accuracy: 29.47%
Pass 3/3, Train Loss: 1.5111, Accuracy: 31.83%
Test Accuracy: 31.6333
Precision (weighted): 0.31
Recall (weighted): 0.32
F1 Score (weighted): 0.28
LLM 0 Accuracy: 20.67%
LLM-name: meta Accuracy: 20.67%
LLM 1 Accuracy: 71.50%
LLM-name: gemini Accuracy: 71.50%
LLM 2 Accuracy: 28.50%
LLM-name: openai Accuracy: 28.50%
LLM 3 Accuracy: 5.67%
LLM-name: mistral Accuracy: 5.67%
LLM 4 Accuracy: 31.83%
LLM-name: distilgpt2 Accuracy: 31.83%
Elapsed time in testing : 1.147132 seconds
