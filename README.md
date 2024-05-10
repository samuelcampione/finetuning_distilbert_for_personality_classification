# Fine-Tuning DistilBERT for Personality Classification of Social Media Text
My goal was to expand the prediction of personality traits like Openness by applying natural language processing to real-world text data, moving beyond traditional survey methods. 

### **Model**: 
The model uses Facebook Status Updates to predict Openness by using DistilBERT with a classifier head as both a feature extractor and a fine-tuned model. Utilizing a language model for this task, specifically a light transformer model like DistilBERT, is a more sophisticated approach to NLP for predicting personality than older methods (e.g., TF-IDF or keyword frequency).

### **Performance**


| Model   | Accuracy | Description |
|---------|----------|----------|
| `BERTClassifierUnfrozen` |74.64%| Fine-Tuned DistilBERT with classifier final layer |
| `BERTClassifier` | 74.54% | Static DistilBERT with classifier final layer |

### **Usage**
- [data.py](https://github.com/samuelcampione/finetuning_distilbert_for_personality_classification/blob/main/data.py):  text preprocessing functions and custom PyTorch Dataset class
- [models.py](https://github.com/samuelcampione/finetuning_distilbert_for_personality_classification/blob/main/models.py): PyTorch models trained,  `BERTClassifier` and `BERTClassifierUnfrozen`
- [train_eval.py](https://github.com/samuelcampione/finetuning_distilbert_for_personality_classification/blob/main/train_eval.py): training and evalution functions
- [predicting_personality.ipynb](https://github.com/samuelcampione/finetuning_distilbert_for_personality_classification/blob/main/predicting_personality.ipynb): final notebook that explores data and trains and evaluates both models
- [requirements.txt](https://github.com/samuelcampione/finetuning_distilbert_for_personality_classification/blob/main/requirements.txt)

### **Personality**: 
Openness to Experience, or simply Openness, is one of the Big Five personality dimensions proposed by Costa and McCrae (1992), and is well-accepted as a measure of open-mindedness and receptivity to new ideas and experiences. (I find this trait to be one of the most fascinating as it captures nuanced attributes.)

### **Application, Ethical Concerns, and Limitations**: 
Using social media text to measure personality is not only academically interesting, providing systematic insights into online user personalities and social dynamics, but it also has applications to personalized content, recommendation systems, and consumer behavior (although here I have many ethical concerns regarding fairness, privacy, bias, manipulation, and limited autonomy). Further, a limitation to using personality is that the Big Five is designed to capture broad patterns of behavior, not necessarily predict individual actions.
