# Finetuning DistilBERT for personality classification

**Inspiration**: My goal was to expand the prediction of personality traits like Openness by applying natural language processing to real-world textual data, moving beyond traditional survey methods.

**Model**: The model uses social media text to predict Openness by using DistilBERT with a classifier head as both a feature extractor and a fine-tuned model. Utilizing a large language model (LLM) for this task is a more sophisticated approach to NLP for predicting personality than older methods (e.g., TF-IDF or keyword frequency).

**Personality**: Openness to Experience, or simply Openness, is one of the Big Five personality dimensions proposed by Costa and McCrae (1992), and is well-accepted as a measure of open-mindedness and receptivity to new ideas and experiences. (I find this trait to be one of the most fascinating as it captures nuanced attributes.)

**Application, Ethical Concerns, and Limitations**: Using social media text to measure personality is not only academically interesting, providing systematic insights into online user personalities and social dynamics, but it also has applications to personalized content, recommendation systems, and consumer behavior (although here I have many ethical concerns regarding fairness, privacy, bias, manipulation, and limited autonomy). Further, a limitation to using personality is that the Big Five is designed to capture broad patterns of behavior, not necessarily predict individual actions.




