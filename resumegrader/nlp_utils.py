import os
import spacy
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer

try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")

try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download("stopwords")


lemmatizer = WordNetLemmatizer()


def replace_entities_with_mask(text, entity_masks=None):
    if entity_masks is None:
        entity_masks = {}

    doc = nlp(text)
    replaced_text = []

    in_entity = {entity: False for entity in entity_masks}

    for token in doc:
        for entity, mask in entity_masks.items():
            if token.ent_type_ == entity:
                if not in_entity[entity]:
                    replaced_text.append(mask)
                    in_entity[entity] = True
                for other_entity in entity_masks:
                    if other_entity != entity:
                        in_entity[other_entity] = False
            else:
                in_entity[entity] = False

        if not any(in_entity.values()):
            replaced_text.append(token.text)

    return " ".join(replaced_text)


def preprocess(text):
    # remove punctuation
    translation_table = str.maketrans("", "", string.punctuation)
    text = text.replace("\n", "").translate(translation_table)

    # mask PERSON entities
    text = (
        replace_entities_with_mask(
            text, entity_masks={"PERSON": "[PERSON]", "ORG": "[ORG]"}
        )
        .replace("[", "")
        .replace("]", "")
    )

    # remove stop words and lemmatize non-stop words
    words = word_tokenize(text)
    text = " ".join(
        [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    )

    # pos_tags = pos_tag(filtered_words)
    return text
