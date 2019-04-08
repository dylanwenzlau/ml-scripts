from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import json
import random
import util
import os


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_NAME = 'meme_text_gen'
MODEL_PATH = util.get_model_path(BASE_PATH, MODEL_NAME)
MAX_OUTPUT_LENGTH = 140
BEAM_WIDTH = 1

model = load_model(MODEL_PATH + '/model.h5')
params = json.load(open(MODEL_PATH + '/params.json'))
SEQUENCE_LENGTH = params['sequence_length']
char_to_int = params['char_to_int']
labels = {v: k for k, v in params['labels_index'].items()}


def predict_meme_text(template_id, num_boxes, init_text = ''):
    template_id = str(template_id).zfill(12)
    min_score = 0.1

    final_texts = [{'text': init_text, 'score': 1}]
    finished_texts = []
    for char_count in range(len(init_text), MAX_OUTPUT_LENGTH):
        texts = []
        for i in range(0, len(final_texts)):
            box_index = str(final_texts[i]['text'].count('|'))
            texts.append(template_id + '  ' + box_index + '  ' + final_texts[i]['text'])
        sequences = util.texts_to_sequences(texts, char_to_int)
        data = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH)
        predictions_list = model.predict(data)
        sorted_predictions = []
        for i in range(0, len(predictions_list)):
            for j in range(0, len(predictions_list[i])):
                sorted_predictions.append({
                    'text': final_texts[i]['text'],
                    'next_char': labels[j],
                    'score': predictions_list[i][j] * final_texts[i]['score']
                })

        sorted_predictions = sorted(sorted_predictions, key=lambda p: p['score'], reverse=True)
        top_predictions = []
        top_score = sorted_predictions[0]['score']
        rand_int = random.randint(int(min_score * 1000), 1000)
        for prediction in sorted_predictions:
            # give each prediction a chance of being chosen corresponding to its score
            if prediction['score'] >= rand_int / 1000 * top_score:
            # or swap above line with this one to enable equal probabilities instead
            # if prediction['score'] >= top_score * min_score:
                top_predictions.append(prediction)
        random.shuffle(top_predictions)
        final_texts = []
        for i in range(0, min(BEAM_WIDTH, len(top_predictions)) - len(finished_texts)):
            prediction = top_predictions[i]
            final_texts.append({
                'text': prediction['text'] + prediction['next_char'],
                # normalize all scores around top_score=1 so tiny floats don't disappear due to rounding
                'score': prediction['score'] / top_score
            })
            if prediction['next_char'] == '|' and prediction['text'].count('|') == num_boxes - 1:
                finished_texts.append(final_texts[len(final_texts) - 1])
                final_texts.pop()

        if char_count >= MAX_OUTPUT_LENGTH - 1 or len(final_texts) == 0:
            final_texts = final_texts + finished_texts
            final_texts = sorted(final_texts, key=lambda p: p['score'], reverse=True)
            return final_texts[0]['text']


print('predicting meme text for ID 61533...')
print(predict_meme_text(61533, 2, ''))
