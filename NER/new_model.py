import spacy

nlp = spacy.load('./output/model-last')

# while True:
#     user_input = input('Enter your text! ---->')
#     if user_input:
#         predict = nlp(user_input)
#         for ent in predict.ents:
#             print(ent.text, '|', ent.start_char, '|', ent.end_char, '|',  ent.label_)
#     if user_input == 'exit':
#         break

def predict(text):
    predict = nlp(text)
    for ent in predict.ents:
            print(ent.text, '|', ent.label_)


if __name__ == '__main__':
    input = 'I want to go on Muztagh Ata in future'
    predict(input)