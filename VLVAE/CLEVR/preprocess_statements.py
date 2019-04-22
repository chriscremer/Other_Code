


import numpy as np


import json


import torch



n_train = 90000
n_val = 10000
n_test = 0


# n_train = 900#00
# n_val = 100#00
# n_test = 0




def invert_dict(d):
    return {v: k for k, v in d.items()}



def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    assert vocab['question_token_to_idx']['<NULL>'] == 0
    assert vocab['question_token_to_idx']['<START>'] == 1
    assert vocab['question_token_to_idx']['<END>'] == 2
    assert vocab['program_token_to_idx']['<NULL>'] == 0
    assert vocab['program_token_to_idx']['<START>'] == 1
    assert vocab['program_token_to_idx']['<END>'] == 2
    return vocab



# from attributes
def get_sentence(list_of_word_idxs, question_idx_to_token): #, answer):
    sentence =''
    list_of_word_idxs = list_of_word_idxs.cpu().numpy()#[0]
    for i in range(len(list_of_word_idxs)):
        word = question_idx_to_token[int(list_of_word_idxs[i])]
        sentence += ' ' + word
    return sentence


#returns list instead of string
def get_sentence2(list_of_word_idxs): #, answer):
    sentence = []
    list_of_word_idxs = list_of_word_idxs.cpu().numpy()#[0]
    for i in range(len(list_of_word_idxs)):
        word = question_idx_to_token[int(list_of_word_idxs[i])]
        sentence.append(word) 
    return sentence







    # # count attributes
    # counts = [0]*vocab_size
    # dataset = test_question_dataset
    # for i in range(len(dataset)):
    #     for j in range(len(dataset[i])):
    #         counts[int(dataset[i][j])] +=1

    # for i in range(len(new_vocab)):
    #     print (new_vocab[i], counts[i])
    # fsadf















def preprocess_v1(loader, vocab_file):

    # val_loader = ClevrDataLoader(**val_loader_kwargs)
    # question_idx_to_token = train_loader.dataset.vocab['question_idx_to_token']

    print('loader has %d samples' % len(loader.dataset))

    image_dataset = loader.feature_h5['features']
    question_dataset = loader.dataset.all_questions
    # answer_dataset = train_loader.dataset.all_answers

    # print(question_dataset.type())
    # fdsaf


    vocab = load_vocab(vocab_file)
    question_idx_to_token = vocab['question_idx_to_token']

    # print (question_idx_to_token)

    # # Remove useless dimensions
    # question_dataset = question_dataset[:,1:7]

    # print (get_sentence(question_dataset[0]))
    # fsdasd
    # print ('vocab size', len(vocab['question_token_to_idx']))

    new_vocab = ['cube', 'sphere', 'cylinder', 
                'cyan', 'red', 'gray', 'blue', 'yellow', 'purple', 'brown', 'green', 
                'large', 'small',
                'matte', 'metallic']
    new_question_idx_to_token = {}
    new_question_token_to_idx = {}
    for i in range(len(new_vocab)):
        new_question_idx_to_token[i] = new_vocab[i]
        new_question_token_to_idx[new_vocab[i]] = i

    vocab_size = len(new_vocab)
    # vocab_size = len(question_idx_to_token)
    print ('vocab size:', vocab_size)

    # Remove useless dimensions
    question_dataset = question_dataset[:,3:7]

    # Convert to new vocab / remove synonyms
    new_question_dataset = torch.zeros(question_dataset.shape[0], question_dataset.shape[1])
    # Remove synonyms
    # Block/cube -> cube
    # Big/large -> large
    # Small/tiny -> small
    # Rubber/matte -> matte
    # Shiny/metallic/metal ->  metallic
    # Ball/sphere -> sphere

    for i in range(len(question_dataset)):
        for j in range(len(question_dataset[i])):
            #word
            word = question_idx_to_token[int(question_dataset[i][j])]
            if word == 'block':
                word = 'cube'
            elif word == 'big':
                word = 'large'
            elif word == 'tiny':
                word = 'small'
            elif word == 'rubber':
                word = 'matte'
            elif word == 'shiny' or word == 'metal':
                word = 'metallic'
            elif word == 'ball':
                word = 'sphere'            

            #new idx of word
            new_idx = new_question_token_to_idx[word]
            new_question_dataset[i][j] = new_idx


    question_dataset = new_question_dataset.long()
    # print(question_dataset.type())
    # fdsaf
    question_idx_to_token = new_question_idx_to_token
    question_token_to_idx = new_question_token_to_idx




    train_image_dataset = image_dataset[:n_train]
    train_question_dataset = question_dataset[:n_train]
    # train_answer_dataset = answer_dataset[:50000]

    val_image_dataset = image_dataset[n_train:n_train+n_val]
    val_question_dataset = question_dataset[n_train:n_train+n_val]
    # val_answer_dataset = answer_dataset[50000:60000]

    test_image_dataset = image_dataset[n_train+n_val:]
    test_question_dataset = question_dataset[n_train+n_val:]
    # test_answer_dataset = answer_dataset[60000:]



    # get index of non-small object images
    train_indexes = []
    val_indexes = []


    for i in range(len(train_question_dataset)):
        for j in range(len(train_question_dataset[i])):
            #word
            word = question_idx_to_token[int(train_question_dataset[i][j])]

            if word == 'large':
                train_indexes.append(i)




    for i in range(len(val_question_dataset)):
        for j in range(len(val_question_dataset[i])):
            #word
            word = question_idx_to_token[int(val_question_dataset[i][j])]

            if word == 'large':
                val_indexes.append(i)


    q_max_len = len(train_question_dataset[0])


    return train_image_dataset, train_question_dataset, val_image_dataset, \
             val_question_dataset, test_image_dataset, test_question_dataset, \
                train_indexes, val_indexes, question_idx_to_token, question_token_to_idx, q_max_len, vocab_size







def preprocess_v2(loader, vocab_file):


    # val_loader = ClevrDataLoader(**val_loader_kwargs)
    # question_idx_to_token = train_loader.dataset.vocab['question_idx_to_token']

    print('loader has %d samples' % len(loader.dataset))

    image_dataset = loader.feature_h5['features']
    question_dataset = loader.dataset.all_questions

    # print (len(image_dataset))
    # print (len(question_dataset))
    # fdfas

    vocab = load_vocab(vocab_file)
    question_idx_to_token = vocab['question_idx_to_token']
    question_token_to_idx = vocab['question_token_to_idx']

    # print (question_idx_to_token)




    # Remove useless dimensions
    question_dataset = question_dataset[:,1:10]

    # print (get_sentence(question_dataset[0], question_idx_to_token))
    # fsdasd
    # # print ('vocab size', len(vocab['question_token_to_idx']))

    # new_vocab = ['cube', 'sphere', 'cylinder', 
    #             'cyan', 'red', 'gray', 'blue', 'yellow', 'purple', 'brown', 'green', 
    #             'large', 'small',
    #             'matte', 'metallic']
    # new_question_idx_to_token = {}
    # new_question_token_to_idx = {}
    # for i in range(len(new_vocab)):
    #     new_question_idx_to_token[i] = new_vocab[i]
    #     new_question_token_to_idx[new_vocab[i]] = i

    # vocab_size = len(new_vocab)
    # # vocab_size = len(question_idx_to_token)
    # print ('vocab size:', vocab_size)

    # # Remove useless dimensions
    # question_dataset = question_dataset[:,3:7]

    # # Convert to new vocab / remove synonyms
    # new_question_dataset = torch.zeros(question_dataset.shape[0], question_dataset.shape[1])
    # # Remove synonyms
    # # Block/cube -> cube
    # # Big/large -> large
    # # Small/tiny -> small
    # # Rubber/matte -> matte
    # # Shiny/metallic/metal ->  metallic
    # # Ball/sphere -> sphere

    # for i in range(len(question_dataset)):
    #     for j in range(len(question_dataset[i])):
    #         #word
    #         word = question_idx_to_token[int(question_dataset[i][j])]
    #         if word == 'block':
    #             word = 'cube'
    #         elif word == 'big':
    #             word = 'large'
    #         elif word == 'tiny':
    #             word = 'small'
    #         elif word == 'rubber':
    #             word = 'matte'
    #         elif word == 'shiny' or word == 'metal':
    #             word = 'metallic'
    #         elif word == 'ball':
    #             word = 'sphere'            

    #         #new idx of word
    #         new_idx = new_question_token_to_idx[word]
    #         new_question_dataset[i][j] = new_idx


    # question_dataset = new_question_dataset.long()
    # # print(question_dataset.type())
    # # fdsaf
    # question_idx_to_token = new_question_idx_to_token
    # question_token_to_idx = new_question_token_to_idx


    # vocab_size = len(new_vocab)
    vocab_size = len(question_idx_to_token)
    print ('vocab size:', vocab_size)

    
    # train_image_dataset = image_dataset[:50000]
    # train_question_dataset = question_dataset[:50000]
    # # train_answer_dataset = answer_dataset[:50000]

    # val_image_dataset = image_dataset[50000:60000]
    # val_question_dataset = question_dataset[50000:60000]
    # # val_answer_dataset = answer_dataset[50000:60000]

    # test_image_dataset = image_dataset[60000:]
    # test_question_dataset = question_dataset[60000:]
    # # test_answer_dataset = answer_dataset[60000:]


    train_image_dataset = image_dataset[:n_train]
    train_question_dataset = question_dataset[:n_train]
    # train_answer_dataset = answer_dataset[:50000]

    val_image_dataset = image_dataset[n_train:n_train+n_val]
    val_question_dataset = question_dataset[n_train:n_train+n_val]
    # val_answer_dataset = answer_dataset[50000:60000]

    test_image_dataset = image_dataset[n_train+n_val:]
    test_question_dataset = question_dataset[n_train+n_val:]
    # test_answer_dataset = answer_dataset[60000:]


    # get index of non-small object images
    # train_indexes = []
    # val_indexes = []


    train_indexes = list(range(len(train_question_dataset)))
    val_indexes = list(range(len(val_question_dataset)))

    # for i in range(len(train_question_dataset)):
    #     for j in range(len(train_question_dataset[i])):
    #         #word
    #         word = question_idx_to_token[int(train_question_dataset[i][j])]

    #         if word == 'large':
    #             train_indexes.append(i)




    # for i in range(len(val_question_dataset)):
    #     for j in range(len(val_question_dataset[i])):
    #         #word
    #         word = question_idx_to_token[int(val_question_dataset[i][j])]

    #         if word == 'large':
    #             val_indexes.append(i)



    q_max_len = len(train_question_dataset[0])

    return train_image_dataset, train_question_dataset, val_image_dataset, \
             val_question_dataset, test_image_dataset, test_question_dataset, \
                train_indexes, val_indexes, question_idx_to_token, question_token_to_idx, q_max_len, vocab_size















def preprocess_v3(loader, vocab_file):


    # val_loader = ClevrDataLoader(**val_loader_kwargs)
    # question_idx_to_token = train_loader.dataset.vocab['question_idx_to_token']

    print('loader has %d samples' % len(loader.dataset))

    image_dataset = loader.feature_h5['features']
    question_dataset = loader.dataset.all_questions
    vocab = load_vocab(vocab_file)
    question_idx_to_token = vocab['question_idx_to_token']
    question_token_to_idx = vocab['question_token_to_idx']

    # print (question_idx_to_token
    # Remove useless dimensions


    question_dataset = question_dataset[:,3:7]

    # print (get_sentence(question_dataset[0], question_idx_to_token))
    # fsdaf

    # vocab_size = len(new_vocab)
    vocab_size = len(question_idx_to_token)
    print ('vocab size:', vocab_size)

    
    # train_image_dataset = image_dataset[:50000]
    # train_question_dataset = question_dataset[:50000]
    # # train_answer_dataset = answer_dataset[:50000]

    # val_image_dataset = image_dataset[50000:60000]
    # val_question_dataset = question_dataset[50000:60000]
    # # val_answer_dataset = answer_dataset[50000:60000]

    # test_image_dataset = image_dataset[60000:]
    # test_question_dataset = question_dataset[60000:]


    train_image_dataset = image_dataset[:n_train]
    train_question_dataset = question_dataset[:n_train]
    # train_answer_dataset = answer_dataset[:50000]

    val_image_dataset = image_dataset[n_train:n_train+n_val]
    val_question_dataset = question_dataset[n_train:n_train+n_val]
    # val_answer_dataset = answer_dataset[50000:60000]

    test_image_dataset = image_dataset[n_train+n_val:]
    test_question_dataset = question_dataset[n_train+n_val:]
    # test_answer_dataset = answer_dataset[60000:]

    train_indexes = list(range(len(train_question_dataset)))
    val_indexes = list(range(len(val_question_dataset)))
    q_max_len = len(train_question_dataset[0])

    return train_image_dataset, train_question_dataset, val_image_dataset, \
             val_question_dataset, test_image_dataset, test_question_dataset, \
                train_indexes, val_indexes, question_idx_to_token, question_token_to_idx, q_max_len, vocab_size







