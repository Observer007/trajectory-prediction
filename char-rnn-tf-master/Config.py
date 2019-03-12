#coding:utf-8

class Config(object):

    # ======================= param for dataset ============================
    city = 'beijing'
    length = 10
    grids = 300
    grid_size = 0
    # ======================= param for model ==============================
    train_from = 0
    inter_fea = 256                # 0 means not use network embedding, input inter dim
    if city == 'chengdu':        # 0 means not use intra feature, input intra dim
        intra_fea = 33
    elif city == 'shanghai':
        intra_fea = 38
    elif city == 'beijing':
        intra_fea = 34
    inter_units = 256
    if inter_fea>0:
        assert inter_fea == inter_units
    else:
        assert inter_units > 0
    intra_units = 16
    ext_dim = 9              # 9 or 0

    init_scale = 0.04
    learning_rate = 0.0002        # 0.00025 0.003 0.0005 0.0007
    # max_grad_norm = 15      #15
    num_layers = 1
    num_steps = 9          # number of steps to calculate the loss of RNN, max is 10
    if city == 'chengdu':
        hidden_size = 256
    elif city == 'shanghai':
        hidden_size = 128  # size of hi dden layer of neurons
    elif city == 'beijing':
        hidden_size = 256
    max_epoch = 20
    save_after = 10         #The step (counted by the number of iterations) at which the model is saved to hard disk.
    keep_prob = 0.7
    batch_size = 128            # 512 for chengdu 64 for shanghai 128 for beijing
    if city == 'chengdu':
        model_path = '../data/model/chengdu/Model' #the path of model that need to save or load
    elif city == 'shanghai':
        model_path = '../data/model/shanghai/Model'
    elif city == 'beijing':
        model_path = '../data/model/beijing/Model'

    #parameters for generation
    test_epoch = 17             #load save_time saved models
    is_sample = False           #true means using sample, if not using max
    is_beams  = False           #whether or not using beam search
    beam_size = 2               #size of beam search
    len_of_generation = 100     #The number of characters by generated
    given_step = 5
    gen_step = 5
    start_sentence = u'那是因为我看到了另一个自己的悲伤' #the seed sentence to generate text