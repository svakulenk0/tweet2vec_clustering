import numpy as np
import lasagne
import theano
import theano.tensor as T
import sys
import batch_char as batch
import cPickle as pkl
import io

from t2v import tweet2vec, init_params, load_params
from settings_char import N_BATCH, MAX_LENGTH, MAX_CLASSES

# import feather
import pandas as pd

def invert(d):
    out = {}
    for k,v in d.iteritems():
        out[v] = k
    return out

def classify(tweet, t_mask, params, n_classes, n_chars):
    # tweet embedding
    emb_layer = tweet2vec(tweet, t_mask, params, n_chars)
    # Dense layer for classes
    l_dense = lasagne.layers.DenseLayer(emb_layer, n_classes, W=params['W_cl'], b=params['b_cl'], nonlinearity=lasagne.nonlinearities.softmax)

    return lasagne.layers.get_output(l_dense), lasagne.layers.get_output(emb_layer)

def generate_embeddings(args):

    data_path = args[0]
    model_path = args[1]
    # save_path = args[2]
    if len(args)>2:
        m_num = int(args[2])

    print("Preparing Data...")
    # Test data
    # read tweet texts into an array
    Xt = []

    # read from file
    with io.open(data_path,'r',encoding='utf-8') as f:
        for line in f:
            Xc = line.rstrip('\n')
            Xt.append(Xc[:MAX_LENGTH])
    print "Tweets:", len(Xt)
    print "Unique tweets:", len(set(Xt))
    # Model
    print("Loading model params...")
    if len(args)>3:
        params = load_params('%s/model_%d.npz' % (model_path,m_num))
    else:
        params = load_params('%s/best_model.npz' % model_path)

    print("Loading dictionaries...")
    with open('%s/dict.pkl' % model_path, 'rb') as f:
        chardict = pkl.load(f)
    with open('%s/label_dict.pkl' % model_path, 'rb') as f:
        labeldict = pkl.load(f)
    n_char = len(chardict.keys()) + 1
    n_classes = min(len(labeldict.keys()) + 1, MAX_CLASSES)
    inverse_labeldict = invert(labeldict)

    print("Building network...")
    # Tweet variables
    tweet = T.itensor3()
    t_mask = T.fmatrix()

    # network for prediction
    predictions, embeddings = classify(tweet, t_mask, params, n_classes, n_char)

    # Theano function
    print("Compiling theano functions...")
    predict = theano.function([tweet,t_mask],predictions)
    encode = theano.function([tweet,t_mask],embeddings)

    # Test
    print("Encoding...")
    out_data = []
    out_pred = []
    out_emb = []
    numbatches = len(Xt)/N_BATCH + 1
    for i in range(numbatches):
        xr = Xt[N_BATCH*i:N_BATCH*(i+1)]
        x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
        p = predict(x,x_m)
        e = encode(x,x_m)
        ranks = np.argsort(p)[:,::-1]
        for idx, item in enumerate(xr):
            out_data.append(item)
            # print [r for r in ranks[idx,:5]]
            # out_pred.append(' '.join([inverse_labeldict[r] for r in ranks[idx,:5]]))
            out_pred.append(ranks[idx,:])
            out_emb.append(e[idx,:])
    return out_emb


    # Save
    # corrected (SV)
    # print("Saving...")
    # with io.open(data_path + '_predicted_tags.txt','wb') as f:
    #     for item in out_pred:
    #         if type(item) == np.ndarray:
    #             item = item.astype('U').tostring()
    #         f.write(item + '\n')

    # readable = []
    # for idx, item in enumerate(out_data):
    #     preds = np.asarray(out_pred)[idx,:10]
    #     plabels = ','.join([labeldict.keys()[ii-1] if ii > 0 else '<unk>' for ii in preds])
    #     readable.append('%s\t%s\n'%(plabels,item))
        # with io.open(data_path + '_predicted_tags.txt','w') as f:
    #     for line in readable:
    #         f.write(line)


def save_embeddings(outfile, modelpath, resultpath, modelepoch=None):
    embeddings = generate_embeddings([outfile, modelpath])
    # print embeddings
    assert embeddings
    print("Saving...")
    with open(resultpath,'w') as f:
        np.save(f,np.asarray(embeddings))
    return embeddings

    # save the matrix in feather format instead (SV)
    # emb_path = save_path + '.feather'
    # Hdense = Hdense.copy()
    # feather.write_dataframe(pd.DataFrame(out_emb), save_path)


if __name__ == '__main__':
    generate_embeddings(sys.argv[1:])
