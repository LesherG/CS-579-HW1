def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def getCIFAR_10(file):
    return unpickle("data/cifar-10-batches-py/" + file)
